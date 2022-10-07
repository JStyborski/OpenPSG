import json
import numpy as np
import logging
import os
from tqdm import tqdm
from PIL import Image, ImageFile

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import clip

###############################
# User Input and Model Import #
###############################

modelName = 'clipViT'

if modelName == 'clipRN':
    clipModel, clipPreproc = clip.load('RN50', device='cuda', jit=False)
elif modelName == 'clipViT':
    clipModel, clipPreproc = clip.load('ViT-B/16', device='cuda', jit=False)

##########################
# Define Custom Datasets #
##########################

# to fix "OSError: image file is truncated"
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Note I eliminated the custom transformation composition because clipPreproc already does something similar
# clipPreproc returns a transformation composition with resize, center crop, RGB convert, ToTensor, and normalize
# The only thing missing is the random horizontal flip and random crop

class CustomDataset(Dataset):
    def __init__(self, stage, root='./data/coco/', numClasses=56):
        super(CustomDataset, self).__init__()

        # Import dataset - a dict with keys: data, predicate_classes, test_image_ids, train_image_ids, val_image_ids
        with open('./data/psg/psg_cls_basic.json') as f:
            dataset = json.load(f)

        # imglist is a list of dicts for each image in a dataset - each dict contains image_id, file_name, and relations
        self.imgList = [d for d in dataset['data'] if d['image_id'] in dataset[f'{stage}_image_ids']]
        self.root = root
        self.numClasses = numClasses

    def __len__(self):
        return len(self.imgList)

    def __getitem__(self, index):
        # sampleDict is a dict containing 1 image's info, keys as image_id, file_name, and relations (class indices)
        sampleDict = self.imgList[index]
        path = os.path.join(self.root, sampleDict['file_name'])

        # Open image and transform it with torch transformations
        try:
            with Image.open(path) as image:
                outputImgTens = clipPreproc(image)
        except Exception as e:
            logging.error('Error, cannot read [{}]'.format(path))
            raise e

        # Generate label - output is a tensor of size numClasses and 1's at indices corresponding to 'relations'
        labelTens = torch.Tensor(self.numClasses).fill_(0)
        labelTens[sampleDict['relations']] = 1

        return outputImgTens, labelTens


# Loading datasets - no training set since we're testing Zero-Shot
#trainDataset = CustomDataset(stage='train')
#trainDataloader = DataLoader(trainDataset, batch_size=batchSize, shuffle=True, num_workers=0) # Originally 8 workers
valDataset = CustomDataset(stage='val')
valDataloader = DataLoader(valDataset, batch_size=32, shuffle=False, num_workers=0)
testDataset = CustomDataset(stage='test')
testDataloader = DataLoader(testDataset, batch_size=32, shuffle=False, num_workers=0)
print('Data Loaded...', flush=True)

######################
# CLIP Text Encoding #
######################

# Define the class "labels" for the relations and use CLIP to tokenize them
categList = ['over', 'in front of', 'beside', 'on', 'in', 'attached to', 'hanging from', 'on back of', 'falling off',
             'going down', 'painted on', 'walking on', 'running on', 'crossing', 'standing on', 'lying on',
             'sitting on', 'flying over', 'jumping over', 'jumping from', 'wearing', 'holding', 'carrying',
             'looking at', 'guiding', 'kissing', 'eating', 'drinking', 'feeding', 'biting', 'catching', 'picking',
             'playing with', 'chasing', 'climbing', 'cleaning', 'playing', 'touching', 'pushing', 'pulling', 'opening',
             'cooking', 'talking to', 'throwing', 'slicing', 'driving', 'riding', 'parked on', 'driving on',
             'about to hit', 'kicking', 'swinging', 'entering', 'exiting', 'enclosing', 'leaning on']
categTokenTens = clip.tokenize(categList).cuda()

# Use CLIP to encode class label text and then normalize each
txtEncTens = clipModel.encode_text(categTokenTens).detach()
txtEncNormdTens = txtEncTens / txtEncTens.norm(dim=-1, keepdim=True)

print('Text Encoded...', flush=True)

#############################
# CLIP Zero-Shot Evaluation #
#############################

# Initialize prediction and ground-truth category lists for comparison later
predList, gtList = [], []

with torch.no_grad():

    # Process validation images with a dataloader
    for imgTens, labelTens in tqdm(valDataloader):

        # Encode images and then normalize each
        imgEncTens = clipModel.encode_image(imgTens.cuda())
        imgEncNormdTens = imgEncTens / imgEncTens.norm(dim=-1, keepdim=True)

        # Compute the similarity measure between batch of images and text labels
        innerProdSoft = (100.0 * torch.inner(imgEncNormdTens, txtEncNormdTens)).softmax(dim=-1)

        # Find the top 3 predictions and return their indices, add indices to predList
        simRankIndex = innerProdSoft.topk(3)[1].cpu().detach().tolist()
        predList.extend(simRankIndex)  # Add pred to predList for each batch

        # Gather list of truth labels for each sample
        for truthLabel in labelTens:
            gtLabel = (truthLabel == 1).nonzero(as_tuple=True)[0].cpu().detach().tolist()  # List of indices where multihot truth labels are 1
            gtList.append(gtLabel)  # Add gtLabel to gtList for each sample in each batch

print('Validation Set Evaluated...', flush=True)

#######################
# Compute Mean Recall #
#######################

# Compute mean recall
# scoreList has 2 columns: col 0 for counting all truth classes, col 1 for counting classes that are common
# gt and pred are each lists of class indices for each sample. gt is truth, pred is topk prediction
scoreList = np.zeros([56, 2], dtype=int)
for gt, pred in zip(gtList, predList):
    for gtIdx in gt:
        scoreList[gtIdx][0] += 1
        if gtIdx in pred:
            scoreList[gtIdx][1] += 1

# Remove the top 6 categories since they are common
scoreList = scoreList[6:]

# Smoothing, convert any 0 classes to 1 in the truth classes counter to avoid dividing by zero
scoreList[:, 0][scoreList[:, 0] == 0] = 1
meanRecall = np.mean(scoreList[:, 1] / scoreList[:, 0])

print('Mean Recall Computed...', flush=True)

#######################
# Test Set Prediction #
#######################

testPredList = []
with torch.no_grad():
    for imgTens, labelTens in testDataloader:

        # Encode images and then normalize each
        imgEncTens = clipModel.encode_image(imgTens.cuda())
        imgEncNormdTens = imgEncTens / imgEncTens.norm(dim=-1, keepdim=True)

        # Compute the similarity measure between batch of images and text labels
        innerProdSoft = (100.0 * torch.inner(imgEncNormdTens, txtEncNormdTens)).softmax(dim=-1)

        # Find the top 3 predictions and return their indices, add indices to predList
        simRankIndex = innerProdSoft.topk(3)[1].cpu().detach().tolist()
        testPredList.extend(simRankIndex)  # Add pred to predList for each batch

# Returns the list of predicted classes for the test set and then write to file
with open(f'results/ZeroShot_{modelName}_{meanRecall}.txt', 'w') as writer:
    for labelList in testPredList:
        a = [str(x) for x in labelList]
        save_str = ' '.join(a)
        writer.writelines(save_str + '\n')
print('Result Saved!', flush=True)