#import io
import json
import logging
import os
from PIL import Image, ImageFile

import torch
import torchvision.transforms as trn
from torch.utils.data import Dataset

# to fix "OSError: image file is truncated"
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Note there used to be a "Convert" class here that was used to convert image.convert('RGB')
# It was called as the first item in the trn.Compose below.
# I removed the class and the trn.Compose call to Convert because the image is already converted to RGB in __getitem__

def get_transforms(stage: str, resolution=(224, 224)):

    # mean and stdev used for normalizing each color channel of an image
    mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]

    # Prepare a list of transformations to be applied (in order listed) to an image
    if stage == 'train':
        return trn.Compose([trn.Resize(resolution),
                            trn.RandomHorizontalFlip(),
                            trn.RandomCrop(resolution, padding=4),
                            trn.ToTensor(),
                            trn.Normalize(mean, std)])
    elif stage in ['val', 'test']:
        return trn.Compose([trn.Resize(resolution),
                            trn.ToTensor(),
                            trn.Normalize(mean, std)])

class PSGClsDataset(Dataset):
    def __init__(self, stage, resolution, root='./data/coco/', numClasses=56):
        super(PSGClsDataset, self).__init__()

        # Import dataset - a dictionary with keys: data, predicate_classes, test_image_ids, train_image_ids, val_image_ids
        with open('./data/psg/psg_cls_basic.json') as f:
            dataset = json.load(f)

        # imglist is a list of dicts for each image in a dataset - each dict contains image_id, file_name, and relations
        self.imgList = [d for d in dataset['data'] if d['image_id'] in dataset[f'{stage}_image_ids']]
        self.root = root
        self.transformImage = get_transforms(stage, resolution)
        self.numClasses = numClasses

    def __len__(self):
        return len(self.imgList)

    def __getitem__(self, index):
        # sampleDict is a dict containing a single image's info, keys as image_id, file_name, and relations (class indices)
        sampleDict = self.imgList[index]
        path = os.path.join(self.root, sampleDict['file_name'])

        # Open image and transform it with torch transformations
        try:
            with Image.open(path).convert('RGB') as image:
                outputImgTens = self.transformImage(image)

            #with open(path, 'rb') as f:
            #    content = f.read()
            #    filebytes = content
            #    buff = io.BytesIO(filebytes)
            #    image = Image.open(buff).convert('RGB')
            #    sample['data'] = self.transform_image(image)
        except Exception as e:
            logging.error('Error, cannot read [{}]'.format(path))
            raise e

        # Generate label - output is a tensor of size numClasses and 1's at indices corresponding to 'relations'
        labelTens = torch.Tensor(self.numClasses).fill_(0)
        labelTens[sampleDict['relations']] = 1

        return outputImgTens, labelTens


