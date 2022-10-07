import argparse
import os
import time

from Dataset import PSGClsDataset
from Evaluator import Evaluator
from Trainer import BaseTrainer

import torch
from torch.utils.data import DataLoader
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models import wide_resnet50_2, Wide_ResNet50_2_Weights
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights
from torchvision.models import vit_b_16, ViT_B_16_Weights

parser = argparse.ArgumentParser()
parser.add_argument('--modelName', type=str, default='res50')
parser.add_argument('--allTrain', type=bool, default=True)
parser.add_argument('--res', type=tuple, default=(224, 224))
parser.add_argument('--epoch', type=int, default=60)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--batchSize', type=int, default=16)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--weightDecay', type=float, default=0.0005)
args = parser.parse_args()

saveName = f'{args.modelName}_at{args.allTrain}_res{args.res}_ep{args.epoch}_lr{args.lr}_bs{args.batchSize}_mom{args.momentum}_wd{args.weightDecay}'
checkpointsDir = './checkpoints'
resultsDir = './results'
os.makedirs(checkpointsDir, exist_ok=True)
os.makedirs(resultsDir, exist_ok=True)

# Loading datasets
trainDataset = PSGClsDataset(stage='train', resolution=args.res)
trainDataloader = DataLoader(trainDataset, batch_size=args.batchSize, shuffle=True, num_workers=0) # Originally 8 workers
valDataset = PSGClsDataset(stage='val', resolution=args.res)
valDataloader = DataLoader(valDataset, batch_size=32, shuffle=False, num_workers=0)
testDataset = PSGClsDataset(stage='test', resolution=args.res)
testDataloader = DataLoader(testDataset, batch_size=32, shuffle=False, num_workers=0)
print('Data Loaded...', flush=True)

# Load model
def define_model(modelName='res18', allTrain=True):

    # Load model weights and model
    if modelName == 'res18':
        weights = ResNet18_Weights.DEFAULT
        model = resnet18(weights=weights)
    elif modelName == 'res50':
        weights = ResNet50_Weights.DEFAULT
        model = resnet50(weights=weights)
    elif modelName == 'wrn50':
        weights = Wide_ResNet50_2_Weights.DEFAULT
        model = wide_resnet50_2(weights=weights)
    elif modelName == 'effNetS':
        weights = EfficientNet_V2_S_Weights.DEFAULT
        model = efficientnet_v2_s(weights=weights)
    elif modelName == 'vitB16':
        weights = ViT_B_16_Weights.DEFAULT
        model = vit_b_16(weights=weights)

    # Freeze layers if we don't want to train everything
    if not(allTrain):
        for p in model.parameters(): p.requires_grad = False

    # Custom bad code for freezing certain ResNet layers
    # ResNet50 has 10 children: conv1, bn1, relu, maxpool, layer1, layer2, layer3, layer4, avgpool, fc
    # The Seq (layer#) blocks contain 3, 4, 6, 3 bottleneck blocks (respectively), with each bottleneck block containing 3-4 convolutions
    #freezeN = 7 # Number of children to freeze (not 0-indexed, just straight up number)
    #childCount = 0 # Initialize counter
    #for child in model.children():
    #    childCount += 1
    #    if childCount > freezeN:
    #        break
    #    for param in child.parameters(): param.requires_grad = False

    # Custom bad code for freezing certain ViT layers
    # ViT-B16 has 3 children: conv_proj, encoder, heads
    # The ViT-B16 model.encoder has 3 children: dropout, layers, ln
    # The ViT-B16 model.encoder.layers has 12 children: encoder_layer_# for 0-11
    #for param in model.conv_proj.parameters(): param.requires_grad = False
    #for param in model.encoder.dropout.parameters(): param.requires_grad = False
    #freezeN = 9  # Number of children to freeze (not 0-indexed, just straight up number)
    #childCount = 0  # Initialize counter
    #for child in model.encoder.layers.children():
    #    childCount += 1
    #    if childCount > freezeN:
    #        break
    #    for param in child.parameters(): param.requires_grad = False

    # Replace last layer
    if modelName == 'res18':
        model.fc = torch.nn.Linear(512, 56)
    elif modelName == 'res50' or modelName == 'wrn50':
        model.fc = torch.nn.Linear(2048, 56)
    elif modelName == 'effNetS':
        model.classifier = torch.nn.Linear(1280, 56)
        #model.classifier[-1] = torch.nn.Linear(1280, 56)
    elif modelName == 'vitB16':
        model.heads.head = torch.nn.Linear(768, 56)

    return model

model = define_model(modelName=args.modelName, allTrain=args.allTrain)
model.cuda()
print('Model Loaded...', flush=True)

def count_params(model):
    trainableParams = sum(p.numel() for p in model.parameters() if p.requires_grad)
    allParams = sum(p.numel() for p in model.parameters())
    return trainableParams, allParams
print(count_params(model))

# Loading trainer
trainer = BaseTrainer(model, trainDataloader, learningRate=args.lr, momentum=args.momentum,
                      weightDecay=args.weightDecay, epochs=args.epoch)
evaluator = Evaluator(model, k=3)

# Training and evaluation loop
if __name__ == '__main__':

    print('Start Training...', flush=True)
    beginEpoch = time.time()
    bestValRecall = 0.0

    for epoch in range(0, args.epoch):

        # Train model for one epoch
        trainMetrics = trainer.train_epoch()

        # Returns average loss and mean recall for the validation set
        valMetrics = evaluator.eval_recall(valDataloader)

        # Print log
        print('{} | Epoch {:3d} | Time {:5d}s | Train Loss {:.4f} | Val Loss {:.3f} | mR {:.2f}'
              .format(saveName, (epoch + 1), int(time.time() - beginEpoch),
                      trainMetrics['train_loss'], valMetrics['avg_loss'], 100.0 * valMetrics['mean_recall']), flush=True)

        # Save best model so far and update best recall value
        if valMetrics['mean_recall'] >= bestValRecall:
            torch.save(model.state_dict(), checkpointsDir + f'/{saveName}_best.ckpt')
            bestValRecall = valMetrics['mean_recall']

    # Save the final model
    torch.save(model.state_dict(), checkpointsDir + f'/{saveName}_last.ckpt')

    print('Training Completed...', flush=True)

    # Load the best checkpoint and initialize the validation/test evaluator
    print('Loading Best Ckpt...', flush=True)
    checkpoint = torch.load(checkpointsDir + f'/{saveName}_best.ckpt')
    model.load_state_dict(checkpoint)
    evalValTest = Evaluator(model, k=3)

    # Evaluate the model for recall on the validation dataset and print
    valMetrics = evalValTest.eval_recall(valDataloader)
    if bestValRecall == valMetrics['mean_recall']:
        print('Successfully load best checkpoint with acc {:.2f}'.format(100 * bestValRecall), flush=True)
    else:
        print('Fail to load best checkpoint')

    # Returns the list of predicted classes for the test set and then write to file
    testResult = evalValTest.predict_classes(testDataloader)
    with open(resultsDir + f'/{saveName}_{bestValRecall}.txt', 'w') as writer:
        for labelList in testResult:
            a = [str(x) for x in labelList]
            save_str = ' '.join(a)
            writer.writelines(save_str + '\n')
    print('Result Saved!', flush=True)
