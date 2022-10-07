import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

class Evaluator:
    # Initialize Evaluator with nn model and k value (for k top class labels)
    def __init__(self, net: nn.Module, k: int):
        self.net = net
        self.k = k

    # Take truth/pred classes and compute recall
    def eval_recall(self, dataLoader: DataLoader):
        # Switch model to eval mode
        self.net.eval()

        # Initialize loss average and prediction and ground truth lists
        lossAvg = 0.0
        predList, gtList = [], []

        # Loop through batches and compute predictions and prediction recall and loss
        with torch.no_grad():
            for imgTens, labelTens in dataLoader:
                data = imgTens.cuda() # Tensor [BS, C, H, W] Validation input data
                logits = self.net(data) # Tensor [BS, Classes] Validation data linear outputs
                prob = torch.sigmoid(logits) # Tensor [BS, Classes] Sigmoid of linear outputs
                target = labelTens.cuda() # Tensor [BS, Classes] Multihot of ground truth classes
                loss = F.binary_cross_entropy(prob, target, reduction='sum') # Tensor [] Single value of loss
                lossAvg += float(loss.data) # Accumulate loss across batches

                # Gather the top k results of the prediction
                predLabel = torch.topk(prob.data, self.k)[1].cpu().detach().tolist() # List of lists [BS, k value] Top k results of the sigmoid prediction
                predList.extend(predLabel) # Add pred to predList for each batch

                # Gather list of truth labels for each sample
                for truthLabel in labelTens:
                    gtLabel = (truthLabel == 1).nonzero(as_tuple=True)[0].cpu().detach().tolist() # List of indices where multihot truth labels are 1
                    gtList.append(gtLabel) # Add gtLabel to gtList for each sample in each batch

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

        # Return avg loss and mean recall
        return {'avg_loss': lossAvg / len(dataLoader), 'mean_recall': meanRecall}

    # Similar to eval_recall, but only returning list of lists of predicted classes
    def predict_classes(self, dataLoader: DataLoader):
        self.net.eval()

        predList = []
        with torch.no_grad():
            for imgTens, labelTens in dataLoader:
                data = imgTens.cuda()
                logits = self.net(data)
                prob = torch.sigmoid(logits)
                predLabel = torch.topk(prob.data, self.k)[1].cpu().detach().tolist()
                predList.extend(predLabel)
        return predList
