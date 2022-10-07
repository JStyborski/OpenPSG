import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

# Cosine learning rate decay from lrMax at step 0 to lrMin at step totalSteps
def cosine_annealing(step, totalSteps, lrMax, lrMin):
    return lrMin + (lrMax - lrMin) * 0.5 * (1 + np.cos(step / totalSteps * np.pi))

class BaseTrainer:
    def __init__(self, net: nn.Module, trainLoader: DataLoader, learningRate: float = 0.1, momentum: float = 0.9,
                 weightDecay: float = 0.0005, epochs: int = 100) -> None:
        self.net = net
        self.trainLoader = trainLoader
        self.optimizer = torch.optim.SGD(net.parameters(), learningRate, momentum=momentum,
                                         weight_decay=weightDecay, nesterov=True)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer,
                    lr_lambda=lambda step: cosine_annealing(step, epochs * len(trainLoader), 1, 1e-6 / learningRate))

    def train_epoch(self):
        # Switch model to training mode
        self.net.train()

        # Initialize average loss
        lossAvg = 0.0

        # Hook and helper function to extract outputs from a middle layer of the model and store it in features['feats']
        #features={}
        #def get_features(name):
        #    def hook(model, input, output):
        #        features[name] = output.detach()
        #    return hook
        #self.net.layer4.register_forward_hook(get_features('feats'))

        # Original batching loop
        # train_dataiter = iter(self.train_loader)
        #for train_step in tqdm(range(1, len(train_dataiter) + 1)):
        #    batch = next(train_dataiter)
        #    data = batch['data'].cuda()
        #    target = batch['soft_label'].cuda()

        # Cleaner batching loop
        for imgTens, labelTens in tqdm(self.trainLoader):
            data = imgTens.cuda()
            target = labelTens.cuda()

            # Feedforward and evaluate BCE loss
            logits = self.net(data)
            loss = F.binary_cross_entropy_with_logits(logits, target, reduction='sum')

            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            # EMA on avg loss to show smooth values
            with torch.no_grad():
                lossAvg = lossAvg * 0.8 + float(loss) * 0.2

        return {'train_loss': lossAvg}
