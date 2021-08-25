import torch
import torch.nn as nn
class NeuralNetwork(nn.Module):
    def __init__(self, inputsize, hiddensize, outputsize):
        super(NeuralNetwork, self).__init__()
        self.layer1 = nn.Linear(inputsize, hiddensize)
        self.layer2 = nn.Linear(hiddensize, hiddensize)
        self.layer3 = nn.Linear(hiddensize, outputsize)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        out = self.layer1(x)
        out = self.relu(out)
        out = self.layer2(out)
        out = self.relu(out)
        out = self.layer3(out)
        return out

