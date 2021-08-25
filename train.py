import json
from nltkutil import tokenize, stem, bag_of_words
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from model import NeuralNetwork

with open('intents.json', 'r') as f:
    intents = json.load(f)

allwords = []
tags = []
patterntag = []
for i in intents['intents']:
    tag = i["tag"]
    tags.append(tag)
    for p in i["patterns"]:
        w = tokenize(p)
        allwords.extend(w)
        patterntag.append((w, tag))
ignore = ['?', '!', '.', ',']
allwords = [stem(word) for word in allwords if word not in ignore]
allwords = sorted(set(allwords))
tags = sorted(tags)
X_train = []
Y_train = []
for sentence, tag in patterntag:
    sentence = [stem(word) for word in sentence if word not in ignore]
    bag = bag_of_words(sentence, allwords)
    #print(bag, sentence)
    X_train.append(bag)

    label = tags.index(tag)
    #for crossentropyloss
    Y_train.append(label)

X_train = np.array(X_train)
Y_train = np.array(Y_train)


class chatDataset(Dataset):
    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = Y_train

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]
    
    def __len__(self):
        return self.n_samples
    
batchsize = 8
inputsize = len(allwords)
hiddensize = 8
outputsize = len(tags)
learningrate = 0.011
numepochs = 1000

dataset = chatDataset()
train_loader = DataLoader(dataset=dataset, batch_size = batchsize, shuffle=True)
model = NeuralNetwork(inputsize, hiddensize, outputsize)
#loss and optimizer parameters
loss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = learningrate)

for epoch in range(numepochs):
    for(words, labels) in train_loader:
        outputs = model(words) 
        #print(outputs)
        l = loss(outputs, labels)
        #backward pass, optimizer
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
    
    if (epoch + 1) %100 ==0:
        print (f'Epoch [{epoch+1}/{numepochs}], Loss: {l.item():.4f}')



data = {
"model_state": model.state_dict(),
"input_size": inputsize,
"hidden_size": hiddensize,
"output_size": outputsize,
"all_words": allwords,
"tags": tags
}

FILE = "data.pth"
torch.save(data, FILE)
