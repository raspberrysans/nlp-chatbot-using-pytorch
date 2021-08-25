import random
import os
import json
from nltk import probability
import torch
from model import NeuralNetwork
from nltkutil import bag_of_words, tokenize, stem

with open('intents.json', 'r') as f:
    intents = json.load(f)

FILE = "data.pth"
data = torch.load(FILE)
inputsize = data['input_size']
hiddensize = data['hidden_size']
outputsize = data['output_size']
allwords = data['all_words']
tags = data['tags']
modelstate = data['model_state']
model = NeuralNetwork(inputsize, hiddensize, outputsize)
model.load_state_dict(modelstate)
model.eval()
ignore = ['?', '!', '.', ',']
botname = "raspberrybot"
print("start the chat!")
while True:
    sentence = input('You: ')
    sentence = tokenize(sentence)
    sentence = [stem(word) for word in sentence if word not in ignore]
    bag = bag_of_words(sentence, allwords)
    bag = bag.reshape(1, bag.shape[0])
    bag = torch.from_numpy(bag)
    output = model(bag)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]

    problist = torch.softmax(output, dim=1)
    prob = problist[0][predicted.item()]
    if prob >= 0.85:
        for i in intents["intents"]:
            if tag == i["tag"]:
                print(botname+": "+ random.choice(i['responses']))
    
    else:
        print(botname+": okay!")
