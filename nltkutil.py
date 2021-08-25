import os
import numpy as np
import pandas as pd
import nltk
#nltk.download('punkt')
from nltk.stem.porter import PorterStemmer

def tokenize(sentence):
    return nltk.word_tokenize(sentence)

def stem(word):
    return PorterStemmer().stem(word.lower())

def bag_of_words(sentence, allwords):
    bag = np.zeros(len(allwords), dtype=np.float32)
    for index, word in enumerate(allwords):
        if word in sentence:
            bag[index] = 1.0
    return bag
