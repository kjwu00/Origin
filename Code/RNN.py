from os import listdir
from os.path import isfile, join
import pandas as pd
import collections
import random
import torch
import time
import torch.nn as nn
from random import randint
import joblib 
import pickle
import torch.optim as optim

NUM_WORDS = 100
NUM_LANGUAGES = 6 #hardcoded how to get language tensor
NUM_DATAPOINTS = 1000
languages = ['en', 'es', 'fr','ko', 'ru', 'pt', 'gr', 'id', 'it']
trainPath = 'Test_Train'
trainFiles = [f for f in listdir(trainPath) if isfile(join(trainPath, f))]
trainFiles.sort()

all_words = None
with open("allWords.txt", "rb") as fp:   
    all_words = pickle.load(fp)

n_words = len(all_words)
n_hidden = 1000
n_categories = 6

def wordToIndex(word):
    return all_words.index(word)

def textToTensor(text):
    tensor = torch.zeros(len(text), 1, n_words)
    for li, word in enumerate(text):
        tensor[li][0][wordToIndex(word)] = 1
    return tensor

def getTrainSample(file):
    filename = None
    try:
        filename = trainFiles[file]
    except:
        print(file)
    language = filename.split("_")
    filename = trainPath + "/" + filename
    language = language[1]
    languageTens = None

    languageTens = torch.tensor([languages.index(language)], dtype=torch.long)
    with open(filename, 'r') as openfile:
        strText = openfile.readlines()
    openfile.close()
    strText = strText[0]
    text = strText.split(" ")
    textTens = textToTensor(text)
    return language, text, languageTens, textTens

   
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)
    
def categoryFromOutput(output):
    top_n, top_i = output.topk(1)
    category_i = top_i[0].item()
    return languages[category_i], category_i

rnn = RNN(n_words, n_hidden, n_categories)
#rnn.cuda()
criterion = nn.NLLLoss()

def train(category_tensor, text_tensor):
    hidden = rnn.initHidden()
    rnn.zero_grad()
    #optimizer.zero_grad()

    #for i in range(text_tensor.size()[0]):
    for i in range(100):
        output, hidden = rnn(text_tensor[i], hidden)
    print(output)
    print(category_tensor)
    loss = criterion(output, category_tensor)
    loss.backward()
    #optimizer.step()

    # Add parameters' gradients to their values, multiplied by learning rate
    for p in rnn.parameters():
        if p.grad is not None:
            learning_rate = 0.05
            p.data.add_(-learning_rate, p.grad.data)
    print()
    print("i2h gradient")
    print(rnn.i2h.weight.data)
    print("i2h weight")
    print(rnn.i2h.weight.data)
    print()
    return output, loss.item()

import math

#n_iters = trainSize
n_iters = NUM_LANGUAGES*NUM_DATAPOINTS*0.8
print(n_iters)
print(len(trainFiles))
print_every = 100
plot_every = 100

# Keep track of losses for plotting
current_loss = 0
all_losses = []

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

start = time.time()
total_correct = 0
total = 0
for i in range(0, 1000): # int(n_iters)
    category, line, category_tensor, line_tensor = getTrainSample(i)  
    output, loss = train(category_tensor, line_tensor)
    current_loss += loss
    total += 1
    if categoryFromOutput(output)[0] == category:
        total_correct += 1
    
    # Print iter number, loss, name and guess
    if i % print_every == 0:
        guess, guess_i = categoryFromOutput(output)
        correct = '✓' if guess == category else '✗ (%s)' % category
        accuracy = total_correct/ total
        #print('%d %d%% (%s) %.4f %s / %s %s' % (i, timeSince(start), loss, line, guess, correct))
        print(str(i) + " " + str(timeSince(start)) + ": " + str(accuracy) + ", " + str(guess) + ", " + str(correct))

    # Add current loss avg to list of losses
    if i % plot_every == 0:
        all_losses.append(current_loss / plot_every)
        current_loss = 0

joblib.dump(rnn, 'RNNmodel.pkl') 
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

plt.figure()
plt.plot(all_losses)

"""
Validation
"""

valPath = 'Val'
valFiles = [f for f in listdir(valPath) if isfile(join(valPath, f))]
valFiles.sort()

def evaluate(text_tensor):
    hidden = rnn.initHidden()
    for i in range(10):
        output, hidden = rnn(text_tensor[i], hidden)
    return output

def predict(text, n_predictions=1):
    with torch.no_grad():
        output = evaluate(text)

        # Get top N categories
        topv, topi = output.topk(n_predictions, 1, True)
        languageTens = torch.tensor([topi], dtype=torch.long)
        return languageTens


right = 0
total = 0

perlangage = [0] * NUM_LANGUAGES
def getValSample(file):
    filename = valFiles[file]
    language = filename.split("_")
    filename = valPath + "/"+ filename
    language = language[1]
    languageTens = None
    lanIndex = None
    if language == languages[0]:
        languageTens = torch.tensor([0], dtype=torch.long)
    elif language == languages[1]:
        languageTens = torch.tensor([1], dtype=torch.long)
    elif language == languages[2]:
        languageTens = torch.tensor([2], dtype=torch.long)
    elif language == languages[3]:
        languageTens = torch.tensor([3], dtype=torch.long)
    elif language == languages[4]:
        languageTens = torch.tensor([4], dtype=torch.long)
    elif language == languages[5]:
        languageTens = torch.tensor([5], dtype=torch.long)

    with open(filename, 'r') as openfile:
        strText = openfile.readlines()
    openfile.close()
    strText = strText[0]
    text = strText.split(" ")
    textTens = textToTensor(text)
    return language, text, languageTens, textTens

perlanguageRight = [0, 0, 0, 0, 0, 0]
perlanguageTotal = [0, 0, 0, 0, 0, 0]
for i in range(int(NUM_DATAPOINTS*0.1)):
    text = None
    language = None
    language, text, languageTens, textTens = getValSample(i)
    
    output = predict(textTens)
    if output == languageTens:
        right += 1
        #perlanguage[lanIndex] += 1
    total += 1
    index = languages.index(language)
    perlanguageTotal[index] += 1

print("testing accuracy: " + str(right/total))
for i in range (6):
    percent = perlanguageRight / perlanguageTotal
    print("language " + str(i) + ": " + str(accuracy))
