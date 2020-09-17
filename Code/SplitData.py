from os import listdir
import os
from os.path import isfile, join
import pandas as pd
import collections
import random
import torch
import time
import torch.nn as nn
from random import randint
import pickle 
from sklearn.utils import shuffle

mypath = '6Lang'
files = [f for f in listdir(mypath) if isfile(join(mypath, f))]
mappings = collections.defaultdict(list)

NUM_WORDS = 100 # original - 100
NUM_LANGAUGES = 6
NUM_DATAPOINTS = 2000 # original - 2000
percentData = 0.5 #original - 0.1

def clean_line(line):
    cleaned_line = line.replace('\n', '')
    cleaned_line = line.replace('*', '')
    cleaned_line = line.replace('\xa0', ' ')
    cleaned_line = line.replace('\xa0\xa0\xa0\u2028', ' ')

    return cleaned_line

# returns dict for a language - mapping {language_name : list of 100 words}
def create_word_dict(language_name):
    all_words = []
    bigram = []
    trigram = []

    for file_name in files:
        if file_name[0:2] != language_name:
            continue
        else:
            pathname = mypath + '/' + file_name
            with open(pathname, 'r') as openfile:
                data = openfile.readlines()
                word_list = []
                for line in data:
                    line = line.lower()
                    line = line.replace('\n', '')
                    line = line.replace('*', '')
                    line = line.replace('\xa0', ' ')
                    line = line.replace('\u2028', '')
                    line = line.replace('\\', '')
                    line = line.replace('\"', '')
                    line = line.replace(',', '')
                    line = line.replace('.', '')
                    line = line.replace(':', '')
                    line = line.replace('_', '')
                    line = line.replace('-', '')
                    line = line.replace(';', '')
                    #line = line.replace('')

                    word_list = line.split(' ')
                    for word in word_list:
                        if word != '' and word != ' ':
                            all_words.append(word)
                   
            evenly_split_list = []
            temp_list = []
            counter = 0
            for word in all_words:
                if len(temp_list) == NUM_WORDS:
                    evenly_split_list.append(temp_list)
                    temp_list = []
                else:
                    temp_list.append(word)

            mappings[language_name] = evenly_split_list

    return all_words, bigram, trigram

languages = ['en', 'es', 'fr','ko', 'ru', 'pt', 'gr', 'id', 'it']
languages = languages[:NUM_LANGAUGES]
all_words = set({})
bigram_words = set({})
trigram_words = set({})

for language in languages:
    all_words_lang, bigram, trigram = create_word_dict(language)
    
    for word in all_words_lang:
        all_words.add(word)
    for word in bigram:
        bigram_words.add(word)
    for word in trigram:
        trigram_words.add(word)

all_words = list(all_words)
with open("allWords.txt", "wb") as fp:
    pickle.dump(all_words, fp)

# MAKE EACH LANGUAGE SAME SIZE
for language in mappings.keys():
    random.shuffle(mappings[language])
    mappings[language] = (mappings[language])[:NUM_DATAPOINTS]

trainSize = NUM_DATAPOINTS * 0.8 * NUM_LANGAUGES * percentData #original: 0.8 -> change to 0.5
trainSize = int(trainSize)
trainSet = [i for i in range(int(trainSize))]
random.shuffle(trainSet)
trainPercent = 0.8
valPercent = 0.1
testPercent = 0.1

seen_indices = collections.defaultdict(set)


# remove old files
filelist_train = [ f for f in os.listdir("Test_Train/") ]
for f in filelist_train:
    os.remove(os.path.join("Test_Train/", f))
    
filelist_val = [ f for f in os.listdir("Test_Val/") ]
for f in filelist_val:
    os.remove(os.path.join("Test_Val/", f))
    
filelist_test = [ f for f in os.listdir("Test_Test/") ]
for f in filelist_test:
    os.remove(os.path.join("Test_Test/", f))
    

def getTrainSample(x):
    print('call go getTrainSample')
    i = x // (NUM_DATAPOINTS*trainPercent*percentData) 
    if i > 6:
        print(i)
    language = languages[int(i)]
    len_lang_list = len(mappings[language])
    
    index = random.randint(0, len_lang_list - 1)
    while index in seen_indices[language]:
        index = random.randint(0, len_lang_list - 1)
    
    text = mappings[language][index]
    seen_indices[language].add(index)
    
    return language, text


seen = set({})
for counter in range(len(trainSet)):
    print('iteration of train for loop')
    i = random.randint(0, len(trainSet)-1)
    while i in seen:
        i = random.randint(0, len(trainSet)-1)
    seen.add(i)
    
    x = trainSet[i]
    language, text = getTrainSample(x)
    filename = "Test_Train/" + str(i) + "_" + language
    print(filename)
    f = open(filename,"w+")
    text = " ".join(text)
    f.write(text)
    f.close()

print('Done adding to train')
    
valSize = NUM_DATAPOINTS * (trainPercent + valPercent) * NUM_LANGAUGES * percentData 
testSize = NUM_DATAPOINTS * (trainPercent + valPercent + testPercent) * NUM_LANGAUGES * percentData
valSize = int(valSize)
testSize = int(testSize)

def getValSample(x):
    i = (x - (NUM_DATAPOINTS*trainPercent* NUM_LANGAUGES*percentData)) // (NUM_DATAPOINTS*percentData*valPercent) # original: 0.1 -> change to 0.4
    if i > 6:
        print(i)
    language = languages[int(i)]
    len_lang_list = len(mappings[language])
    
    index = random.randint(0, len_lang_list - 1)
    while index in seen_indices[language]:
        index = random.randint(0, len_lang_list - 1)
    
    text = mappings[language][index]
    seen_indices[language].add(index)
    
    return language, text


def getTestSample(x):
    i = (x - (NUM_DATAPOINTS*(trainPercent + valPercent)* NUM_LANGAUGES*percentData)) // (NUM_DATAPOINTS*percentData*testPercent)
    if i > 6:
        print(i)
    language = languages[int(i)]
    len_lang_list = len(mappings[language])
    
    index = random.randint(0, len_lang_list - 1)
    while index in seen_indices[language]:
        index = random.randint(0, len_lang_list - 1)
    
    text = mappings[language][index]
    seen_indices[language].add(index)
    
    return language, text

valSet = [i for i in range(trainSize, valSize)]
random.shuffle(valSet)
testSet = [i for i in range(valSize, testSize)]
random.shuffle(testSet)

seen = set({})
for counter in range(len(valSet)):
    i = random.randint(0, len(valSet)-1)

    while i in seen:
        i = random.randint(0, len(valSet)-1)
    seen.add(i)
    x = valSet[i]
    language, text = getValSample(x)

    filename = "Test_Val/" + str(i) + "_" + language
    print(filename)

    f = open(filename,"w+")
    text = " ".join(text)
    
    f.write(text)
    f.close()
    
seen = set({})
for counter in range(len(testSet)):
    i = random.randint(0, len(testSet)-1)

    while i in seen:
        i = random.randint(0, len(testSet)-1)
    seen.add(i)
    x = testSet[i]
    language, text = getTestSample(x)

    filename = "Test_Test/" + str(i) + "_" + language
    print(filename)

    f = open(filename,"w+")
    text = " ".join(text)
    
    f.write(text)
    f.close()

from os import listdir
import os
from os.path import isfile, join
import pickle

trainPath = 'Test_Train/'
trainFiles = [f for f in listdir(trainPath) if isfile(join(trainPath, f))]

bigram_words = set()
trigram_words = set()

j = 0
for file_name in trainFiles:
    j += 1
    print(str(j) + " / " + str(len(trainFiles)))
    with open(trainPath + file_name, 'r') as openfile:
        word_list = openfile.read()
        
    openfile.close()

    word_list = word_list.split(" ")

    for i in range(len(word_list)-1):
        if i < len(word_list) - 1:
            bigram_words.add(word_list[i] + " " + word_list[i+1])
   
        if  i < len(word_list) - 2:
            trigram_words.add(word_list[i] + " " + word_list[i+1] + " " + word_list[i+2])
            

bigram_words = list(bigram_words)
with open("allBigramWords.txt", "wb") as fp:
    pickle.dump(bigram_words, fp)
trigram_words = list(trigram_words)
with open("allTrigramWords.txt", "wb") as fp:
    pickle.dump(trigram_words, fp)
