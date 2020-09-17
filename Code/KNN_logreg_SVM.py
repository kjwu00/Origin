import random
import collections
import math
import sys
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from os import listdir
from os.path import isfile, join
import pandas as pd
import pickle
import collections
import time

model = 1 #1 is unigram, 2 is bigram.

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)
start = time.time()

allWords = set()
if model == 1:
    with open("allWords.txt", "rb") as fp:   
        allWords = pickle.load(fp)
elif model == 2:
    with open("allBigramWords.txt", "rb") as fp:   
        allWords = pickle.load(fp)
else:
    with open("allTrigramWords.txt", "rb") as fp:   
        allWords = pickle.load(fp)

print(len(allWords))
dataset = []

featurestemplate = dict.fromkeys(allWords, 0)

def extractWordFeatures(x):
    words = x.split(" ")
    features = dict.fromkeys(allWords, 0)
    for word in words:
        features[word] += 1
    return features

def extractBigramFeatures(x):
    words = x.split(" ")
    features = dict.fromkeys(allWords, 0)
    for i in range(len(words)-1):
        word = words[i] + " " + words[i+1]
        features[word] += 1
    return features

def extractTrigramFeatures(x):
    words = x.split(" ")
    features = dict.fromkeys(allWords, 0)
    for i in range(len(words)-2):
        word = words[i] + " " + words[i+1] + " "+ words[i+2]
        features[word] += 1
    return features

trainPath = 'Test_Train/'
trainFiles = [f for f in listdir(trainPath) if isfile(join(trainPath, f))]

print("begin training")
print(timeSince(start))
print()

i = 0
for file_name in trainFiles:
    i += 1
    if i %1000 == 0:
        print(str(i) + " " + str(timeSince(start)))
        
    language = file_name.split("_")
    language = language[1]
    with open(trainPath + file_name, 'r') as openfile:
        data = openfile.read()
    openfile.close()
    words = data[0]  
    
    file = None
    if model == 1:
        file = extractWordFeatures(data)
    elif model == 2:
        file = extractBigramFeatures(data)
    else:
        file = extractTrigramFeatures(data)
        
    file["text_language"] = language
    dataset.append(file)


print("made dataset")
print(timeSince(start))
print()

allData = pd.DataFrame(dataset)
print("got data")
print(timeSince(start))
print()
print(allData)
X = allData.loc[:, allData.columns != 'text_language']
Y = allData.loc[:, allData.columns == 'text_language']

print('X train')
print(X)
print('Y train')
print(Y)



def extractBigramFeatures(x):
    words = x.split(" ")
    features = dict.fromkeys(allWords, 0)
    
    no_key_wtf = set({})
    
    for i in range(len(words)-1):
        word = words[i] + " " + words[i+1]
        if word in features.keys():
            features[word] += 1
        else:
            no_key_wtf.add(word)  
    return features


print("start validation")
print(timeSince(start))
print()
#Validate
valPath = 'Test_Val/'
valFiles = [f for f in listdir(valPath) if isfile(join(valPath, f))]

valdata = []
for file_name in valFiles:
    language = file_name.split("_")
    language = language[1]
    
    with open(valPath + file_name, 'r') as openfile:
        data = openfile.read()
        
    openfile.close()
    words = data[0]
    file = None
    if model == 1:
        file = extractWordFeatures(data)
    elif model == 2:
        file = extractBigramFeatures(data)
    else:
        file = extractTrigramFeatures(data)
        
    file["text_language"] = language
    valdata.append(file)
    

    
print("read validation files")
print(timeSince(start))
print()
valData = pd.DataFrame(valdata)
print("put validation in panda")
print(timeSince(start))
print()
X_val = valData.loc[:, allData.columns != 'text_language']
Y_val = valData.loc[:, allData.columns == 'text_language']

print('X validation')
print(X_val)
print('Y validation')
print(Y_val)


print("train KNN 3")
print(timeSince(start))
print()
knn3 = KNeighborsClassifier(n_neighbors=3)
knn3.fit(X,Y)
#Y_knn = knn.predict(X)

print("validate KNN 3")
print(timeSince(start))
print()
y_rnn_val = knn3.predict(X_val)
print(metrics.accuracy_score(Y_val, y_rnn_val))
print(metrics.confusion_matrix(Y_val, y_rnn_val))

print("train KNN 5")
print(timeSince(start))
print()
knn5 = KNeighborsClassifier(n_neighbors=5)
knn5.fit(X,Y)

print("validate KNN 5")
print(timeSince(start))
print()
y_knn_val2 = knn5.predict(X_val)
print(metrics.accuracy_score(Y_val, y_knn_val2))
print(metrics.confusion_matrix(Y_val, y_knn_val2))

print("train KNN 7")
print(timeSince(start))
print()
knn7 = KNeighborsClassifier(n_neighbors=7)
knn7.fit(X,Y)

print("validate KNN 7")
print(timeSince(start))
print()
y_knn_val3 = knn7.predict(X_val)
print(metrics.accuracy_score(Y_val, y_knn_val3))
print(metrics.confusion_matrix(Y_val, y_knn_val3))

print("train KNN 9")
print(timeSince(start))
print()
knn9 = KNeighborsClassifier(n_neighbors=9)
knn9.fit(X,Y)

print("validate KNN")
print(timeSince(start))
print()
y_knn_val4 = knn9.predict(X_val)
print(metrics.accuracy_score(Y_val, y_knn_val4 ))
print(metrics.confusion_matrix(Y_val, y_knn_val4 ))


print("train logistic l1 c=0.01")
print(timeSince(start))
print()
from sklearn.linear_model import LogisticRegression
logistic_l1_c01 = LogisticRegression(random_state=0, penalty='l1', C=0.01)
logistic_l1_c01.fit(X, Y)

print("train accuracy l1 C=0.01")
print(timeSince(start))
print()
y_log_l1_c01 = logistic_l1_c01.predict(X)
print(metrics.accuracy_score(Y, y_log_l1_c01))

print("validate logistic l1 C=0.01")
print(timeSince(start))
print()
y_log_l1_c01_val = logistic_l1_c01.predict(X_val)
print(metrics.accuracy_score(Y_val, y_log_l1_c01_val))
print(metrics.confusion_matrix(Y_val, y_log_l1_c01_val))

print("train logistic l1 c=0.1")
print(timeSince(start))
print()
from sklearn.linear_model import LogisticRegression
logistic_l1_c1 = LogisticRegression(random_state=0, penalty='l1', C=0.1)
logistic_l1_c1.fit(X, Y)

print("train accuracy l1 C=0.1")
print(timeSince(start))
print()
y_log_l1_c1 = logistic_l1_c1.predict(X)
print(metrics.accuracy_score(Y, y_log_l1_c1))

print("validate logistic l1 C=0.1")
print(timeSince(start))
print()
y_log_l1_c1_val = logistic_l1_c1.predict(X_val)
print(metrics.accuracy_score(Y_val, y_log_l1_c1_val))
print(metrics.confusion_matrix(Y_val, y_log_l1_c1_val))

print("train logistic l1 c=1")
print(timeSince(start))
print()
from sklearn.linear_model import LogisticRegression
logistic_l1_c_1 = LogisticRegression(random_state=0, penalty='l1', C=1)
logistic_l1_c_1.fit(X, Y)

print("train logistic accuracy l1 C=1")
print(timeSince(start))
print()
y_log_l1_c_1 = logistic_l1_c_1.predict(X)
print(metrics.accuracy_score(Y, y_log_l1_c_1))


print("validate logistic l1 C=1")
print(timeSince(start))
print()
y_log_l1_c_1_val = logistic_l1_c_1.predict(X_val)
print(metrics.accuracy_score(Y_val, y_log_l1_c_1_val))
print(metrics.confusion_matrix(Y_val, y_log_l1_c_1_val))

print("train logistic l1 c=10")
print(timeSince(start))
print()
from sklearn.linear_model import LogisticRegression
logistic_l1_c10 = LogisticRegression(random_state=0, penalty='l1', C=10)
logistic_l1_c10.fit(X, Y)


print("train accuracy l1 C=10")
print(timeSince(start))
print()
y_log_l1_c10 = logistic_l1_c1.predict(X)
print(metrics.accuracy_score(Y, y_log_l1_c10))

print("validate logistic l1 C=10")
print(timeSince(start))
print()
y_log_l1_c10_val = logistic_l1_c1.predict(X_val)
print(metrics.accuracy_score(Y_val, y_log_l1_c10_val))
print(metrics.confusion_matrix(Y_val, y_log_l1_c10_val))

print("train logistic l2 c=0.01")
print(timeSince(start))
print()
from sklearn.linear_model import LogisticRegression
logistic_l2_c01 = LogisticRegression(random_state=0, penalty='l2', C=0.01)
logistic_l2_c01.fit(X, Y)


print("train accuracy l2 C=0.01")
print(timeSince(start))
print()
y_log_l2_c01 = logistic_l2_c01.predict(X)
print(metrics.accuracy_score(Y, y_log_l2_c01))


print("validate logistic l2 C=0.01")
print(timeSince(start))
print()
y_log_l2_c01_val = logistic_l2_c01.predict(X_val)
print(metrics.accuracy_score(Y_val, y_log_l2_c01_val))
print(metrics.confusion_matrix(Y_val, y_log_l2_c01_val))

print("train logistic l2 c=0.1")
print(timeSince(start))
print()
from sklearn.linear_model import LogisticRegression
logistic_l2_c1 = LogisticRegression(random_state=0, penalty='l2', C=0.1)
logistic_l2_c1.fit(X, Y)

print("train accuracy l2 C=0.1")
print(timeSince(start))
print()
y_log_l2_c1 = logistic_l2_c1.predict(X)
print(metrics.accuracy_score(Y, y_log_l2_c1))

print("validate logistic l2 C=0.1")
print(timeSince(start))
print()
y_log_l2_c1_val = logistic_l2_c1.predict(X_val)
print(metrics.accuracy_score(Y_val, y_log_l2_c1_val))
print(metrics.confusion_matrix(Y_val, y_log_l2_c1_val))

print("train logistic l2 c=1")
print(timeSince(start))
print()
from sklearn.linear_model import LogisticRegression
logistic_l2_c_1 = LogisticRegression(random_state=0, penalty='l2', C=1)
logistic_l2_c_1.fit(X, Y)


print("train accuracy l2 C=1")
print(timeSince(start))
print()
y_log_l2_c_1 = logistic_l2_c_1.predict(X)
print(metrics.accuracy_score(Y, y_log_l2_c_1))

print("validate logistic l2 C=1")
print(timeSince(start))
print()
y_log_l2_c_1_val = logistic_l2_c_1.predict(X_val)
print(metrics.accuracy_score(Y_val, y_log_l2_c_1_val))
print(metrics.confusion_matrix(Y_val, y_log_l2_c_1_val))

print("train logistic l2 c=10")
print(timeSince(start))
print()
from sklearn.linear_model import LogisticRegression
logistic_l2_c10 = LogisticRegression(random_state=0, penalty='l2', C=10)
logistic_l2_c10.fit(X, Y)

print("train accuracy l2 C=10")
print(timeSince(start))
print()
y_log_l2_c10 = logistic_l2_c1.predict(X)
print(metrics.accuracy_score(Y, y_log_l2_c10))

print("validate logistic l2 C=10")
print(timeSince(start))
print()
y_log_l2_c10_val = logistic_l2_c1.predict(X_val)
print(metrics.accuracy_score(Y_val, y_log_l2_c10_val))
print(metrics.confusion_matrix(Y_val, y_log_l2_c10_val))
#print(metrics.confusion_matrix(Y_val, y_log_
pred2).diag()/metrics.confusion_matrix(Y_val, y_log_pred2).sum(1))

# training a SVM classifier 
from sklearn.svm import SVC

print("SVC")

kernels = ['linaer', 'poly', 'rbf']
gammas = ['scale']
cs = [0.1, 1, 10]
degrees = [2, 3]

for kernel in kernels: 
    for gamma in gammas:
        for c in cs:
            if kernel == 'poly':
                for degree in degrees:
                    print("train SVC: kernel = " + kernel + ", degree = " + str(degree) +", gamma = " + str(gamma) + ", c = " + str(c))
                    print(timeSince(start))
                    print()
                    svm_model = SVC(kernel = kernel, gamma = gamma, C = c).fit(X, Y)
                    
                    print("train svm accuracy: kernel = " + kernel + ", gamma = " + str(gamma) + ", c = " + str(c))
                    print(timeSince(start))
                    print()
                    y_svm = svm_model.predict(X)
                    print(metrics.accuracy_score(Y, y_svm))
                    print(metrics.confusion_matrix(Y, y_svm))

                    print("validate svm: kernel = " + kernel + ", degree = " + str(degree) +", gamma = " + str(gamma) + ", c = " + str(c))
                    print(timeSince(start))
                    print()
                    y_svm_val = svm_model.predict(X_val)
                    print(metrics.accuracy_score(Y_val, y_svm_val))
                    print(metrics.confusion_matrix(Y_val, y_svm_val))

            else:
                print("train SVC: kernel = " + kernel + ", gamma = " + str(gamma) + ", c = " + str(c))
                print(timeSince(start))
                print()
                svm_model = SVC(kernel = kernel, gamma = gamma, C = c).fit(X, Y)

                print("train svm accuracy: kernel = " + kernel + ", gamma = " + str(gamma) + ", c = " + str(c))
                print(timeSince(start))
                print()
                y_svm = svm_model.predict(X)
                print(metrics.accuracy_score(Y, y_svm))
                print(metrics.confusion_matrix(Y, y_svm))
                
                print("validate svm accuracy: kernel = " + kernel + ", gamma = " + str(gamma) + ", c = " + str(c))
                print(timeSince(start))
                print()
                y_svm_val = svm_model.predict(X_val)
                print(metrics.accuracy_score(Y_val, y_svm_val))
                print(metrics.confusion_matrix(Y_val, y_svm_val))
