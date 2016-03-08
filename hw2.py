import numpy as np
import scipy as py
from sklearn.naive_bayes import MultinomialNB

traindata = np.loadtxt('Downloads/20news-bydate/matlab/train.data')
trainlabel = np.loadtxt('Downloads/20news-bydate/matlab/train.label')
testdata = np.loadtxt('Downloads/20news-bydate/matlab/test.data')
testlable = np.loadtxt('Downloads/20news-bydate/matlab/test.label')

with open('Downloads/vocabulary.txt','r') as f:
    vocabr = f.read()
vocabr = vocabr.split()


def split_array(Singrp):
    indlist = [0]*(len(vocabr))
    for i in Singrp[:,1]:
        indlist[i-1] = Singrp[Singrp[:, 1].index(i),2]
    return(indlist)


count = list()
train = list()
def bulit_array(obj):
    for i in np.unique(traindata[:,0]).astype(np.int):
        count.append(obj[:,0].tolist().count(i))
    trainarray = np.vsplit(obj, count)
    for j in len(trainarray):
        elearray = split_array(trainarray[j])
        train.append(elearray)
    trainArray = np.array(train)
    return(trainArray)

# smooth and to avoid underflow
train = bulit_array(traindata)


lsmthtrain = np.log(np.add(train, 1))


clf = MultinomialNB()
clf = clf.fit(trainarry, trainlabel)

