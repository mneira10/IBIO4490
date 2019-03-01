#!/usr/bin/python3
# coding: utf-8
import numpy as np
import pickle
import sys
from skimage import color
from skimage import io
from skimage.transform import resize
from fbRun import fbRun
import numpy as np
from computeTextons import computeTextons
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
import time
from sklearn.metrics import confusion_matrix
from assignTextons import assignTextons
import cifar10 as cf


def fileExists(path):
    return Path(path).exists()


def toPickle(obj, name):
    pickle.dump(obj, open(name+'.pkl', "wb"))


def loadPickle(name):
    return pickle.load(open(name, "rb"))

def histc(X, bins):
    import numpy as np
    map_to_bins = np.digitize(X, bins)
    r = np.zeros(bins.shape)
    for i in map_to_bins:
        r[i-1] += 1
    return np.array(r)

k = 100
n = 10
# iterate through k


pklpath = './data/histo' + str(k) + '.pkl'

#get histograms
print('Retrieving histograms for k =', k)
histograms = loadPickle(pklpath)
histograms = np.array(histograms)

#get labels
labels = loadPickle('./data/trainLabels.pkl')

#classify
clf = RandomForestClassifier(n_estimators=100, random_state=0)
print('fitting RF with k=', k, 'n_estimators=', n)
start = time.time()
clf.fit(histograms, labels)
print('Fit took {} seconds'.format(time.time()-start))

#saving best model
toPickle(clf, './data/bestAlgo.pkl')

#evaluating in train data

trainPreds = clf.predict(histograms)

trainCM = confusion_matrix(labels, trainPreds)

toPickle(trainCM,'./data/trainConfusionMatrix')
print('Trainning confusion matrix:')
print(trainCM)


#evaluating in test data


testTextonMapPath = './data/testTextonMap.pkl'
if not fileExists(testTextonMapPath):
    print('Loading test images')
    testImgs = loadPickle('./data/testFilterResponses.pkl')
    print('Loading textons')
    textonPath = './data/mapAndTexton'+str(k)+'.pkl'
    textons = loadPickle(textonPath)['textons']
    print('Asigning textons to test images')
    textonMap = assignTextons(testImgs,textons.transpose())
    toPickle(textonMap,'./data/testTextonMap')
else:
    textonMap = loadPickle(testTextonMapPath)

print('Loading test labels')
testLabels = cf.load_cifar10('./cifar-10-batches-py/',mode='test')['labels']


nTest = len(testLabels)
rfPred  = []
print('Evaluating on test set')
for t in range(nTest):
    print('\r {:.2f}%'.format((t+1)*100/nTest),end='')
    img = textonMap[:,t*32:(t+1)*32]
    histo = histc(img.flatten(), np.arange(k))
    rfPred.append(clf.predict([histo])[0])

testCM = confusion_matrix(testLabels,rfPred)

toPickle(testCM,'./data/testConfusionMatrix')
print()
print('Test confusion matrix:')
print(testCM)





