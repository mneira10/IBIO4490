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
# from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
import time
from sklearn.metrics import confusion_matrix


def fileExists(path):
    return Path(path).exists()


def toPickle(obj, name):
    pickle.dump(obj, open(name+'.pkl', "wb"))


def loadPickle(name):
    return pickle.load(open(name, "rb"))


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
print(trainCM)
