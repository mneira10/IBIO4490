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
import matplotlib.pyplot as plt
import os 



def numToName(num):
    translate = {0:'airplane',1:'automobile',2:'bird',3:'cat',4:'deer',5:'dog',6:'frog',7:'horse',8:'ship',9:'truck'}
    return translate[num]
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

#evaluating in test data
k = 100
n = 10

bestAlgPath = './data/bestAlgo.pkl'

clf = loadPickle(bestAlgPath)

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
testImages = cf.load_cifar10('./cifar-10-batches-py/',mode='test')['data']

nTest = len(testLabels)
indicesToTest = np.random.choice(nTest,6,replace=False)
fig = plt.figure(figsize = (20,15))
print('Evaluating on test set')
for i,t in enumerate(indicesToTest):
    plt.subplot(2,3,i+1)
    img = textonMap[:,t*32:(t+1)*32]
    histo = histc(img.flatten(), np.arange(k))
    pred = clf.predict([histo])[0]
    plt.imshow(testImages[t])
    plt.title("Label: {} Pred: {}".format(numToName(testLabels[t]),numToName(pred)))

plt.savefig('./demo.png')
os.system('display ./demo.png')


