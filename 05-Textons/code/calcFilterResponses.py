#!/usr/bin/python3
# coding: utf-8
import numpy as np
#import matplotlib.pyplot as plt
import cifar10 as cf
import pickle
import sys
from fbCreate import fbCreate
from skimage import color
from skimage import io
from skimage.transform import resize
from fbRun import fbRun
import numpy as np
from computeTextons import computeTextons
from pathlib import Path


sys.path.append('../python')

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


def imagesToGreyscale(imgs, name):
    greys = []
    print("Turning", name, "to greyscale...")
    for i in range(len(imgs['data'])):
        print('\r' + '{:.2f}'.format((i+1)*100/len(imgs['data']))+ '% ',end="")
        greys.append(color.rgb2gray(resize(imgs['data'][i], (32, 32))))

    toPickle(greys, './data/'+name)
    print()
    print(name, "turned and saved to greyscale.")
    return greys

if fileExists('./data/trainImages.pkl') and fileExists('./data/testImages.pkl'):
    print('Loading greyscale pickled greyscale images...')
    trainGreys = loadPickle('./data/trainImages.pkl') 
    testGreys = loadPickle('./data/testImages.pkl')
    print('Loaded greyscale images.')

else:
    print("Loading CIFAR10 data...")
    trainData = cf.load_cifar10(meta='./cifar-10-batches-py/', mode=1)
    
    labels = trainData['labels'] 
    
    labelIds = np.unique(labels) 

    imgs = []
    outLabels = []
    np.random.seed(0)

    for imgId in labelIds: 
        objIDs = np.where(labels==imgId)[0] 
        rand100 = np.random.choice(objIDs,100,replace=False) 
        outLabels += list(labels[rand100]) 
        imgs += list(trainData['data'][rand100])
    
    
    testData = cf.load_cifar10(meta='./cifar-10-batches-py/', mode='test')
    print("CIFAR10 data loaded.")
    
    toPickle(outLabels,'./data/trainLabels')

    imgDict = {'data':imgs}
    trainGreys = imagesToGreyscale(imgDict, 'trainImages')
    testGreys = imagesToGreyscale(testData, 'testImages')


if fileExists('./data/trainFilterResponses.pkl') and fileExists('./data/testFilterResponses.pkl'):
    print('Loading filter responses...')
    trainFilterResponses = loadPickle('./data/trainFilterResponses.pkl')
    testFilterResponses = loadPickle('./data/testFilterResponses.pkl')
    print('Filter responses loaded.')

else:
    # Create a filter bank with deafult params
    # fbCreate(**kwargs, vis=True) for visualization
    print('Creating filters...')
    fb = fbCreate(support=2, startSigma=0.6)

    print("generating train filter responses...")
    trainFilterResponses = fbRun(fb, np.hstack(trainGreys))
    toPickle(trainFilterResponses, './data/trainFilterResponses')

    print("generating test filter responses...")
    testFilterResponses = fbRun(fb, np.hstack(testGreys))
    toPickle(testFilterResponses, './data/testFilterResponses')

ks = [10,20,50,100,150,200,500]

trainFR = {'data': trainFilterResponses, 'name': 'train'}
print('Calculating textons...')
# Computer textons from filter
for k in ks:
    print("k="+str(k) + " out of: " + str(ks))
    map, textons = computeTextons(trainFR['data'], k)
    mapAndTexton = {'map':map,'textons':textons}
    toPickle(mapAndTexton,'./data/mapAndTexton'+str(k))






