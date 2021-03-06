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
        print((i+1)*100/len(imgs['data']), '% ')
        greys.append(color.rgb2gray(resize(imgs['data'][i], (32, 32))))

    toPickle(greys, './data/'+name)
    print()
    print(name, "turned and saved to greyscale.")
    return greys


ks = [10,20,50,100,150,200,500]
# iterate through k
for k in ks:
    kpath = './data/mapAndTexton'+str(k) + '.pkl'

    # if info for k exists
    if fileExists(kpath): 

        print('Retrieving map and textons for k =',k)
        
        mapNTextons = loadPickle('./data/mapAndTexton'+str(k) + '.pkl')
        map = mapNTextons['map']
        textons = mapNTextons['textons']

        print('Calculating histograms...')
        numImages = int(map.shape[1]/32)
        histograms = []
        #iterate through images
        for i in range(numImages):
            print('\r'+'{:.2f}'.format((i+1)*100/numImages)+'%',end="")
            img = map[:,i*32:(i+1)*32]
            histograms.append(histc(img.flatten(), np.arange(k)))
            
        toPickle(histograms,'./data/histo'+str(k))
        print()
        print("k=",k,"histograms calculated.")

    

