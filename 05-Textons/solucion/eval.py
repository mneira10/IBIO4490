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


ks = [10,20,50,100,150,200,500]
ns = [3,5,10,20,50]



# iterate through k
for k in ks:
    for n in ns:
        knnpath = './data/KNN_n_'+str(n)+'_k_'+str(k)+'.pkl'
        rfpath =  './data/RF_n_'+str(n)+'_k_'+str(k)+'.pkl'

        # if info for k exists
        if fileExists(knnpath): 
            
            print('Retrieving knn model with k=',k, 'and n=',n)
            
            knn = loadPickle(knnpath)
                

                

