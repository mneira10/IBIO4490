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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

def fileExists(path):
    return Path(path).exists()

def toPickle(obj, name):
    pickle.dump(obj, open(name+'.pkl', "wb"))

def loadPickle(name):
    return pickle.load(open(name, "rb"))



ks = [10,20,50,100,150,200,500]
ns = [3,5,10,20,50]
# iterate through k
for k in ks:

    pklpath = './data/histo'+ str(k) + '.pkl'
    # if info for k exists
    if fileExists(pklpath): 

        print('Retrieving histograms for k =', k)
        
        histograms = loadPickle(pklpath)
        histograms = np.array(histograms)
        
        labels = loadPickle('./data/trainLabels.pkl')
        
        for n in ns:
            
            neigh = KNeighborsClassifier(n_neighbors=3)
            print('fitting KNN with k=',k,'neighbours=',n)
            neigh.fit(histograms, labels) 
            

            toPickle(neigh,'./data/KNN_n_'+str(n)+'_k_'+str(k))

            clf = RandomForestClassifier(n_estimators=100, random_state=0)
            print('fitting RF with k=',k,'n_estimators=',n)
            clf.fit(histograms, labels)

            toPickle(clf,'./data/RF_n_'+str(n)+'_k_'+str(k))


            
        

    
