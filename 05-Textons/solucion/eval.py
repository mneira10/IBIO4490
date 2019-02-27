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
from sklearn.metrics import precision_recall_fscore_support
import cifar10 as cf
from assignTextons import assignTextons

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

testLabels = cf.load_cifar10('./cifar-10-batches-py/',mode='test')['labels']
testImgs = loadPickle('./data/testFilterResponses.pkl')
filename = './results/resultsData.dat'
f= open(filename,"w+")
f.write('alg,k,n,precision,recall,fscore\n')
f.close()

cm = np.zeros((10,10))
# iterate through k
for k in ks:

    textonPath = './data/mapAndTexton'+str(k)+'.pkl'
    if fileExists(textonPath):
        textons = loadPickle(textonPath)['textons']
        textonMap = assignTextons(testImgs,textons.transpose())

        for n in ns:
            
            knnpath = './data/KNN_n_'+str(n)+'_k_'+str(k)+'.pkl'
            rfpath =  './data/RF_n_'+str(n)+'_k_'+str(k)+'.pkl'

            # if info for k exists
            if fileExists(knnpath): 
                
                print('Retrieving knn model with k=',k, 'and n=',n)
                
                knn = loadPickle(knnpath)
                rf  = loadPickle(rfpath)

                nTest = len(testLabels)
                knnPred = []
                rfPred  = []
                for t in range(nTest):
                    img = textonMap[:,t*32:(t+1)*32]
                    histo = histc(img.flatten(), np.arange(k))
                    knnPred.append(knn.predict([histo])[0])
                    rfPred.append(rf.predict([histo])[0])
                

                print('KNN with k=',k,'n=',n)
                knnprf = (precision_recall_fscore_support(testLabels, knnPred, average='weighted'))
                print(knnprf)
                print('RF with k=',k,'n=',n)
                rfprf = (precision_recall_fscore_support(testLabels, rfPred, average='weighted'))           
                print(rfprf)
                f= open(filename,"a")
                f.write('knn,'+str(k)+','+str(n)+','+str(knnprf[0])+','+str(knnprf[1])+','+str(knnprf[2])+'\n')
                f.write('rf,'+str(k) + ',' +str(n)+','+str(rfprf[0])+','+str(rfprf[1])+','+str(rfprf[2])+'\n')
                             
                f.close()


