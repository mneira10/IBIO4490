#!/usr/bin/python3
# -*- coding: utf-8 -*-
from Segment import segmentByClustering, evalMetric
import imageio
import matplotlib.pyplot as plt
import toMat
import os
import numpy as np

def imshow(img, seg, title='Image'):
    import matplotlib.pyplot as plt
    plt.imshow(img, cmap=plt.get_cmap('gray'))
    plt.imshow(seg, cmap=plt.get_cmap('rainbow'), alpha=0.5)
    cb = plt.colorbar()
    cb.set_ticks(range(seg.max()+1))
    plt.title(title)
    plt.axis('off')
    plt.show()
def groundtruth(img_file):
    import scipy.io as sio
    img = imageio.imread(img_file)
    gt = sio.loadmat(img_file.replace('jpg', 'mat'))
    segm = gt['groundTruth'][0, 5][0][0]['Segmentation']
    imshow(img, segm, title='Groundtruth')
def gtSegmentation(imgName):

    import scipy.io as sio
    gt = sio.loadmat(imgName.replace('jpg', 'mat'))
    #print(gt['groundTruth'])
    segm = gt['groundTruth'][0,1][0][0]['Segmentation']
    # plt.imshow(segm)
    # plt.show()
    return segm

def check_dataset(folder):
    import os
    if not os.path.isdir(folder):
        os.system('wget http://157.253.196.67/BSDS_small.zip')
        os.system('unzip BSDS_small')

def evalDB(files,color,method):
    ans = 0
    for i,daFile in enumerate(files):
        imag = imageio.imread(daFile)
        segSet = []
        for j,k in enumerate([2,5,10,15,20]):
            print('\r{:.2f}% {:.2f}%'.format(100*(i+1)/len(files),100*(j+1)/5),end='')
            
            seg = evalFile(imag,color,k,method)
            # print(type(seg))
            segSet.append(seg)
        savePath = './segmentations/'+method+'/'+daFile.split('/')[-1].split('.')[0]
        saveObj = np.array(segSet)
        toMat.toMat(saveObj,savePath)
    print()
  


def evalFile(imag,color,k,method):
    # print(daFile)
    # daFile = './BSDS_small/val/'+daFile
    
    # gt = gtSegmentation(daFile+'.jpg')
    seg = segmentByClustering(imag, color, k, method) 
    return seg

if __name__ == '__main__':
    from os import listdir
    from os.path import isfile, join
    mypath = './BSR/BSDS500/data/images/test/'
    onlyfiles = [mypath+f for f in listdir(mypath) if (isfile(join(mypath, f)) and f.endswith('.jpg') )]
    
    os.system('mkdir segmentations')
    for method in ['kmeans', 'gmm', 'hierarchical', 'watershed']:
        os.system('mkdir segmentations/'+method)
        for color in ['lab']:
            
            print("Color: {} Method: {}".format(color,method))
            evalDB(onlyfiles,color,method)
            

