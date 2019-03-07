#!/usr/bin/python3
# -*- coding: utf-8 -*-
from Segment import segmentByClustering, evalMetric
import imageio
import matplotlib.pyplot as plt


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

def evalDB(files,color,k,method):
    ans = 0
    for i,daFile in enumerate(files):
        print('\r{:.2f}%'.format(100*(i+1)/len(files)),end='')
        # print(daFile)
        ans += evalFile(daFile,color,k,method)
    print()
    return ans/len(files)


def evalFile(daFile,color,k,method):
    # print(daFile)
    daFile = './BSDS_small/val/'+daFile
    imag = imageio.imread(daFile+'.jpg')
    gt = gtSegmentation(daFile+'.jpg')
    seg = segmentByClustering(imag, color, k, method)
    
    return evalMetric(gt, seg)
    
    

if __name__ == '__main__':
    from os import listdir
    from os.path import isfile, join
    mypath = './BSDS_small/val/'
    onlyfiles = [f.split('.')[0] for f in listdir(mypath) if (isfile(join(mypath, f)) and f.endswith('.jpg') )]
    

    for k in [5,10,15,20]:
        for method in ['kmeans', 'gmm', 'hierarchical', 'watershed']:
            for color in ['rgb', 'lab', 'hsv', 'rgb+xy', 'lab+xy', 'hsv+xy']:
                
                print("{} {} {}".format(color,k,method))
                metric = evalDB(onlyfiles,color,k,method)
                print(metric)
                f= open("results.txt","a")
                f.write("{} {} {} {}\n".format(color,k,method,metric))
                f.close()

