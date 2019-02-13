#!/usr/bin/python3
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib.image as mpimg
import os
import glob
import numpy as np

#print('installing dependencies...')
#os.system('pip3 install quickdraw')
#print('installed dependencies.')

from quickdraw import QuickDrawDataGroup, QuickDrawData

def downloadData():
    print('No data found.')
    print('Downloading data...')

    # n = # of images
    # m = # of classes
    # there are simply too many images and classes to download
    # we limit the datast to 100 images of 10 classes
    n = 100
    m = 20

    # getting class names
    qd = QuickDrawData()
    classes = qd.drawing_names
    classes = np.random.choice(classes, m, replace=False)

    # creating data folder
    os.system('mkdir data')

    for i, c in enumerate(classes):
        saveClassImages(c)
        print('-'[0]*100)
        print("{:.2f} % done".format(i*100/len(classes)))
        print('-'[0]*100)

    print('Data downloaded. {} images per class, {} classes. Total of {} images'.format(
        n,
        len(classes),
        len(classes)*n))


def saveClassImages(className):
    classNameNoSpaces = className.replace(' ', '_')
    os.system('mkdir ./data/'+classNameNoSpaces)

    qdg = QuickDrawDataGroup(className, max_drawings=5)
    for i, drawing in enumerate(qdg.drawings):
        drawing.image.save('./data/'+classNameNoSpaces +
                           '/'+classNameNoSpaces + '_'+str(i)+'.png')


dataDownloaded = os.path.isdir("./data/")

if not dataDownloaded:
    downloadData()
else:
    print('Data already downloaded, continuing with sampling')

files = glob.glob('./data' + '/**/*.png', recursive=True)

# # of images to sample and see
N = 12

chosen = np.random.choice(files, N, replace=False)

# plotting paramters
nrow = 3
ncol = 4

# taken from https://stackoverflow.com/questions/41071947/how-to-remove-the-space-between-subplots-in-matplotlib-pyplot?noredirect=1&lq=1
fig = plt.figure(figsize=(16, 12))

gs = gridspec.GridSpec(nrow, ncol, wspace=0.0, hspace=0.0)
cont = 0
for i in range(nrow):
    for j in range(ncol):
        im = mpimg.imread(chosen[cont])
        name = ' '.join(chosen[cont].split('/')[-1].split('_')[:-1])
        ax = plt.subplot(gs[i, j])
        # print(ax.get_ylim())
        ax.text(100, 250, name, bbox={
                'facecolor': 'lightblue', 'alpha': 1, 'edgecolor': 'black', 'pad': 1}, fontdict = dict(size=15))
        ax.imshow(im)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        cont += 1

plt.savefig('./sample.png')
os.system('display ./sample.png &')
