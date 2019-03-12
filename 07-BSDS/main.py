#!/usr/bin/python3
# -*- coding: utf-8 -*-

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


def check_dataset(folder):
    import os
    if not os.path.isdir(folder):
        os.system('wget http://157.253.196.67/BSDS_small.zip')
        os.system('unzip BSDS_small')


if __name__ == '__main__':
    import argparse
    import imageio
    # Change this line if your function has a different name
    from Segment import segmentByClustering
    parser = argparse.ArgumentParser()

    # If you use more please add them to this list.
    parser.add_argument('--color', type=str, default='rgb',
                        choices=['rgb', 'lab', 'hsv', 'rgb+xy', 'lab+xy', 'hsv+xy'])
    parser.add_argument('--k', type=int, default=4)
    parser.add_argument('--method', type=str, default='kmeans',
                        choices=['kmeans', 'gmm', 'hierarchical', 'watershed'])
    parser.add_argument('--img_file', type=str, required=True)

    opts = parser.parse_args()

    check_dataset(opts.img_file.split('/')[0])

    img = imageio.imread(opts.img_file)
    seg = segmentByClustering(img, opts.color, opts.k, opts.method)
    imshow(img, seg, title='Prediction')
    groundtruth(opts.img_file)
