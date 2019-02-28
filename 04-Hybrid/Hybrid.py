
# coding: utf-8


import cv2 as cv2
# import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st
from scipy import fftpack
from matplotlib.colors import LogNorm
from scipy.misc import imsave
import os
# Imagenes tomadas de: [aqui](https://www.semana.com/confidenciales-semanacom/articulo/duque-se-refiere-a-alvaro-uribe-como-presidente/601095#)


im = cv2.imread('./imgs/duque.jpg')
duque = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)


im = cv2.imread('./imgs/uribe.jpg')
uribe = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
print(uribe.shape)


def fourier(im):
    res = []
    for i in range(3):
        res.append(fftpack.fft2(im[:,:,i]))
        res[i] = np.fft.fftshift(res[i])
    res = np.dstack((res[0],res[1],res[2]))
    return res
def ifourier(im):
    res = [] 
    for i in range(3):
        res.append(np.fft.fftshift(im[:,:,i]))
        res[i] = fftpack.ifft2(res[i]).real
    res = np.dstack((res[0],res[1],res[2]))
    res[res<0] = 0
    res[res>255] = 255
    res = res.astype(int)
    return res
def plot_spectrum(im_fft):
    # A logarithmic colormap
    plt.imshow(np.abs(im_fft), norm=LogNorm(vmin=5))
    plt.colorbar()



def inCircle(x,y,r):
    if((x**2+y**2)**0.5<r):
        return True
    return False



def highFilter(im,r):
    lenx,leny,n = im.shape
    for i in range(lenx):
        for j in range(leny):
            for k in range(n):
                if( inCircle(i-lenx//2,j-leny//2,r)):
                    im[i,j,k]=0+0j
    return im
    
    



def lowFilter(im,r):
    highim = im.copy()
    highim = highFilter(highim,r)
    return im-highim



def highFrecImage(im,r):
    im = fourier(im)
    im = highFilter(im,r)
    im  = ifourier(im)
    return im
def lowFrecImage(im,r):
    imf = fourier(im)
    lowim = lowFilter(imf,r)
    ifim  = ifourier(lowim)
    return ifim



def makeHybrid(im1,im2,r):
    low = lowFrecImage(im1,r)
    high = highFrecImage(im2,r)
    return low+high

finalHybrid=makeHybrid(duque,uribe,8)
imsave('./imgs/hybrid.png',finalHybrid)
os.system('display ./imgs/hybrid.png')
