from quickdraw import QuickDrawDataGroup, QuickDrawData
# import matplotlib.pyplot as plt 
import os
import glob

import numpy as np

def downloadData():
  print('No data found.')
  print('Downloading data...')

  # we'll download 5 images from each class for now. There are far
  # too may and download times are slow  
  n=100


  m=10

  #getting class names
  qd = QuickDrawData()
  classes = qd.drawing_names
  classes = np.random.choice(classes,m,replace=False)

  #creating data folder
  os.system('mkdir data')

  for i,c in enumerate(classes):
    saveClassImages(c)
    print('-'[0]*100)
    print("{:.2f} % done".format(i*100/len(classes)))
    print('-'[0]*100)

  print('Data downloaded. {} images per class, {} classes. Total of {} images'.format(
    n,
    len(classes),
    len(classes)*n))

def saveClassImages(className):
  classNameNoSpaces = className.replace(' ','_')
  os.system('mkdir ./data/'+classNameNoSpaces)
  
  qdg = QuickDrawDataGroup(className,max_drawings=5)
  for i,drawing in enumerate(qdg.drawings):
      drawing.image.save('./data/'+classNameNoSpaces + '/'+classNameNoSpaces+ '_'+str(i)+'.png')


# #remove pre-existing data folder
# os.system('rm -rf ./data/')


dataDownloaded = os.path.isdir("./data/")

if not dataDownloaded: 
  downloadData()
else:
  print('Data already downloaded, continuing with sampling')


# dirs = os.listdir('./data/')
# imgs = [img for img in (os.listdir('./data/'+f) for f in dirs)]
# onlyfiles = [f for f in os.listdir('./data/') if os.path.isfile(os.path.join('./data/', f))]
# print(os.listdir('./data/'))

files = glob.glob('./data' + '/**/*.png', recursive=True)

N = 12









    


