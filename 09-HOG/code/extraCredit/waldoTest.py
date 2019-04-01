import pickle 
import cv2
import os
from glob import glob
from sklearn import svm
import trainWaldo 
import tqdm

def waldoHere(pathIm,clf):
  im = cv2.imread(pathIm,0)
  inHere =False
  width = im.shape[1]
  for w in [int(width*s) for s in [0.2,0.1,0.05]]:
    resized = image_resize(im,width=w)
    trainWaldo.showImage(resized)
    if windowSlide(resized,clf):
      inHere = True
      break
  return inHere




def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    # taken from https://stackoverflow.com/questions/44650888/resize-an-image-without-distortion-opencv
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized

def windowSlide(im,clf):
  # yjumps = im.shape[1]//36-1
  # xjumps = im.shape[0]//36-1
  # inHere = False
  # for i in range(xjumps):
  #   for j in range(yjumps):
  # #     cropped = im[i*36:i*36+36, j*36:j*36+36]
  #     trainWaldo.showImage(cropped)
  #     # print(cropped.shape,i*36,i*36+36, j*36,j*36+36,im.shape)
  #     # print(i*36-(i*36+36), j*36-(j*36+36))
  #     feats = trainWaldo.HOGDescriptor(cropped)
  #     if clf.predict([feats]):
  #       inHere = True
  #       break

  inHere = False 
  w = im.shape[0]
  h = im.shape[1]
  i=0
  while w-i>36:
    j=0
    while h-j>36:
      cropped = im[i:i+36, j:j+36]
      # trainWaldo.showImage(cropped)
      # print(cropped.shape,i*36,i*36+36, j*36,j*36+36,im.shape)
      # print(i*36-(i*36+36), j*36-(j*36+36))
      feats = trainWaldo.HOGDescriptor(cropped)
      print(clf.predict([feats]))
      if clf.predict([feats]):
        print('entraaaaa')
        inHere = True
        break
      j += 4
    i+=4
  return inHere


def resize(im):
  return cv2.resize(im,(36,36))


PATH = '../Waldo/13--Interview/'
images = [y for x in os.walk(PATH) for y in glob(os.path.join(x[0], '*.jpg'))]

clf = pickle.load(open('./waldoClassifier.pkl', 'rb'))



# for i,imPath in tqdm.tqdm(enumerate(images),total=len(images)):
if waldoHere('../Waldo/13--Interview/13_Interview_Interview_On_Location_13_558.png',clf):
  print(imPath)