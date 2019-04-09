from os import listdir
from os.path import isfile, join
import cv2
import tqdm

mypath = './Emotions_test/'
cascPath = "haarcascade_frontalface_default.xml"

images = [f for f in listdir(mypath) if isfile(join(mypath, f))]

faceCascade = cv2.CascadeClassifier(cascPath)

for tau,im in tqdm.tqdm(enumerate(images),total=len(images)):
  image = cv2.imread(mypath+im)
  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

  faces = faceCascade.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(30, 30),
    flags=cv2.CASCADE_SCALE_IMAGE
    # flags = cv2.CV_HAAR_SCALE_IMAGE
  )

  

  # Draw a rectangle around the faces
  if len(faces)!=1:
    maxFace=None
    maxW = 0
    for i,(x, y, w, h) in enumerate(faces):
      if w>maxW:
        maxW = w
        maxFace = faces[i]
    (x, y, w, h) = maxFace
    # cv2.rectangle(gray, (x, y), (x+w, y+h), (0, 255, 0), 2)
    # cv2.imshow("Faces found", gray)
    # cv2.waitKey(0)
  else:
    (x, y, w, h) = faces[0]
  
  gray = gray[y:y+h,x:x+w]
  gray = cv2.resize(gray,(48,48))
  
  # if len(faces)!=1:
  #   cv2.imshow("Faces found", gray)
  #   cv2.waitKey(0)

  cv2.imwrite('./cleanImages/'+im,gray)