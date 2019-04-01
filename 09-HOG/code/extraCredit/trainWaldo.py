import cv2
import numpy as np
from matplotlib import pyplot as plt
# import features
from os.path import isfile, join
from os import listdir
from sklearn import svm
import pickle

def HOGDescriptor(img):
  '''
  Taken word for word from https://stackoverflow.com/questions/6090399/get-hog-image-features-from-opencv-python
  '''
  cell_size = (8, 8)  # h x w in pixels
  block_size = (2, 2)  # h x w in cells
  nbins = 9  # number of orientation bins

  # winSize is the size of the image cropped to an multiple of the cell size
  hog = cv2.HOGDescriptor(_winSize=(img.shape[1] // cell_size[1] * cell_size[1],
                                    img.shape[0] // cell_size[0] * cell_size[0]),
                          _blockSize=(block_size[1] * cell_size[1],
                                      block_size[0] * cell_size[0]),
                          _blockStride=(cell_size[1], cell_size[0]),
                          _cellSize=(cell_size[1], cell_size[0]),
                          _nbins=nbins)

  n_cells = (img.shape[0] // cell_size[0], img.shape[1] // cell_size[1])
  hog_feats = hog.compute(img)\
                .reshape(n_cells[1] - block_size[1] + 1,
                          n_cells[0] - block_size[0] + 1,
                          block_size[0], block_size[1], nbins) \
                .transpose((1, 0, 2, 3, 4))  # index blocks by rows first
  # hog_feats now contains the gradient amplitudes for each direction,
  # for each cell of its group for each group. Indexing is by rows then columns.
  # print(hog_feats.shape)
  gradients = np.zeros((n_cells[0], n_cells[1], nbins))

  # count cells (border cells appear less often across overlapping groups)
  cell_count = np.full((n_cells[0], n_cells[1], 1), 0, dtype=int)

  for off_y in range(block_size[0]):
      for off_x in range(block_size[1]):
          gradients[off_y:n_cells[0] - block_size[0] + off_y + 1,
                    off_x:n_cells[1] - block_size[1] + off_x + 1] += \
              hog_feats[:, :, off_y, off_x, :]
          cell_count[off_y:n_cells[0] - block_size[0] + off_y + 1,
                    off_x:n_cells[1] - block_size[1] + off_x + 1] += 1
  return gradients.reshape(4*4*9)

def showImage(img):
  cv2.imshow('img',img)
  cv2.waitKey(0)
  cv2.destroyAllWindows()

def cropImage(img,x,y):
  crop_img = img[y:y+36, x:x+36]
  return crop_img

if __name__ == "__main__":
    

  facesPath = '../LabHOG/data/caltech_faces/Caltech_CropFaces/'
  faces = [f for f in listdir(facesPath) if isfile(join(facesPath, f))]
  print('loading faces')
  faces = [cv2.imread(facesPath+f,0) for f in faces]
  print('loaded faces')
  nonFacesRootPath = '../LabHOG/data/train_non_face_scenes/'
  nonFacesPaths = [f for f in listdir(nonFacesRootPath) if (isfile(join(nonFacesRootPath, f)) and (f.endswith('jpg')))]
  nonFaces = []

  for nonFacePath in nonFacesPaths:
    # print(nonFacePath)
    nf = cv2.imread(nonFacesRootPath+nonFacePath,0)
    # print(nf.shape)
    yjumps = nf.shape[0]//36
    xjumps = nf.shape[1]//36
    for i in range(xjumps):
      for j in range(yjumps):
        crop = cropImage(nf,i*36,j*36)
        nonFaces.append(crop)
    # break
  print(len(nonFaces))
  print(len(faces))

  faceFeatures = []

  print('extracting features')
  for f in faces:
    faceFeatures.append(list(HOGDescriptor(f))+[0])

  assert len(faceFeatures)==len(faces)

  nonFaceFeatures = []

  for f in nonFaces:
    nonFaceFeatures.append(list(HOGDescriptor(f))+[0])

  assert len(nonFaceFeatures)==len(nonFaces)

  print('features extracted ')

  train = faceFeatures+nonFaceFeatures

  waldo = cv2.imread('./waldo.png',0)
  waldo = cv2.resize(waldo,(36,36))
  # showImage(waldo)
  waldo = list(HOGDescriptor(waldo))+[1]
  # print(waldo)
  train = train+[waldo]*3000
  train = np.array(train)
  # print(train.shape)
  feats = train[:,:-1]
  labels = train[:,-1]
  # print(feats)
  # print(labels)

  clf = svm.SVC()
  clf.fit(feats, labels)
  print('is it waldo?')
  print(clf.predict([waldo[:-1]]))
  print('not waldo')
  input1 = [train[234,:-1]]
  # print(input1)
  print(clf.predict(input1))
  pickle.dump(clf,open('waldoClassifier.pkl','wb'))


  # print(faces)


  # print(HOGDescriptor(img))