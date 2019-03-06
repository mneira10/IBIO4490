from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import imageio
import cv2 as cv2
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from skimage.morphology import watershed
from sklearn.feature_extraction.image import grid_to_graph



# Initialize variables
filename = 'BSDS_small/train/22090.jpg'
colordos = 'lab'
method = 'kmeans'
n_clusters = 3

## Colour Spaces

def moveColor(image, color):
    choices={'rgb':-10,'lab':cv2.COLOR_RGB2LAB,'hsv':cv2.COLOR_RGB2HSV,'rgb+xy':-10,'lab+xy':cv2.COLOR_RGB2LAB,'hsv+xy':cv2.COLOR_RGB2HSV,'gray': cv2.COLOR_RGB2GRAY}


    transform = choices[color]
    if transform != -10:
        image = cv2.cvtColor(image, transform)
    if color.endswith('xy'):
        hlen,wlen,zlen = image.shape
        
        hm = np.zeros((hlen,wlen,1))
        wm = np.zeros((hlen,wlen,1))
        for i in range(hlen):
            hm[i,:,:] +=i
        for j in range(wlen):
            wm[:,j,:] +=j

        image = np.append(image,hm,axis=2)
        image = np.append(image,wm,axis=2)
        print(image.shape)

        # image = np.concatenate((image,wm),axis=2)



    return normalizeImage(image)    
    

def normalizeImage(image):
    print("Image dimensions {} ".format(image.shape))
    # if(len(image.shape)>2):
    #     for i in range(image.shape[2]):
    #         fig = plt.figure(figsize=(20,15))
    #         plt.hist(image[:,:,i].flatten())
    #         plt.show()
    #         plt.close()
    normalized_image = cv2.normalize(image, None, alpha = 0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    return normalized_image
## Clustering


 
def clusterImage(image, method,k):
    # choices=['kmeans','gmm',hierarchical','watershed']
    #scikit-learn expects 2D num arrays for the training set for a fit function.
    #stackoverflow/questions/34972142
    prediction = None 
    channels = image.shape[2]
    train = image.reshape((nx*ny,channels)) # vectors must live in RGB=R^3.
    if method == 'kmeans':
        kmeans = KMeans(k)
        kmeans.fit(train)
        prediction = kmeans.labels_.astype(np.int).reshape(nx,ny)
    if method == 'gmm':
        gmm = GaussianMixture(k)
        gmm.fit(train)
        prediction = gmm.predict(train).reshape(nx,ny)
    if method == 'hierarchical':
        connectivity = grid_to_graph(image.shape[0],image.shape[1])
        X = np.reshape(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY), (-1, 1))

        ward = AgglomerativeClustering(n_clusters=k,
                linkage='ward', connectivity=connectivity)
        ward = ward.fit(X)
        prediction = ward.labels_.astype(np.int).reshape(nx,ny)
        
#   TODO
#    if method == 'watershed':
#        return watershed
    return prediction

#def buildPrediction(labels,[x,y,z]):
#    if hasattr(,'labels_'):
#        prediction = 
#    return

def groundtruth():
    import scipy.io as sio
    gth = sio.loadmat(filename.replace('jpg','mat'))
    seg = gth['groundTruth'][0,5][0][0]['Segmentation']
    return seg

## Debugging

image = imageio.imread(filename)
nx, ny, channels = image.shape
# rgbImage = moveColor(image,'rgb')
# hsvImage = moveColor(image,'hsv')
# labImage = moveColor(image,'lab')
# grayImage = moveColor(image,'gray')

# figure = plt.figure(1)
# axis = figure.add_subplot(3,1,1)
# plt.imshow(labImage)
# axis = figure.add_subplot(3,1,2)
# plt.imshow(hsvImage)
# axis = figure.add_subplot(3,1,3)
# plt.imshow(grayImage,cmap='gray')
# plt.show()

# kmeans = clusterImage(image,'kmeans')
#Al final no recupero los centroides; unicamente las clases.
#kmeans_centers = kmeans.cluster_centers_
#kmeans_labels = kmeans.labels_
gmm = clusterImage(image,'gmm',3)
# No memory for HCL :(
hcl = clusterImage(image,'hierarchical',5)
segmentation = groundtruth()

figure = plt.figure(2)
axis = figure.add_subplot(4,1,1)
plt.imshow(hcl)

axis = figure.add_subplot(4,1,2)
plt.imshow(gmm)
axis = figure.add_subplot(4,1,3)
plt.imshow(segmentation)
axis = figure.add_subplot(4,1,4)
plt.imshow(image)
plt.show()

def segmentByClustering( rgbImage, colorSpace, clusteringMethod, numberOfClusters):
    return segmentation