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
from scipy import ndimage
from skimage.feature import peak_local_max
from skimage import filters


# Colour Spaces


def moveColor(image, color):
    choices = {'rgb': -10, 'lab': cv2.COLOR_RGB2LAB, 'hsv': cv2.COLOR_RGB2HSV, 'rgb+xy': -10,
               'lab+xy': cv2.COLOR_RGB2LAB, 'hsv+xy': cv2.COLOR_RGB2HSV, 'gray': cv2.COLOR_RGB2GRAY}

    transform = choices[color]
    # print('Transforming from rgb to {}'.format(transform))
    if transform != -10:
        image = cv2.cvtColor(image, transform)
    if color.endswith('xy'):
        print('Adding xy spatial dimensions')
        hlen, wlen, zlen = image.shape

        hm = np.zeros((hlen, wlen, 1))
        wm = np.zeros((hlen, wlen, 1))
        for i in range(hlen):
            hm[i, :, :] += i
        for j in range(wlen):
            wm[:, j, :] += j

        image = np.append(image, hm, axis=2)
        image = np.append(image, wm, axis=2)

    return normalizeImage(image)


def normalizeImage(image):
    # print("Image dimensions {} ".format(image.shape))
    normalized_image = cv2.normalize(
        image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    return normalized_image


def clusterImage(image, method, k):
    prediction = None
    nx, ny, channels = image.shape
    train = image.reshape((nx*ny, channels))  # vectors must live in RGB=R^3.
    if method == 'kmeans':
        kmeans = KMeans(k)
        kmeans.fit(train)
        prediction = kmeans.labels_.astype(np.int).reshape(nx, ny)
    if method == 'gmm':
        gmm = GaussianMixture(k)
        gmm.fit(train)
        prediction = gmm.predict(train).reshape(nx, ny)
    if method == 'hierarchical':
        connectivity = grid_to_graph(image.shape[0], image.shape[1])
        X = np.reshape(cv2.cvtColor(
            image[:, :, :3], cv2.COLOR_RGB2GRAY), (-1, 1))

        ward = AgglomerativeClustering(n_clusters=k,
                                       linkage='ward', connectivity=connectivity)
        ward = ward.fit(X)
        prediction = ward.labels_.astype(np.int).reshape(nx, ny)

    if method == 'watershed':
        gray = np.mean(image, axis=2)

        seeds = np.zeros(gray.shape, dtype=int)
        edges = filters.sobel(gray)
        coordinates = peak_local_max(gray, min_distance=20)
        print(coordinates)

        vals = np.zeros(coordinates.shape[0])
        for i in range(len(vals)):
            vals[i] = gray[coordinates[i][0]][coordinates[i][1]]

        ii = vals.argsort()
        coordinates = coordinates[ii]
        if k > len(vals):
            k = len(vals)
        for i in range(k):
            seeds[coordinates[i][0], coordinates[i][1]] = i+1

        prediction = watershed(edges, seeds)

    return prediction


def groundtruth(filename):
    import scipy.io as sio
    gth = sio.loadmat(filename.replace('jpg', 'mat'))
    seg = gth['groundTruth'][0, 5][0][0]['Segmentation']
    return seg


def evalMetric(originalLabels, predictedLabels):
    uniqueValsOriginal = len(np.unique(originalLabels))
    uniqueValsPredicted = len(np.unique(predictedLabels))
    # eval matrix init
    m = np.zeros((uniqueValsOriginal, uniqueValsPredicted))
    # image dims
    nx, ny = originalLabels.shape
    # build eval matrix
    for i in range(nx):
        for j in range(ny):
            m[originalLabels[i][j]-1][predictedLabels[i][j]-1] += 1
    # get max per row
    ans = 0
    for i in range(uniqueValsOriginal):
        ans += np.max(m[i])
    # normalize and return
    return ans/(nx*ny)


def segmentByClustering(image, color, k, method):
    # print('Segmenting image. Colorspace = {} Method = {} k = {}'.format(color,method,k))
    image = moveColor(image, color)
    # print('Clustering')
    segmentedImage = clusterImage(image, method, k)
    return segmentedImage


if __name__ == '__main__':
    # Initialize variables
    filename = 'BSDS_small/train/22090.jpg'
    colordos = 'lab'
    method = 'kmeans'
    n_clusters = 3

    image = imageio.imread(filename)

    gmm = segmentByClustering(image, 'lab+xy', 15, 'gmm')
    hcl = segmentByClustering(image, 'lab+xy', 15, 'hierarchical')
    ws = segmentByClustering(image, 'lab+xy', 15, 'watershed')
    gt = groundtruth(filename)

    print('GMM')
    print(evalMetric(gt, gmm))
    print('HCL')
    print(evalMetric(gt, hcl))

    figure = plt.figure(2)
    axis = figure.add_subplot(5, 1, 1)
    plt.imshow(hcl)
    plt.title('heirarchical')
    axis = figure.add_subplot(5, 1, 2)
    plt.imshow(gmm)
    plt.title('gaussians')
    axis = figure.add_subplot(5, 1, 3)
    plt.imshow(gt)
    plt.title('ground truth')
    axis = figure.add_subplot(5, 1, 4)
    plt.imshow(image)
    plt.title('original')
    axis = figure.add_subplot(5, 1, 5)
    plt.imshow(ws)
    plt.title('watershed')
    plt.show()
