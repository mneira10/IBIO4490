import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn
import pickle

def plotCM(CM,name):
    df_cm = pd.DataFrame(CM, index = [i for i in range(10)],
                  columns = [i for i in range(10)])
    plt.figure(figsize = (10,7))
    sn.heatmap(df_cm, annot=True)
    plt.savefig(name+'.png')
    plt.close()

def normalizeCM(CM):
  # CM.astype(float)
  # for i in range(CM.shape[0]):
  #   sumCols= float(np.sum(CM[:,i]))
  #   CM[:,i] = CM[:,i]/sumCols
  # return CM
  return CM / CM.sum(axis=0)
def fileExists(path):
    return Path(path).exists()
def toPickle(obj, name):
    pickle.dump(obj, open(name+'.pkl', "wb"))


def loadPickle(name):
    return pickle.load(open(name, "rb"))


trainCM = loadPickle('../data/trainConfusionMatrix.pkl')
testCM  = loadPickle('../data/testConfusionMatrix.pkl')
print(trainCM)
print(testCM)
print(type(trainCM))
print(type(testCM))
trainCM = normalizeCM(trainCM)
testCM = normalizeCM(testCM)
print(trainCM)
print(testCM)
plotCM(trainCM,'trainCM')
plotCM(testCM,'testCM')