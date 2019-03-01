import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn


def plotCM(CM,name):
    df_cm = pd.DataFrame(CM, index = [i for i in range(10)],
                  columns = [i for i in range(10)])
    plt.figure(figsize = (10,7))
    sn.heatmap(df_cm, annot=True)
    sn.savefig(name+'.png')
    plt.close()


