import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 

data = pd.read_csv('./resultsData.dat')
# fig  = plt.figure(figsize=(20,15))
for alg in ['knn','rf']:
  algData = data[data['alg']==alg]
  for n in [3,5,10,20,50]:
    # color= ""
    name = "Rand. Forest" if alg == 'rf' else "KNN"
    color =('red' if alg =='knn' else 'blue')
    parameter = '# trees' if alg == 'rf' else '# neigh.'
  
    nData = algData[algData['n']==n]

    plt.scatter(nData['k'],nData['precision'] ,label = '{} {}={}'.format(name,parameter,str(n)),color=color)
  
plt.xlim(0,1000)
plt.legend()
plt.xlabel('# of textons')
plt.ylabel('Average Precision')
plt.savefig('avePrecisionPerAlg.png')