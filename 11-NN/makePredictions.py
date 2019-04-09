from os import listdir
from os.path import isfile, join
import cv2
import tqdm
import models
import argparse
import torch

parser = argparse.ArgumentParser()

parser.add_argument('--modelPath', type=str, required=True)
# parser.add_argument('--k', type=int, default=4)
# parser.add_argument('--method', type=str, default='watershed', choices=['kmeans', 'gmm', 'hierarchical', 'watershed'])
# parser.add_argument('--img_file', type=str, required=True)

opts = parser.parse_args()

print("loading model {}".format(opts.modelPath))
model = models.ConvNet2()
model.load_state_dict(torch.load(opts.modelPath))
model.eval()

mypath = './cleanImages/'
images = [f for f in listdir(mypath) if isfile(join(mypath, f))]
images.sort()

f= open("predictions.txt","w+")
for im in images:
  image = cv2.imread(mypath+im)
  image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  image = torch.from_numpy(image).float()
  image = image.unsqueeze(0).unsqueeze(0)
  
  output = model(image)
  _,predicted = torch.max(output,1)
  predicted = predicted[0].numpy()
  f.write(im.split('.')[0]+','+str(predicted)+'\n')
f.close()
