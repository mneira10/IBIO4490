
import numpy as np
import torch 
import torch.nn as nn
import tqdm

def validate(model,test_loader,device,test_dataset):
    criterion = nn.BCELoss()
    with torch.no_grad():
        model.eval()
        correct = 0
        total = 0
        epochLoss = 0
        for i, (images, labels) in tqdm.tqdm(enumerate(test_loader),total=len(test_loader),desc='evaulating...'):
            #images = images.reshape(-1, 48*48).float().to(device)
            images = images.float().to(device)
            labels = labels.float().to(device)
            # images = images.reshape(-1, 28*28).to(device)
            # labels = labels.to(device)
            outputs = model(images)

            #loss
            loss = criterion(outputs, labels)
            epochLoss += outputs.shape[0] * loss.item()

            #_, predicted = torch.max(outputs.data, 1)
            #total += labels.size(0)
            #print(outputs)
            #print(labels)
            #correct += (predicted == labels).sum().item()

        # print('Accuracy of the network on the test images: {} %'.format(100 * correct / total))
        model.train()
        return epochLoss/len(test_dataset)
