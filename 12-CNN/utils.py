
import numpy as np
import torch 
import torch.nn as nn


def validate(model,test_loader,device,test_dataset):
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        model.eval()
        correct = 0
        total = 0
        epochLoss = 0
        for images, labels in test_loader:
            #images = images.reshape(-1, 48*48).float().to(device)
            images = images.unsqueeze(1).to(device)
            labels = labels.to(device)
            
            # images = images.reshape(-1, 28*28).to(device)
            # labels = labels.to(device)
            outputs = model(images)

            #loss
            loss = criterion(outputs, labels)
            epochLoss += outputs.shape[0] * loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        # print('Accuracy of the network on the test images: {} %'.format(100 * correct / total))
        model.train()
        return correct / total, epochLoss/len(test_dataset)
