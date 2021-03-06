import numpy as np
#import matplotlib.pyplot as plt
import tqdm 
import os 
from utils import get_data, validate
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import dataloaders
import models


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters 
input_size = 48*48
hidden_size = 500
num_classes = 7
num_epochs = 100
batch_size = 200
learning_rate = 0.001


train_dataset = dataloaders.Fer2013Dataset()
test_dataset  = dataloaders.Fer2013Dataset(test=True)


# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                           batch_size=batch_size, 
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                          batch_size=batch_size, 
                                          shuffle=False)

# Fully connected neural network with one hidden layer
# model = models.SimpleFeedForward(input_size, hidden_size, num_classes).to(device)

#simple convolutional nn
# model = models.ConvNet2().to(device)

#convolutional net with droput
model = models.ConvNetDropout().to(device)




# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  

#SCHEDULER
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')


# Train the model

total_step = len(train_loader)
for epoch in range(num_epochs):
    epochLoss = 0
    for i, (images, labels) in enumerate(train_loader):  
        
        # Move tensors to the configured device
        #for feed forward
        # images = images.reshape(-1, 48*48).float().to(device)
        images = images.unsqueeze(1).float().to(device)
        labels = labels.long().to(device)
        
        # Forward pass
        outputs = model(images)
        #print(outputs)
        #print(labels)
        loss = criterion(outputs, labels)
        epochLoss += outputs.shape[0] * loss.item()

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        
    acca,val_loss = validate(model,test_loader,device,test_dataset)
    for param_group in optimizer.param_groups:
        currentLr = param_group['lr']

    print('Epoch [{}/{}] Loss: {:.4f} Test-loss: {:.4f} Test ACCA: {:.2f}% lr: {}' .format(epoch+1,num_epochs,epochLoss/len(train_dataset),val_loss,acca*100,currentLr))
 
    #SCHEDULER 
    
    scheduler.step(val_loss)


# Save the model checkpoint
torch.save(model.state_dict(), 'model.pth')


