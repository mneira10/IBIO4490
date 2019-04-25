import numpy as np
#import matplotlib.pyplot as plt
import tqdm 
import os 
from utils import  validate
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
num_classes = 10
num_epochs = 100
batch_size = 75
learning_rate = 0.01


train_dataset = dataloaders.CelebADataset('train')
val_dataset  = dataloaders.CelebADataset('val')


# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                           batch_size=batch_size, 
                                           shuffle=True)

val_loader = torch.utils.data.DataLoader(dataset=val_dataset, 
                                          batch_size=batch_size, 
                                          shuffle=False)

# Fully connected neural network with one hidden layer
# model = models.SimpleFeedForward(input_size, hidden_size, num_classes).to(device)

#simple convolutional nn
model = models.ConvNet2(num_classes).to(device)

#convolutional net with droput
# model = models.ConvNetDropout().to(device)




# Loss and optimizer
# criterion = nn.CrossEntropyLoss()
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  

#SCHEDULER
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')


# Train the model
min_val_loss = 9999999
total_step = len(train_loader)
for epoch in range(num_epochs):
    epochLoss = 0
    for i, (images, labels) in tqdm.tqdm(enumerate(train_loader),total=len(train_loader),desc='training..'):  
        
        # Move tensors to the configured device
        #for feed forward
        # images = images.reshape(-1, 48*48).float().to(device)
        
        # images = images.unsqueeze(1).float().to(device)
        # labels = labels.long().to(device)
        
        images = images.float().to(device)
        labels = labels.float().to(device)

        # Forward pass
     #   print('entra!')
        outputs = model(images)
        # print('outputs','labels')
        # print(outputs,labels)
        #print(outputs)
        #print(labels)
        loss = criterion(outputs, labels)
        epochLoss += outputs.shape[0] * loss.item()

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        
    val_loss = validate(model,val_loader,device,val_dataset)
    #for param_group in optimizer.param_groups:
    #    currentLr = param_group['lr']
    if val_loss < min_val_loss:
        min_val_loss = val_loss
        torch.save(model.state_dict(), 'bestCurrentModelLargeTrain.pth')
    print('Epoch [{}/{}] Loss: {:.4f} Test-loss: {:.4f}' .format(epoch+1,num_epochs,epochLoss/len(train_dataset),val_loss))
 
    #SCHEDULER 
    
    # scheduler.step(val_loss)


# Save the model checkpoint
torch.save(model.state_dict(), 'model.pth')


