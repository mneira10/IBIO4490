
import numpy as np
import torch 
import torch.nn as nn


def get_data():
    # angry, disgust, fear, happy, sad, surprise, neutral
    with open("fer2013.csv") as f:
        content = f.readlines()

    lines = np.array(content)
    num_of_instances = lines.size
    print("number of instances: ", num_of_instances)
    print("instance length: ", len(lines[1].split(",")[1].split(" ")))

    x_train, y_train, x_test, y_test = [], [], [], []

    for i in range(1, num_of_instances):
        emotion, img, usage = lines[i].split(",")
        pixels = np.array(img.split(" "), 'float32')
        emotion = 1 if int(emotion) == 3 else 0  # Only for happiness
        if 'Training' in usage:
            y_train.append(emotion)
            x_train.append(pixels)
        elif 'PublicTest' in usage:
            y_test.append(emotion)
            x_test.append(pixels)

    #------------------------------
    # data transformation for train and test sets
    x_train = np.array(x_train, 'float64')
    y_train = np.array(y_train, 'float64')
    x_test = np.array(x_test, 'float64')
    y_test = np.array(y_test, 'float64')

    x_train /= 255  # normalize inputs between [0, 1]
    x_test /= 255

    x_train = x_train.reshape(x_train.shape[0], 48, 48)
    x_test = x_test.reshape(x_test.shape[0], 48, 48)
    y_train = y_train.reshape(y_train.shape[0], 1)
    y_test = y_test.reshape(y_test.shape[0], 1)

    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    # plt.hist(y_train, max(y_train)+1); plt.show()

    return x_train, y_train, x_test, y_test

def validate(model,test_loader,device,test_dataset):
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        model.eval()
        correct = 0
        total = 0
        epochLoss = 0
        for images, labels in test_loader:
            #images = images.reshape(-1, 48*48).float().to(device)
            images = images.unsqueeze(1).float().to(device)
            labels = labels.long().to(device)
            
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
