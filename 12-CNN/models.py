import torch.nn as nn

class SimpleFeedForward(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(SimpleFeedForward, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size) 
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)  
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

class ConvNet(nn.Module):
    def __init__(self, num_classes=7):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(12*12*32, num_classes)

    def forward(self, x):
        # print(x.shape)
        out = self.layer1(x)
        # print(out.shape)
        out = self.layer2(out)
        # print(out.shape)
        out = out.reshape(out.size(0), -1)
        # print(out.shape)
        out = self.fc(out)
        # print(out.shape)
        return out

class ConvNet2(nn.Module):
    def __init__(self, num_classes=7):
        super(ConvNet2, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.fc = nn.Linear(22*27*32, 500)
        self.fc2 = nn.Linear(500, num_classes)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()


    def forward(self, x):
        # print('x',x.shape)
        out = self.layer1(x)
        # print('l1',out.shape)
        out = self.layer2(out)
        # print('l2',out.shape)
        out = self.layer3(out)
        # print('l3',out.shape)
        out = out.reshape(out.size(0), -1)
        # print('reshape',out.shape)
        out = self.relu(self.fc(out))
        # print('final',out.shape)
        out = self.fc2(out)
        # print(out)
        out = self.softmax(out)
        return out

class ConvNet2Large(nn.Module):
    def __init__(self, num_classes=7):
        super(ConvNet2Large, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 128, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.fc = nn.Linear(6*6*128, 512)
        self.fc2 = nn.Linear(512, num_classes)

        self.relu = nn.ReLU()


    def forward(self, x):
        #print('x',x.shape)
        out = self.layer1(x)
        #print('l1',out.shape)
        out = self.layer2(out)
        #print('l2',out.shape)
        out = self.layer3(out)
        #print('l3',out.shape)
        out = out.reshape(out.size(0), -1)
        #print('reshape',out.shape)
        out = self.relu(self.fc(out))
        #print('final',out.shape)
        out = self.fc2(out)
        return out

class ConvNetDropout(nn.Module):
    def __init__(self, num_classes=7):
        super(ConvNetDropout, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5, stride=1, padding=2),
            nn.Dropout(0.5),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=5, stride=1, padding=2),
            nn.Dropout(0.5),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 128, kernel_size=5, stride=1, padding=2),
            nn.Dropout(0.5),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.fc = nn.Sequential(
            nn.Linear(6*6*128, 500),
            nn.Dropout(0.5),
            nn.ReLU())
        self.fc2 = nn.Sequential(
            nn.Linear(500, num_classes),
            )

        


    def forward(self, x):
        #print('x',x.shape)
        out = self.layer1(x)
        #print('l1',out.shape)
        out = self.layer2(out)
        #print('l2',out.shape)
        out = self.layer3(out)
        #print('l3',out.shape)
        out = out.reshape(out.size(0), -1)
        #print('reshape',out.shape)
        out = self.fc(out)
        #print('final',out.shape)
        out = self.fc2(out)
        return out
