import numpy as np 
from torch.utils.data import Dataset, DataLoader

class Fer2013Dataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, test = False, transform=None):
        with open("fer2013.csv") as f:
          content = f.readlines()

        lines = np.array(content)
        num_of_instances = lines.size
        # print("number of instances: ", num_of_instances)
        # print("instance length: ", len(lines[1].split(",")[1].split(" ")))

        # x_train, y_train, x_test, y_test = [], [], [], []
        x, y = [], []

        for i in range(1, num_of_instances):
            emotion, img, usage = lines[i].split(",")
            pixels = np.array(img.split(" "), 'float32')
            # emotion = 1 if int(emotion) == 3 else 0  # Only for happiness
            if test:
              if 'PublicTest' in usage:
                  y.append(emotion)
                  x.append(pixels)
            else: 
               if 'Training' in usage:
                  y.append(emotion)
                  x.append(pixels)


        #------------------------------
        # data transformation for train and test sets
        # x_train = np.array(x_train, 'float64')
        # y_train = np.array(y_train, 'float64')
        # x_test = np.array(x_test, 'float64')
        # y_test = np.array(y_test, 'float64')

        x = np.array(x, 'float64')
        y = np.array(y, 'float64')

        # x_train /= 255  # normalize inputs between [0, 1]
        # x_test /= 255

        # x_train = x_train.reshape(x_train.shape[0], 48, 48)
        # x_test = x_test.reshape(x_test.shape[0], 48, 48)
        # y_train = y_train.reshape(y_train.shape[0], 1)
        # y_test = y_test.reshape(y_test.shape[0], 1)

        x = x.reshape(x.shape[0], 48, 48)
        y = y.reshape(y.shape[0], 1)

        if test:
          print(x.shape[0], 'test samples')
        else:
          print(x.shape[0], 'training samples')

        # plt.hist(y_train, max(y_train)+1); plt.show()

        # return x_train, y_train, x_test, y_test

        self.x = x
        self.y = y
        self.transform = transform

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, idx):
        # img_name = os.path.join(self.root_dir,
        #                         self.landmarks_frame.iloc[idx, 0])
        # image = io.imread(img_name)
        # landmarks = self.landmarks_frame.iloc[idx, 1:].as_matrix()
        # landmarks = landmarks.astype('float').reshape(-1, 2)
        # sample = {'image': image, 'landmarks': landmarks}

        sample = {'image':self.x[idx],'label':self.y[idx][0]}

        # if self.transform:
        #     sample = self.transform(sample)

        return (self.x[idx], self.y[idx][0])