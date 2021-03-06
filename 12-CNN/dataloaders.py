import numpy as np
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import cv2
from skimage import io, transform


class CelebADataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, partition='-', transform=None):

        #img_id, partition
        partitions = pd.read_csv('./data/list_eval_partition.csv')

        partId = None
        if partition == 'train':
            partId = 0
        elif partition == 'val':
            partId = 1
        elif partition == 'test':
            partId = 2
        else:
            raise Exception(
                'Partition invalid. Please input train, test or val.')

        attributes = ['Eyeglasses', 'Bangs', 'Black_Hair', 'Blond_Hair',
                      'Brown_Hair', 'Gray_Hair', 'Male', 'Pale_Skin', 'Smiling', 'Young']
        img_id_str = 'image_id'
        self.attributes = attributes

        partitions = partitions[partitions.partition == partId]

        self.ids = np.array(list(partitions[img_id_str]))

        if partition =='train':
            self.ids = np.random.choice(self.ids,len(self.ids)//5,replace=False)
        labels = pd.read_csv('./data/list_attr_celeba.csv')
        labels = labels[[img_id_str]+attributes]
        labels = labels[labels[img_id_str].isin(self.ids)]

        self.labels = labels

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):

        imgName = self.ids[idx]
        # (218, 178, 3)
        #img = cv2.imread('./data/img_align_celeba/'+imgName)
        #img = np.swapaxes(img,0,2)
        img=np.asarray(io.imread('./data/img_align_celeba/'+imgName))
        img = np.swapaxes(img,0,2)
        lab = self.labels[self.labels.image_id == imgName]

        label = lab[self.attributes].values[0]
        label[label==-1]=0
        # sample = {'image':,'label':self.y[idx][0]}

        # if self.transform:
        #     sample = self.transform(sample)

        return (img, label)
