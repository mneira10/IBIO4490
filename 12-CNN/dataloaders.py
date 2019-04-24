import numpy as np
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import cv2


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

        self.ids = list(partitions[img_id_str])

        labels = pd.read_csv('./data/list_attr_celeba.csv')
        labels = labels[[img_id_str]+attributes]
        labels = labels[labels[img_id_str].isin(self.ids)]

        self.labels = labels

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):

        imgName = self.ids[idx]
        # (218, 178, 3)
        img = cv2.imread('./data/img_align_celeba/'+imgName)
        lab = self.labels[self.labels.image_id == imgName]

        label = lab[self.attributes].values[0]

        # sample = {'image':,'label':self.y[idx][0]}

        # if self.transform:
        #     sample = self.transform(sample)

        return (img, label)
