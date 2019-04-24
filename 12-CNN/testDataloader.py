import dataloaders

data = dataloaders.CelebADataset('test')


print(data.__len__())
print(data.__getitem__(0))