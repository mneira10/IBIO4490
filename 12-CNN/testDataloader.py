import dataloaders

data = dataloaders.CelebADataset('test')


# print(data.__len__())
# print(data.__getitem__(0))
img,lab = data.__getitem__(0)

print(img.shape)