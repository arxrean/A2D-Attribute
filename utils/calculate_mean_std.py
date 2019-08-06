import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader

dataset = datasets.ImageFolder('/mnt/lustre/jiangsu/dlar/home/zyk17/data/A2D/Release/pngs320H',
                               transform=transforms.ToTensor())

loader = DataLoader(
    dataset,
    batch_size=1,
    num_workers=0,
    shuffle=False
)


mean = 0.
std = 0.
nb_samples = 0.
for iter, data in enumerate(loader):
    print(iter)
    data = data[0]
    batch_samples = data.size(0)
    data = data.view(batch_samples, data.size(1), -1)
    mean += data.mean(2).sum(0)
    std += data.std(2).sum(0)
    nb_samples += batch_samples

mean /= nb_samples
std /= nb_samples
print('mean:{}'.format(mean))
print('std:{}'.format(std))
