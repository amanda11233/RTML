
import torchvision
import torch
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader


preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])


train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download= True, transform = preprocess)

train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [40000, 10000])


test_dataset = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=preprocess)

batch_size = 16

train_dataloader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True, num_workers=2)
valid_dataloader = DataLoader(val_dataset, batch_size = batch_size, shuffle = False, num_workers = 2)
test_dataloader = DataLoader(test_dataset, batch_size = batch_size, shuffle = False, num_workers = 2)
