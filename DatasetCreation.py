import torchvision.datasets as dsets
from torch.utils.data import Dataset

class Dataset(Dataset):
    def __init__(self,train = True,transforms=None):
        directory = "./data"
        self.dataset = dsets.MNIST(root=directory,train=train,download=True,transform=transforms)
        #self.transform = transforms

    def __len__(self):
        self.len = len(self.dataset)
        return self.len

    def __getitem__(self, idx):
        images = self.dataset[idx][0]
        Y = self.dataset[idx][1]
        return images,Y