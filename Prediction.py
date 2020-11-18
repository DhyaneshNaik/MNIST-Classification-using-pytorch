import torch
import matplotlib.pyplot as plt
from torchvision.transforms import  transforms
from torch.utils.data import Dataset,DataLoader
from DatasetCreation import *

model = torch.load("entire_model.pt")

Image_size = 16
composed = transforms.Compose([transforms.Resize((Image_size,Image_size)),transforms.ToTensor()])

val_data = Dataset(train=False,transforms=composed)
val_loader = DataLoader(val_data,batch_size=1)
i=0
for x,y in val_loader:
    if i > 10:
        break
    z = model(x)
    _,yhat = torch.max(z.data,1)
    plt.imshow(x.numpy().reshape(Image_size,Image_size),cmap='gray')
    plt.title(f"y = {y}, Predicted = {yhat}")
    plt.show()
    print((yhat==y))
    i+=1
