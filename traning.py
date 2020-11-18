from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from TraninModel import *
from ModelCreation import *
from DatasetCreation import *
from torchvision.transforms import transforms

Image_size = 16
composed = transforms.Compose([transforms.Resize((Image_size,Image_size)),transforms.ToTensor()])

train_data = Dataset(train=True,transforms=composed)
val_data = Dataset(train=False,transforms=composed)
print(len(train_data))
print(len(val_data))
path = "entire_model.pt"
train_loader = DataLoader(train_data,batch_size=100)
val_loader = DataLoader(val_data,batch_size=5000)
model = CNN()
optimizer = torch.optim.SGD(model.parameters(),lr=0.1)
criterion = nn.CrossEntropyLoss()
n_epoch = 5
n_test = len(val_data)

#plt.imshow(train_data[0])
#plt.show()
#print(type(train_data))
#print(train_data)

#for x,y in train_loader:
#    print(x,y)
#    exit

modelobj = train_model(model=model,train_loader=train_loader,val_loader=val_loader,optimizer=optimizer,criterion= criterion,n_test=n_test,n_epochs=n_epoch,path=path)

accuracy,loss = modelobj.train()
print(accuracy)


plt.plot(loss,'r',label='Loss')
plt.xlabel("Iteration")
plt.title("Loss")
plt.legend()
fig1 = plt.gcf()
plt.show()
plt.draw()
fig1.savefig('Loss.png', dpi=100)


plt.plot(accuracy,'r',label='Accuracy')
plt.xlabel("Epochs")
plt.title("Accuracy")
plt.legend()
fig2 = plt.gcf()
plt.show()
plt.draw()
fig2.savefig('Accuracy.png', dpi=100)