import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self,out1=16,out2=32,out3=64,out4=128):
        super(CNN,self).__init__()
        self.cnn1 = nn.Conv2d(in_channels=1,out_channels=out1,kernel_size=5,padding=2)
        self.conv1_bn = nn.BatchNorm2d(out1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2,2))
        self.dropout1 = nn.Dropout(0.25)

        self.cnn2 = nn.Conv2d(in_channels=out1, out_channels=out2, kernel_size=5, padding=2)
        self.conv2_bn = nn.BatchNorm2d(out2)
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 2))
        self.dropout2 = nn.Dropout(0.25)

        self.cnn3 = nn.Conv2d(in_channels=out2, out_channels=out3, kernel_size=3, padding=2)
        self.conv3_bn = nn.BatchNorm2d(out3)
        self.maxpool3 = nn.MaxPool2d(kernel_size=(2, 2))
        self.dropout3 = nn.Dropout(0.25)

        self.cnn4 = nn.Conv2d(in_channels=out3, out_channels=out4, kernel_size=3, padding=2)
        self.conv4_bn = nn.BatchNorm2d(out4)
        self.maxpool4 = nn.MaxPool2d(kernel_size=(2, 2))
        self.dropout4 = nn.Dropout(0.25)

        self.fc1 = nn.Linear(in_features=out3 * 8,out_features=out3 * 8)
        self.dropout5 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(in_features=out3 * 8, out_features= 10)

        self.conv5_bn = nn.BatchNorm1d(10)

    def forward(self, x):
        x = self.cnn1(x)
        x = self.conv1_bn(x)
        x = torch.relu(x)
        x = self.maxpool1(x)
        x = self.dropout1(x)

        x = self.cnn2(x)
        x = self.conv2_bn(x)
        x = torch.relu(x)
        x = self.maxpool2(x)
        x = self.dropout2(x)

        x = self.cnn3(x)
        x = self.conv3_bn(x)
        x = torch.relu(x)
        x = self.maxpool3(x)
        x = self.dropout3(x)

        x = self.cnn4(x)
        x = self.conv4_bn(x)
        x = torch.relu(x)
        x = self.maxpool4(x)
        x = self.dropout4(x)

        x = x.view(x.size(0),-1)

        x = self.fc1(x)
        x = torch.relu(x)
        x = self.dropout5(x)

        x = self.fc2(x)
        x = self.conv5_bn(x)

        return x