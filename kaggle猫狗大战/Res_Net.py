#此处构建卷积神经网络的模型(resnet)
import torch
from torch import nn
class Res_Block(nn.Module):
    def __init__(self,IN_chanels,OUT_chanels,Stride):
        super(Res_Block, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=IN_chanels, out_channels=OUT_chanels, kernel_size=3, stride=Stride, padding=1, bias=False),
            nn.BatchNorm2d(OUT_chanels),
            nn.ReLU(),
            nn.Conv2d(in_channels=OUT_chanels, out_channels=OUT_chanels, kernel_size=3, stride=1, padding=1,bias=False),
            nn.BatchNorm2d(OUT_chanels),
        )
        if Stride != 1 or IN_chanels != OUT_chanels:   #这里是残差块，如果输入和输出的尺寸或通道不同，必须进行调整
            self.shortcut = nn.Sequential(
                nn.Conv2d(IN_chanels, OUT_chanels, kernel_size=1, stride=Stride),
                nn.BatchNorm2d(OUT_chanels)
            )
        else:
            self.shortcut = nn.Sequential()
    def forward(self,x):
        y = self.conv1(x)
        y += self.shortcut(x)
        y = nn.functional.relu(y)
        return y
class RESNET(nn.Module):            #这里决定了我们使用多少层的网络
    def __init__(self):
        super(RESNET,self).__init__()
        self.Block0 = nn.Sequential(                                   #  输入224*224*3
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=0),      #  222*222*32
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),          #111*111*32
        )
        self.Res = nn.Sequential(
            Res_Block(32, 32, 1) , # 111*111*32
            Res_Block(32, 32, 1) , # 111*111*32
            Res_Block(32, 64, 2) , # 56*56*64
            Res_Block(64, 64, 1) , # 56*56*64
            Res_Block(64, 128, 2) , # 28*28*128
            Res_Block(128, 128, 1),  # 28*28*128
            Res_Block(128, 256, 2) , # 14*14*256
            Res_Block(256, 256, 1) , # 14*14*256
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),  # 7*7*256
        )
        self.fc1 = nn.Linear(12544,2)
    def forward(self,x):
        y = self.Block0(x)
        y = self.Res(y)
        y = self.fc1(y.view(y.shape[0],-1))
        return y                                  #返回的是一个（1，2）的tensor

