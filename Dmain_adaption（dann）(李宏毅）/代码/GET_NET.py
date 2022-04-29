import torch
from torch import nn
class FeatureExtractor(nn.Module):

    def __init__(self):
        super(FeatureExtractor, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1),   #batch *64*32*32
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),              #batch *64*16*16

            nn.Conv2d(64, 128, 3, 1, 1),      #batch *128*16*16
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),                   #batch *128*8*8

            nn.Conv2d(128, 256, 3, 1, 1),        #batch *256*8*8
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),                    #batch *256*4*4

            nn.Conv2d(256, 256, 3, 1, 1),          #batch *256*4*4
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),                        #batch *256*2*2

            nn.Conv2d(256, 512, 3, 1, 1),        #batch *512*2*2
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2)                  #batch *512*1*1
        )

    def forward(self, x):
        x = self.conv(x).squeeze()  #  squeeze() 这个函数的作用是去掉矩阵里维度为1的维度
        '''
        比如我做个小实验
        net = FeatureExtractor()
        x = torch.rand(3, 1, 32, 32)
        print(net(x).shape)
        结果如下：
        torch.Size([3, 512])
        '''
        return x                     #最后返回的结果为batch*512


class LabelPredictor(nn.Module):

    def __init__(self):
        super(LabelPredictor, self).__init__()

        self.layer = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.ReLU(),

            nn.Linear(512, 10),             #最后输出10个类别，也就是判别器作用
        )

    def forward(self, h):
        c = self.layer(h)
        return c


class DomainClassifier(nn.Module):

    def __init__(self):
        super(DomainClassifier, self).__init__()

        self.layer = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 1),       #最后输出1个类别，分辨是sourse还是target
        )

    def forward(self, h):
        y = self.layer(h)
        return y
