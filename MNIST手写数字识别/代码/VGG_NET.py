import torch
from torch import nn
class Get_Vgg(nn.Module):
    def __init__(self):
        super(Get_Vgg,self).__init__()
        #第一个卷积层
        self.ConV1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        #第二个卷积层
        self.ConV2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        # 第三个卷积层
        self.ConV3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1)
        )
        #全连接层
        self.FC = nn.Sequential(
            nn.Linear(in_features=2304, out_features=500, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=500, out_features=500, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=500, out_features=10, bias=True),
            # nn.Softmax()
        )
    def forward(self,x):
        feature_1 = self.ConV1(x)
        feature_2 = self.ConV2(feature_1)
        feature_3 = self.ConV3(feature_2)
        out_put = self.FC(feature_3.view(x.shape[0],-1))
        return out_put