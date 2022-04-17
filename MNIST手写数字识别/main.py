import torch
from torch import nn
import torchvision
import matplotlib.pyplot as plt
import sys
import d2lzh_pytorch as d21
import Get_data #读取数据的文件
import VGG_NET #获取神经网络
import Train
import Test
device = "cuda:0" if torch.cuda.is_available() else "cpu"        #设置GPU设备
batch_size = 1000                                                 #设置mini——batch的大小
train_iter,test_iter = Get_data.get_data(batch_size=batch_size)  #接收数据作为迭代器
net = VGG_NET.Get_Vgg()                                          #接受网络
lr,num_epochs = 0.0008,30
optimizer = torch.optim.Adam(net.parameters(),lr=lr)
Train.train(net, train_iter,test_iter, batch_size, optimizer, device, num_epochs)



