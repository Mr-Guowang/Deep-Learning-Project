import random
import torch
import numpy as np
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch import optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import GET_DATA
import GET_NET
import TRAIN
from tqdm import tqdm
Path = 'C:/Users/Mr.Guo/Desktop/实验数据/faces'
device = "cuda:0" if torch.cuda.is_available() else "cpu"        #设置GPU设备

batch_size,z_dim,lr,n_epochs,n_critic, clip_value = 64,100,1e-4,10,1,0.01      # Training hyperparameters
z_sample = Variable(torch.randn(100, z_dim)).to(device) #z为随机生成64*100的高斯分布数据（均值为0，方差为1）也叫噪声

G = GET_NET.Generator(in_dim=z_dim).to(device)               # Model
D = GET_NET.Discriminator(3).to(device)
G.train()
D.train()
'''
model.train()是保证BN层用每一批数据的均值和方差，而model.eval()是保证BN用全部训练数据的均值和方差；
对于BN，因为所有的数据一开始就已经归一化过了，所以测试的时候为了准确性，所以要关掉它
每一次mini-batch，我个人认为是为了加速训练，使这个批次拥有统一的分布
而对于Dropout，model.train()是随机取一部分网络连接来训练更新参数，而model.eval()是利用到了所有网络连接。
在训练中，每个隐层的神经元先乘概率P，然后在进行激活，等于说概率小于多少直接归零
在测试中，所有的神经元先进行激活，然后每个隐层神经元的输出乘P。
'''

criterion = nn.BCELoss()                       # Loss损失函数使用二元交叉熵损失""" Medium: Use RMSprop for WGAN. """

opt_D = torch.optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))  #使用Adam优化器更新参数Adam+pytorch，学习率设置为0.0002 Betal=0.5。
opt_G = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
# opt_D = torch.optim.RMSprop(D.parameters(), lr=lr)
# opt_G = torch.optim.RMSprop(G.parameters(), lr=lr)

# DataLoader
dataset = GET_DATA.get_dataset(Path)
if __name__ == "__main__":   #使用多线程的时候需要这样，不然会报错
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    TRAIN.train(G,D,opt_G,opt_D,n_epochs,criterion,dataloader,z_dim,n_critic,z_sample)






