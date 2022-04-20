import torch
from torch import nn
import torchvision
import matplotlib.pyplot as plt
import sys
import d2lzh_pytorch as d21
import Get_data
import Image_show
import Res_Net
import Train
device = "cuda:0" if torch.cuda.is_available() else "cpu"        #设置GPU设备
Train_save = 'C:/Users/Mr.Guo/Desktop/猫狗识别/image-and-labe/Train_data'       #自建存储训练集图片的位置
Test_save = 'C:/Users/Mr.Guo/Desktop/猫狗识别/image-and-labe/Test_data'       #自建存储测试集图片的位置
Train_datas = Get_data.Get_data(Train_save)                                 #获取数据
Test_datas = Get_data.Get_data(Test_save)
batch_size = 50
lr,num_epochs = 0.0008,20
train_iter = torch.utils.data.DataLoader(Train_datas,batch_size = batch_size,shuffle=True)  #返回迭代器
test_iter = torch.utils.data.DataLoader(Train_datas,batch_size = batch_size,shuffle=True)
net = Res_Net.RESNET()
optimizer = torch.optim.Adam(net.parameters(),lr=lr)
Train.train(net, train_iter,test_iter, batch_size, optimizer, device, num_epochs)
#保存模型和模型参数
torch.save(net.state_dict(),'./net_params.pth')
torch.save(net,'./net.pth')