import torch
import os
import glob
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
class CrypkoDataset(Dataset):                     #继承dataset类来进行图片读取
    def __init__(self, fnames, transform):
        self.transform = transform                #读取transfrom
        self.fnames = fnames                      #存一下图片路径的列表（看了上下文代码应该是名称列表）
        self.num_samples = len(self.fnames)        #计算图片列表长度（也就是图片数量）

    def __getitem__(self,idx):                                   #通过下标一个一个去访问
        fname = self.fnames[idx]                                 #获取每一个图片
        img = torchvision.io.read_image(fname)
        '''
        将JPEG或PNG图像读入三维RGB张量。可选地将图像转换为所需的格式。输出张量的值是uint8，在0到255之间
        path : 获取图片地址
        ImageReadMode: 读取模式
        '''
        img = self.transform(img)
        return img

    def __len__(self):
        return self.num_samples

def get_dataset(root):
    fnames = glob.glob(os.path.join(root, '*'))                    #获取指定文件下的所有图片的路径
    '''
    将路径以list的形式返回
    glob模块被用来查找符合特定规则的文件路径名。跟使用windows下的文件搜索差不多。
    glob.glob函数的参数是字符串，查找文件只用到三个匹配符："*"， "?"， "[ ]"。
    其中，"*"表示匹配任意字符串，"?" 匹配任意单个字符， “[ ]” 匹配指定范围内的字符
    如：[0-9] 与 [a-z] 表示匹配 0-9 的单个数字与 a-z 的单个字符
    '''
    # 1. Resize the image to (64, 64)
    # 2. Linearly map [0, 1] to [-1, 1]
    compose = [
        transforms.ToPILImage(),
        transforms.Resize((64, 64)),
        transforms.ToTensor(),  #这里会有一个归一化
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),  #这里就是减去均值再除以方差(标准化）
    ]
    transform = transforms.Compose(compose)
    dataset = CrypkoDataset(fnames, transform)
    return dataset
# images = [dataset[i] for i in range(16)]
# grid_img = torchvision.utils.make_grid(images, nrow=4)
# plt.figure(figsize=(10,10))
# plt.imshow(grid_img.permute(1, 2, 0))
# plt.show()