'''
自定义数据的时候必须继承torch.utils.data.Dataset,然后重写下面的函数：
__len__: 使得len(dataset)返回数据集的大小；
__getitem__：使得支持dataset[i]能够返回第i个数据样本这样的下标操作
'''
#导入一些包
#本代码可以返回一个 图片+标签的东西，而且在类内实现了数据预处理，返回的东西类似于字典
#其实我觉得就是字典，因为我用d = {} d.upload(d1) 可以做到，也就是说，新建一个空字典可以直接读入这个实例化的类
#应该是父类里重载了这个方法，使他拥有了字典的属性，这个以后慢慢学吧
import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
class Image_and_Label(Dataset):
    def __init__(self,Data_path,NUMs,test_or_train,transform=None):
        super(Image_and_Label, self).__init__()
        self.Data_path = Data_path
                     #需求的数据量
             #true 表示训练，false表示测试
        if transform is None:
            self.transform = transforms.Compose(
                [
                    transforms.Resize(size=(224, 224)),
                    transforms.ToTensor(),  # convert PIL.Image to tensor, which is GY
                    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # Normalization
                ]
            )
        else:
            self.transform = transform
        path_list = os.listdir(Data_path)              #得到文件夹中所有图片名称组成的列表
        if test_or_train == True:                        #这一步是为了划分测试集和训练集
            self.Path_list = path_list[:NUMs]
        else:
            self.Path_list = path_list[NUMs:]

    def __len__(self) -> int:
        return len(self.Path_list)

    def __getitem__(self, idx: int):  #img to tensor, label to tensor
        Img_Path = self.Path_list[idx]
        Abs_Img_Path = os.path.join(self.Data_path,Img_Path)
        img = Image.open(Abs_Img_Path)
        img = self.transform(img)
        if Img_Path.split('.')[0] == 'dog':    #狗的标签为1，猫的标签为0
            label = 1
        else:
            label = 0
        label = torch.as_tensor(label, dtype=torch.int)
        return img, label

Dog_save = 'C:/Users/Mr.Guo/Desktop/猫狗识别/image-and-labe/dog-image'
cat_save = 'C:/Users/Mr.Guo/Desktop/猫狗识别/image-and-labe/cat-image'
dog = Image_and_Label(Dog_save,1600,True)
cat = Image_and_Label(cat_save,1600,False)
print(len(dog))
print(len(cat))