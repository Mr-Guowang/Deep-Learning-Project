import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
class Image_and_Label(Dataset):
    def __init__(self,Data_path,transform=None):
        super(Image_and_Label, self).__init__()
        self.Data_path = Data_path
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
        self.Path_list = os.listdir(Data_path)              #得到文件夹中所有图片名称组成的列表

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
        label = torch.as_tensor(label, dtype=torch.int64)
        return img, label
def Get_data(Path,transform=None):     #获得训练集或测试集
    Data = Image_and_Label(Path)
    return Data
