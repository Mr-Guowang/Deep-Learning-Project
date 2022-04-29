import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import numpy as np
import cv2
# def no_axis_show(img, title='', cmap=None):
#   fig = plt.imshow(img, interpolation='nearest', cmap=cmap)
#   #interpolation代表的是插值运算，'nearest'只是选取了其中的一种插值方式。cmap表示绘图时的样式
#   fig.axes.get_xaxis().set_visible(False)
#   fig.axes.get_yaxis().set_visible(False) #删除坐标轴
#   plt.title(title)
#
# titles = ['horse', 'bed', 'clock', 'apple', 'cat', 'plane', 'television', 'dog', 'dolphin', 'spider']
# plt.figure(figsize=(18, 18))
# for i in range(10):
#     plt.subplot(1, 10, i+1)
#     fig = no_axis_show(plt.imread(f'C:/Users/Mr.Guo/Desktop/实验数据/real_or_drawing/train_data/{i}/{500*i}.bmp'),
#                        title=titles[i])#读取像素值，然后传入显示no_axis_show
# plt.show()
# plt.figure(figsize=(18, 18))
# for i in range(10):
#     plt.subplot(1, 10, i+1)
#     fig = no_axis_show(plt.imread(f'C:/Users/Mr.Guo/Desktop/实验数据/real_or_drawing/test_data/0/'
#                                   + str(i).rjust(5, '0') + '.bmp')) #rjust方法返回一个右对齐排版的字符串。
# plt.show()
Path_train = 'C:/Users/Mr.Guo/Desktop/实验数据/real_or_drawing/train_data'
Path_test = 'C:/Users/Mr.Guo/Desktop/实验数据/real_or_drawing/test_data'
source_transform = transforms.Compose([
    transforms.Grayscale(),# Turn RGB to grayscale. (Bacause Canny do not support RGB images.)
    transforms.Lambda(lambda x: cv2.Canny(np.array(x), 170, 300)),#cv2不支持img，只支持numpy
    transforms.ToPILImage(),    # Transform np.array back to the skimage.Image.
    transforms.RandomHorizontalFlip(),#依据概率p对PIL图片进行水平翻转，p默认0.5
    transforms.RandomRotation(15, fill=(0,)),# 旋转正负15°，如果有空像素，那就补0
    transforms.ToTensor(),
])
target_transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((32, 32)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15, fill=(0,)),
    transforms.ToTensor(),
])
source_dataset = ImageFolder(Path_train, transform=source_transform)#这玩意返回的是一个{图片，标签}，其中标签就是子文件夹名称
target_dataset = ImageFolder(Path_test, transform=target_transform)
source_dataloader = DataLoader(source_dataset, batch_size=32, shuffle=True)
target_dataloader = DataLoader(target_dataset, batch_size=32, shuffle=True)
test_dataloader = DataLoader(target_dataset, batch_size=128, shuffle=False)

