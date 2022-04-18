#这里的代码是从训练数据集中复制一部分到自建文件夹中作为训练数据
import os
from PIL import Image
import shutil
Train_path = 'C:/Users/Mr.Guo/Desktop/实验数据/kaggle/train'                 #训练数据的位置
Cat_save = 'C:/Users/Mr.Guo/Desktop/猫狗识别/image-and-labe/cat-image'       #自建存储狗图片的位置
Dog_save = 'C:/Users/Mr.Guo/Desktop/猫狗识别/image-and-labe/dog-image'       #自建存储猫图片的位置
def set_image(Train_path,Cat_save,Dog_save,dog_num,cat_num):         #(训练数据地址，猫存储地址，狗存储地址，狗数量，猫数量)
    data_file = os.listdir(Train_path)  # 读取训练文档中每个图片的名称，作为列表返回
    dog_file = list(filter(lambda x: x[:3] == 'dog', data_file))  # 筛选出上述列表中含有‘dog’的文件名
    cat_file = list(filter(lambda x: x[:3] == 'cat', data_file))  # 筛选出上述列表中含有‘cat’的文件名
    print('狗:', str(len(dog_file)), '\n猫:', str(len(cat_file)))  # 查看猫狗图片数量
    for i, file in enumerate(dog_file):
        if (i < 2000):  # 读取两千张图片
            shutil.copy(os.path.join(Train_path, file), Dog_save)  # 将文件复制到新的文件夹中
    for i, file in enumerate(cat_file):
        if (i < 2000):  # 读取两千张图片
            shutil.copy(os.path.join(Train_path, file), Cat_save)  # 将文件复制到新的文件夹中
    #下面查看图片数量
    dog_save_file = os.listdir(Dog_save)
    cat_save_file = os.listdir(Cat_save)
    print('狗:', str(len(dog_save_file)), '\n猫:', str(len(cat_save_file)))
set_image(Train_path,Cat_save,Dog_save,2000,2000)                 #执行



