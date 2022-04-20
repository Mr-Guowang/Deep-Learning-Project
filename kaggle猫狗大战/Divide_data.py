#本文件用于划分下载好的数据集，将一部分猫狗图片放入训练集，另一部分放入测试集
#这里的代码是从训练数据集中复制一部分到自建文件夹中作为训练数据
import os
from PIL import Image
import shutil
Train_path = 'C:/Users/Mr.Guo/Desktop/实验数据/kaggle/train'                 #训练数据的位置
Train_save = 'C:/Users/Mr.Guo/Desktop/猫狗识别/image-and-labe/Train_data'       #自建存储训练图片的位置
Test_save = 'C:/Users/Mr.Guo/Desktop/猫狗识别/image-and-labe/Test_data'       #自建存储测试图片的位置
def set_image(Train_path,Train_save,Test_save,Train_num,Test_num):         #(原始数据地址，训练存储地址，测试存储地址，训练数量，测试数量)
    data_file = os.listdir(Train_path)  # 读取训练文档中每个图片的名称，作为列表返回
    dog_file = list(filter(lambda x: x[:3] == 'dog', data_file))  # 筛选出上述列表中含有‘dog’的文件名
    cat_file = list(filter(lambda x: x[:3] == 'cat', data_file))  # 筛选出上述列表中含有‘cat’的文件名
    print('狗:', str(len(dog_file)), '\n猫:', str(len(cat_file)))  # 查看猫狗图片数量
    all_num = int(Train_num+Test_num)
    all_num = int(all_num/2)
    train_num = int(Train_num/2)
    for i, file in enumerate(dog_file):
        if (i < all_num):  # 读取图片
            if(i<train_num):
                shutil.copy(os.path.join(Train_path, file), Train_save)  # 将文件复制到新的文件夹中
            else:
                shutil.copy(os.path.join(Train_path, file), Test_save)  # 将文件复制到新的文件夹中
    for i, file in enumerate(cat_file):
        if (i < all_num):  # 读取图片
            if (i < train_num):
                shutil.copy(os.path.join(Train_path, file), Train_save)  # 将文件复制到新的文件夹中
            else:
                shutil.copy(os.path.join(Train_path, file), Test_save)  # 将文件复制到新的文件夹中
    #下面查看图片数量
    train_save_file = os.listdir(Train_save)
    test_save_file = os.listdir(Test_save)
    train_dog = list(filter(lambda x: x[:3] == 'dog', train_save_file))
    train_cat = list(filter(lambda x: x[:3] == 'cat', train_save_file))
    test_dog = list(filter(lambda x: x[:3] == 'dog', test_save_file))
    test_cat = list(filter(lambda x: x[:3] == 'cat', test_save_file))
    print('训练集数量:', str(len(train_save_file)),'狗数量：',str(len(train_dog)),'猫数量', str(len(train_cat)),
          '\n测试集图片数量:', str(len(test_save_file)),'狗数量：',str(len(test_dog)),'猫数量', str(len(test_cat)))

set_image(Train_path,Train_save,Test_save,3200,800)                 #执行

#执行结果
'''
狗: 12500 
猫: 12500
训练集数量: 3200 狗数量： 1600 猫数量 1600 
测试集图片数量: 800 狗数量： 400 猫数量 400
'''
