def get_data(batch_size):
    import torch
    import torchvision
    import sys
    transform = torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor(),torchvision.transforms.Normalize(mean=[0.5], std=[0.5])]
    )
    #设置路径地址
    path = 'C:/Users/Mr.Guo/Desktop/实验数据/手写数字识别/MNIST'
    trainData = torchvision.datasets.MNIST(path, train=True, transform=transform)
    testData = torchvision.datasets.MNIST(path, train=False, transform=transform)
    #使用多线程读取数据
    if sys.platform.startswith('win'):
        numworkers = 0
    else:
        numworkers = 4
    train_iter = torch.utils.data.DataLoader(trainData, batch_size=batch_size, shuffle=True,num_workers = numworkers)
    test_iter = torch.utils.data.DataLoader(testData, batch_size=batch_size, shuffle=True, num_workers=numworkers)
    # 返回两个迭代器
    return train_iter,test_iter