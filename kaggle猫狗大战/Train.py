import time
import torch
import Test
from torch import nn
import sys
import d2lzh_pytorch as d21
import matplotlib.pyplot as plt
def train(net,train_iter,test_iter,batch_size,optimizer,device,num_epochs):
    net = net.to(device)
    print("train on",device)
    loss = torch.nn.CrossEntropyLoss()
    batch_count = 0
    train_ac,test_ac,losss = [],[],[]
    for epoch in range(num_epochs):
        train_l_sum,train_acc_sum,n,start = 0.0,0.0,0,time.time()
        for image,label in train_iter:
            image = image.to(device)
            label = label.to(device)
            label_hat = net(image)
            l = loss(label_hat,label)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            train_l_sum +=l.cpu().item()
            train_acc_sum +=(label_hat.argmax(dim=1)==label).sum().cpu().item()
            n+=label.shape[0]
            batch_count+=1
        Test_acc = Test.test(test_iter, net, device)
        train_ac.append(train_acc_sum/n)
        test_ac.append(Test_acc)
        losss.append(train_l_sum)
        print('epoch:%d,loss=%.4f,train acc=%.3f,test acc=%.4f time=%.1f sec'%(epoch+1,train_l_sum/batch_count,train_acc_sum/n,Test_acc,time.time()-start))
    d21.semilogy(range(1,num_epochs+1),train_ac,'epochs','acc',range(1,num_epochs+1),test_ac,['train','test'])
    plt.show()
    d21.semilogy(range(1, num_epochs + 1),losss, 'epochs', 'loss')
    plt.show()