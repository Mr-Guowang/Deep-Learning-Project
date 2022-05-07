import torch
import torch.nn as nn
import torch.utils.data as Data
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os
import GET_DATA
from gensim.models import keyedvectors
from tqdm import tqdm

device = torch.device('cuda:0')
Embedding_size = 50
Batch_Size = 32
Kernel = 3
Filter_num = 10#卷积核的数量。
Epoch = 100
dropout = 0.6
Learning_rate = 0.001

TrainDataLoader,TestDataLoader,ValaDataLoader,vocab,w2v,w2v_embedding,word2idx,idx2word,vocab_size = GET_DATA.get_data()
class TextCNN(nn.Module):
    def __init__(self):
        super(TextCNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size,50)
        self.constant_embedding = nn.Embedding(vocab_size, 10)
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=60, out_channels=20, kernel_size=2),
            nn.ReLU(),
            nn.MaxPool2d((20, 1))
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(59, 2)
        self.softmax = nn.Softmax(dim=1) # 行

    def forward(self, X): # [batch_size, 63]
        batch_size = X.shape[0]
        X = torch.cat((self.embedding(X.to(device)),self.constant_embedding(X.to(device))),dim=2)
        #X = X.unsqueeze(1) # [batch_size, 1, 63, 50]
        X = self.conv(X) # [batch_size, 10, 1, 1]
        X = X.view(batch_size, -1) # [batch_size, 10]
        X = self.fc(X) # [batch_size, 2]
        X = self.softmax(X) # [batch_size, 2]
        return X

print("+++++++++++start train+++++++++++")
if __name__ == '__main__':

    text_cnn = TextCNN()
    text_cnn.embedding.weight.data.copy_(w2v_embedding)
    text_cnn.embedding.weight.requires_grad = True
    optimizer = torch.optim.Adam(text_cnn.parameters(), lr=Learning_rate)
    text_cnn.to(device)
    loss_fuc = nn.CrossEntropyLoss()
    text_cnn.train()

    epochs =range(Epoch)
    train_list = []
    test_list = []
    for epoch in epochs:
        train_acc_num,test_acc_num,vala_acc_num,train_num,test_num,vala_num = 0,0,0,0,0,0
        text_cnn.train()
        for data,label in TrainDataLoader:
            data.to(device)
            label = label.to(device)
            predicted = text_cnn(data)
            loss = loss_fuc(predicted,label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_acc_num+=(predicted.argmax(dim=1)==label).float().sum().cpu().item()
            train_num += label.shape[0]
        accuracy = train_acc_num / train_num
        train_list.append(accuracy)
        text_cnn.eval()
        for data, label in ValaDataLoader:
            data.to(device)
            label = label.to(device)
            predicted = text_cnn(data)
            vala_acc_num += (predicted.argmax(dim=1) == label).float().sum().cpu().item()
            vala_num += label.shape[0]
        valaaccuracy = vala_acc_num / vala_num
        text_cnn.eval()
        for data,label in TestDataLoader:
            data.to(device)
            label=label.to(device)
            predicted = text_cnn(data)
            test_acc_num += (predicted.argmax(dim=1) == label).float().sum().cpu().item()
            test_num += label.shape[0]
        testaccuracy = test_acc_num / test_num

        test_list.append(testaccuracy)
        print('epoch:', epoch+1, ' | train loss:%.4f' % loss.item(), ' | train accuracy:', accuracy,' | vala accuracy:', valaaccuracy,' | test accuracy:', testaccuracy)
    import matplotlib.pyplot as plt


    plt.plot(range(len(test_list)), test_list, color='blue', label='test_acc')
    plt.plot(range(len(test_list)), train_list, color='red', label='train_acc')
    plt.title("The accuracy of textCNN model")
    plt.legend()
    plt.show()






