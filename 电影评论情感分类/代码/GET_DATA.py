import pandas as pd
import os
import torch
import numpy as np
import torch.utils.data as Data
from gensim.models import keyedvectors
Embedding_size = 50
Batch_Size = 32
Kernel = 3
Filter_num = 10#卷积核的数量。
Epoch = 60
Dropout = 0.5
Learning_rate = 1e-3


path_train = './Dataset/train.txt'
path_test = './Dataset/test.txt'
path_validation = './Dataset/validation.txt'
train_data = pd.read_csv(path_train,names=["label","comment"],sep="\t") #读取数据
test_data = pd.read_csv(path_test,names=["label","comment"],sep="\t") #读取数据
validation_data = pd.read_csv(path_validation,names=["label","comment"],sep="\t") #读取数据
# comments_len=train_data.iloc[:,1].apply(lambda x:len(x.split())) #这里保存的是每一行词的数量
# train_data["comments_len"]=comments_len
# train_data["comments_len"].describe(percentiles=[.5,.95])
# comments_len=test_data.iloc[:,1].apply(lambda x:len(x.split())) #这里保存的是每一行词的数量
# test_data["comments_len"]=comments_len
# test_data["comments_len"].describe(percentiles=[.5,.95])
comments_len=validation_data.iloc[:,1].apply(lambda x:len(x.split())) #这里保存的是每一行词的数量
validation_data["comments_len"]=comments_len
validation_data["comments_len"].describe(percentiles=[.5,.95])
#数据预处理
# from collections import Counter
# words=[]
# for i in range(len(train_data)):
#     com=train_data["comment"][i].split()
#     words=words+com
# print(len(words))                     #一共893491个词
#
# Freq=1                                #将频率小于5的词汇去掉
# import os
# with open(os.path.join('./Dataset',"word_freq.txt"), 'w', encoding='utf-8') as fout:
#     for word,freq in Counter(words).most_common():
#         if freq>Freq:
#             fout.write(word+"\n")
#这个时候，我们的文件夹下多了一个文件:“word_freq.txt”。
#初始化vocab
with open(os.path.join('./Dataset',"word_freq.txt"), encoding='utf-8') as fin:
    vocab = [i.strip() for i in fin]
vocab=list(set(vocab))
word2idx = {i:index for index, i in enumerate(vocab)}
idx2word = {index:i for index, i in enumerate(vocab)}
vocab_size = len(vocab)

# print(len(vocab))                    #可以看到，频率小于1的已经被删去
#我们把那些出现频率低于1的词都变成"把“字
pad_id=word2idx["把"]
sequence_length = 60
#对输入数据进行预处理,主要是对句子用索引表示且对句子进行截断与padding，将填充使用”把“来。
def tokenizer_train():
    inputs = []
    sentence_char = [i.split() for i in train_data["comment"]]
    # 将输入文本进行padding
    for index,i in enumerate(sentence_char):
        temp=[word2idx.get(j,pad_id) for j in i]#表示如果词表中没有这个稀有词，无法获得，那么就默认返回pad_id。
        if(len(i)<sequence_length):
            #应该padding。
            for _ in range(sequence_length-len(i)):
                temp.append(pad_id)
        else:
            temp = temp[:sequence_length]
        inputs.append(temp)
    return inputs
def tokenizer_test():
    inputs = []
    sentence_char = [i.split() for i in test_data["comment"]]
    # 将输入文本进行padding
    for index,i in enumerate(sentence_char):
        temp=[word2idx.get(j,pad_id) for j in i]#表示如果词表中没有这个稀有词，无法获得，那么就默认返回pad_id。
        if(len(i)<sequence_length):
            #应该padding。
            for _ in range(sequence_length-len(i)):
                temp.append(pad_id)
        else:
            temp = temp[:sequence_length]
        inputs.append(temp)
    return inputs
def tokenizer_vala():
    inputs = []
    sentence_char = [i.split() for i in validation_data["comment"]]
    # 将输入文本进行padding
    for index,i in enumerate(sentence_char):
        temp=[word2idx.get(j,pad_id) for j in i]#表示如果词表中没有这个稀有词，无法获得，那么就默认返回pad_id。
        if(len(i)<sequence_length):
            #应该padding。
            for _ in range(sequence_length-len(i)):
                temp.append(pad_id)
        else:
            temp = temp[:sequence_length]
        inputs.append(temp)
    return inputs
train_data_input = tokenizer_train()
test_data_input = tokenizer_test()
validation_data_input = tokenizer_vala()

class TextCNNDataSet(Data.Dataset):
    def __init__(self, data_inputs, data_targets):
        self.inputs = torch.LongTensor(data_inputs)
        self.label = torch.LongTensor(data_targets)

    def __getitem__(self, index):
        return self.inputs[index], self.label[index]

    def __len__(self):
        return len(self.inputs)




Train_TextCNNDataSet = TextCNNDataSet(train_data_input, list(train_data["label"]))
test_TextCNNDataSet = TextCNNDataSet(test_data_input, list(test_data["label"]))
vala_TextCNNDataSet = TextCNNDataSet(validation_data_input, list(validation_data["label"]))
TrainDataLoader = Data.DataLoader(Train_TextCNNDataSet, batch_size=Batch_Size, shuffle=True)
TestDataLoader = Data.DataLoader(test_TextCNNDataSet, batch_size=Batch_Size, shuffle=True)
ValaDataLoader = Data.DataLoader(vala_TextCNNDataSet, batch_size=Batch_Size, shuffle=True)
# w2v = keyedvectors.load_word2vec_format('./Dataset/wiki_word2vec_50.bin', binary=True)
from tqdm import tqdm
# step = tqdm(range(len(vocab)))
# for i in step:
#         try:
#             w2v[vocab[i]] = w2v[vocab[i]]
#         except Exception:
#             w2v[vocab[i]] = np.random.randn(50, )
# torch.save(w2v,'./Dataset/w2v.pt')
w2v = torch.load('./Dataset/w2v.pt')
#这时候我们需要获得embedding的权重
# w2v_embedding = []
# i = 0
# steps = tqdm(vocab)
# for x in steps:
#     y = w2v[x]
#     w2v_embedding.append(y)
# w2v_embedding = torch.tensor(w2v_embedding)
# torch.save(w2v_embedding,'./Dataset/w2v_embedding')
w2v_embedding = torch.load('./Dataset/w2v_embedding')
print(w2v_embedding.shape)


def get_data():
    return TrainDataLoader,TestDataLoader,ValaDataLoader,vocab,w2v,w2v_embedding,word2idx,idx2word,vocab_size