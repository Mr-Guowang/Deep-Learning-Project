import Model_transformer
import Get_data
import torch
from torch import nn
import Train
import Predict
device = torch.device('cuda')
src_vocab = Get_data.data.en_total_words
tgt_vocab = Get_data.data.cn_total_words
epochs = 30
train_data = Get_data.data.train_data
model = Model_transformer.make_model(src_vocab,tgt_vocab)
model.to(device)
loss = nn.CrossEntropyLoss()
optim = torch.optim.Adam(model.parameters(), lr=0.001)
# Train.train(epochs,model,train_data,loss,optim,device)
# torch.save(model,'./model.pt')
model = torch.load('./model.pt')
model.to(device)
Predict.machine_translate('Do you know how to swim.', model)

