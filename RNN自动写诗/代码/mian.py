import sys, os
import torch
import torch as t
from Get_data import get_data
from model import PoetryModel
from torch import nn
import tqdm
from torchnet import meter
device = "cuda:0"
data, word2ix, ix2word = get_data()
data = t.from_numpy(data)
dataloader = t.utils.data.DataLoader(data,batch_size=128,shuffle=True,num_workers=0)
model = PoetryModel(len(word2ix), 128, 256).to(device)
optimizer = t.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()
def generate(model, start_words, ix2word, word2ix, prefix_words=None):
    """
    给定几个词，根据这几个词接着生成一首完整的诗歌
    start_words：u'春江潮水连海平'
    比如start_words 为 春江潮水连海平，可以生成：
    """
    results = list(start_words)
    start_word_len = len(start_words)
    # 手动设置第一个词为<START>
    input = t.Tensor([word2ix['<START>']]).view(1, 1).long()
    hidden = None

    for i in range(200):
        output, hidden = model(input, hidden)

        if i < start_word_len:
            w = results[i]
            input = input.data.new([word2ix[w]]).view(1, 1)
        else:
            top_index = output.data[0].topk(1)[1][0].item()
            w = ix2word[top_index]
            results.append(w)
            input = input.data.new([top_index]).view(1, 1)
        if w == '<EOP>':
            del results[-1]
            break
    return results
def train():
    for epoch in range(50):
        print('epoch',epoch+1)
        for ii, data_ in tqdm.tqdm(enumerate(dataloader)):
            data_ = data_.long().transpose(1, 0).contiguous()
            data_ = data_.to(device)
            optimizer.zero_grad()
            input_, target = data_[:-1, :], data_[1:, :]
            output, _ = model(input_)
            loss = criterion(output, target.view(-1))
            loss.backward()
            optimizer.step()
        model.to('cpu')
        out = generate(model,'湖光秋月两相和', ix2word, word2ix)
        for x in out:
            print(x,end = "")
        model.to(device)
        print(' ')
        if(epoch>38):
            t.save(model.state_dict(), './epoch%s.pth' % ( epoch+1))


def gen_acrostic(model, start_words, ix2word, word2ix, prefix_words=None):
    """
    生成藏头诗
    start_words : u'深度学习'
    生成：
    深木通中岳，青苔半日脂。
    度山分地险，逆浪到南巴。
    学道兵犹毒，当时燕不移。
    习根通古岸，开镜出清羸。
    """
    results = []
    start_word_len = len(start_words)
    input = (t.Tensor([word2ix['<START>']]).view(1, 1).long())
    hidden = None
    index = 0  # 用来指示已经生成了多少句藏头诗
    # 上一个词
    pre_word = '<START>'
    if prefix_words:
        for word in prefix_words:
            output, hidden = model(input, hidden)
            input = (input.data.new([word2ix[word]])).view(1, 1)

    for i in range(200):
        output, hidden = model(input, hidden)
        top_index = output.data[0].topk(1)[1][0].item()
        w = ix2word[top_index]

        if (pre_word in {u'。', u'！', '<START>'}):
            # 如果遇到句号，藏头的词送进去生成

            if index == start_word_len-1:
                # 如果生成的诗歌已经包含全部藏头的词，则结束
                break
            else:
                # 把藏头的词作为输入送入模型
                w = start_words[index]
                index += 1
                input = (input.data.new([word2ix[w]])).view(1, 1)
        else:
            # 否则的话，把上一次预测是词作为下一个词输入
            input = (input.data.new([word2ix[w]])).view(1, 1)
        results.append(w)
        pre_word = w
    return results
model.load_state_dict(torch.load('./epoch50.pth'))
model.to('cpu')
# print("生成普通诗句")
# out = generate(model,'十年生死两茫茫', ix2word, word2ix)
# for x in out:
#     print(x,end = "")
# print(' ')

print("生成藏头")
out = gen_acrostic(model,'天下为先，', ix2word, word2ix)
for x in out:
    print(x,end = "")
print(' ')


