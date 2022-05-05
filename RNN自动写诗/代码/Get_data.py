import numpy as np
def get_data():
    datas = np.load('tang.npz',allow_pickle=True)
    data,word2ix,ix2word = datas['data'],datas['word2ix'].item(),datas['ix2word'].item()
    return data,word2ix,ix2word

