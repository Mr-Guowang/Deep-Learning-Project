import torch
from torch import nn
import torch.optim as optim

import GET_DATA
import GET_NET
import GET_NET
import Train
#参考网址https://colab.research.google.com/github/ga642381/ML2021-Spring/blob/main/HW11/HW11_ZH.ipynb#scrollTo=hrxKelBy0PJ7
device = "cuda:0" if torch.cuda.is_available() else "cpu"
feature_extractor = GET_NET.FeatureExtractor().to(device)    #特征提取网络
label_predictor = GET_NET.LabelPredictor().to(device)        #分类器
domain_classifier = GET_NET.DomainClassifier().to(device)    #domain分类器

class_criterion = nn.CrossEntropyLoss()                  #分类损失函数，因为有多类，所以最好用这个
domain_criterion = nn.BCEWithLogitsLoss()                #domain二分类，用bce即可

optimizer_F = optim.Adam(feature_extractor.parameters())
optimizer_C = optim.Adam(label_predictor.parameters())
optimizer_D = optim.Adam(domain_classifier.parameters())

source_dataloader = GET_DATA.source_dataloader
target_dataloader = GET_DATA.target_dataloader
test_dataloader = GET_DATA.test_dataloader

# train 200 epochs
for epoch in range(200):
    # You should chooose lamnda cleverly.
    train_D_loss, train_F_loss, train_acc = Train.train(source_dataloader, target_dataloader,
          feature_extractor,label_predictor,domain_classifier,class_criterion,domain_criterion,
          optimizer_D,optimizer_F,optimizer_C,lamb=0.1,device=device)

    torch.save(feature_extractor.state_dict(), f'extractor_model.bin')
    torch.save(label_predictor.state_dict(), f'predictor_model.bin')

    print('epoch {:>3d}: train D loss: {:6.4f}, train F loss: {:6.4f}, acc {:6.4f}'.format(epoch, train_D_loss, train_F_loss, train_acc))