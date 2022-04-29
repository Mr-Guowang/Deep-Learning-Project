import torch
from torch import nn
def train(source_dataloader, target_dataloader,feature_extractor,label_predictor,domain_classifier,class_criterion,domain_criterion,
          optimizer_D,optimizer_F,optimizer_C,lamb,device):
    # D loss: Domain Classifier的loss
    # F loss: Feature Extrator & Label Predictor的loss
    running_D_loss, running_F_loss = 0.0, 0.0
    total_hit, total_num = 0.0, 0.0
    for i, ((source_data, source_label), (target_data, _)) in enumerate(zip(source_dataloader, target_dataloader)):
        source_data = source_data.to(device)
        source_label = source_label.to(device)
        target_data = target_data.to(device)
        # 混合source data和target data, or it'll mislead the running params
        #   of batch_norm. (runnning mean/var of soucre and target data are different.)
        mixed_data = torch.cat([source_data, target_data], dim=0)   #这里是为了判断是否为sourse数据，下面两行代码是为了给这个数据配标签
        domain_label = torch.zeros([source_data.shape[0] + target_data.shape[0], 1]).to(device)
        # set domain label of source data to be 1.
        domain_label[:source_data.shape[0]] = 1 #这里的意思是前一半标为1，后一半为0

        # Step 1 : train domain classifier
        feature = feature_extractor(mixed_data)               #此处单独训练domain_classifier，所以要截断feature_extractor的通道
        # We don't need to train feature extractor in step 1.
        # Thus we detach the feature neuron to avoid backpropgation.
        domain_logits = domain_classifier(feature.detach())
        loss = domain_criterion(domain_logits, domain_label)
        running_D_loss += loss.item()
        loss.backward()
        optimizer_D.step()

        # Step 2 : train feature extractor and label classifier 训练feature extractor和标签分类器
        class_logits = label_predictor(feature[:source_data.shape[0]]) #只对source_data进行标签分类
        domain_logits = domain_classifier(feature)
        # loss = cross entropy of classification - lamb * domain binary cross entropy.
        #  The reason why using subtraction is similar to generator loss in disciminator of GAN
        loss = class_criterion(class_logits, source_label) - lamb * domain_criterion(domain_logits, domain_label)
        running_F_loss += loss.item()
        loss.backward()
        optimizer_F.step()
        optimizer_C.step()

        optimizer_D.zero_grad()
        optimizer_F.zero_grad()
        optimizer_C.zero_grad()

        total_hit += torch.sum(torch.argmax(class_logits, dim=1) == source_label).item() #
        total_num += source_data.shape[0]
        print(i, end='\r')


    return running_D_loss / (i + 1), running_F_loss / (i + 1), total_hit / total_num

