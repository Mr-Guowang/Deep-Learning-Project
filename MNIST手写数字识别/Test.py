import torch


def test(test_iter,net,device):
    acc_sum,n = 0.0,0
    net = net.to(device)
    with torch.no_grad():
        for test_image,test_label in test_iter:
            acc_sum +=(net(test_image.to(device)).argmax(dim=1)==test_label.to(device)).float().sum().cpu().item()
            n+=test_label.shape[0]
    return acc_sum/n

