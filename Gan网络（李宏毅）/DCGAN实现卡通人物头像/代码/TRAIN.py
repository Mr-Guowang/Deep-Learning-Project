import random
import torch
import numpy as np
import torch.nn as nn
import torchvision
from torch import optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
def train(G,D,opt_G,opt_D,n_epoch,criterion,dataloader,z_dim,n_critic,z_sample):
    steps = 0
    for e, epoch in enumerate(range(n_epoch)):
        progress_bar = tqdm(dataloader)
        for i,data in enumerate(progress_bar):
            imgs = data  # 此处将图片取出，一次取出一个bactch的图片 size*3*64*64
            imgs = imgs.cuda()
            bs = imgs.size(0)  # 获得batchSize的大小

            # ============================================
            #  Train D
            # ============================================
            z = Variable(torch.randn(bs, z_dim)).cuda()  # 生成bs个噪声
            r_imgs = Variable(imgs).cuda()
            '''在python之前的版本中，variable可以封装tensor，计算反向传播梯度时需要将tensor封装在variable中。
            但是在python 0.4版本之后，将variable和tensor合并，也就是说不需要将tensor封装在variable中就可以计算梯度。
            tensor具有variable的性质。作为能否autograd的标签，requires_grad现在是Tensor的属性.
            所以，只要当一个操作的任何输入Tensor具有requires_grad = True的属性，autograd就可以自动追踪历史和反向传播了。
            '''
            f_imgs = G(z)  # 假图片

            """ Medium: Use WGAN Loss. """
            # Label
            r_label = torch.ones((bs)).cuda()  # 给真图片附上1的标签
            f_label = torch.zeros((bs)).cuda()  # 给假图片附上0的标签

            # Model forwarding
            r_logit = D(r_imgs.detach())  # 计算真图片的预测值
            f_logit = D(f_imgs.detach())  # 计算假图片的预测值

            # Compute the loss for the discriminator.
            r_loss = criterion(r_logit, r_label)  # 计算真图片的loss
            f_loss = criterion(f_logit, f_label)  # 计算假图片的loss
            loss_D = (r_loss + f_loss) / 2  # 求平均

            # WGAN Loss
            # loss_D = -torch.mean(D(r_imgs)) + torch.mean(D(f_imgs))

            # Model backwarding
            D.zero_grad()  # 清空D的梯度
            loss_D.backward()  # 反向传播

            # Update the discriminator.
            opt_D.step()  # 更新参数

            """ Medium: Clip weights of discriminator. """
            # for p in D.parameters():
            #    p.data.clamp_(-clip_value, clip_value)

            # ============================================
            #  Train G
            # ============================================
            if steps % n_critic == 0:  # 如果参数更新次数满了，那就开始训练我们的生成网络
                # Generate some fake images.
                z = Variable(torch.randn(bs, z_dim)).cuda()  # 首先生成一些假图片
                f_imgs = G(z)

                # Model forwarding
                f_logit = D(f_imgs)  # 然后预测出它们的标签

                """ Medium: Use WGAN Loss"""
                # Compute the loss for the generator.
                loss_G = criterion(f_logit, r_label)  # 然后计算loss，注意，这里是把它和真标签进行计算的
                # WGAN Loss
                # loss_G = -torch.mean(D(f_imgs))

                # Model backwarding
                G.zero_grad()
                loss_G.backward()  # 然后更新生成器的参数

                # Update the generator.
                opt_G.step()

            steps += 1

            progress_bar.set_description("epoch %i"%(e+1))
            progress_bar.set_postfix(Loss_D= round(loss_D.item(), 4),Loss_G=round(loss_G.item(), 4),Step= steps)


        G.eval()  # 切换为评估模式
        f_imgs_sample = (G(z_sample).data + 1) / 2.0
        filename = os.path.join('C:/Users/Mr.Guo/Desktop/用gan网络生成卡通头像/生成图片文件夹', f'Epoch_{epoch + 1:03d}.jpg')
        torchvision.utils.save_image(f_imgs_sample, filename, nrow=10)
        print(f' | Save some samples to {filename}.')
        if(epoch>8):
            # Show generated images in pycharm.
            grid_img = torchvision.utils.make_grid(f_imgs_sample.cpu(), nrow=10)
            plt.figure(figsize=(10, 10))
            plt.imshow(grid_img.permute(1, 2, 0))
            plt.show()
        G.train()


        # if (e + 1) % 5 == 0 or e == 0:
        #     # Save the checkpoints.
        #     torch.save(G.state_dict(), './G.pth')
        #     torch.save(D.state_dict(), './D.pth')


