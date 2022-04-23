import torch.nn as nn
import torch.nn.functional as F
#DCGAN指出，所有的权重都以均值为0，标准差为0.2的正态分布随机初始化。
# weights_init 函数读取一个已初始化的模型并重新初始化卷积层，转置卷积层，batch normalization 层。
# 这个函数在模型初始化之后使用。
def weights_init(m):
    classname = m.__class__.__name__      #用来查看每一个submodule的类名
    if classname.find('Conv') != -1:      # 如果类名当中有Conv
        m.weight.data.normal_(0.0, 0.02)     #对权重进行初始化
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class Generator(nn.Module):
    """
    input (N, in_dim)
    output (N, 3, 64, 64)
    """
    def __init__(self, in_dim, dim=64):
        super(Generator, self).__init__()
        def dconv_bn_relu(in_dim, out_dim):  #后文中dim=64
            return nn.Sequential(
                nn.ConvTranspose2d(in_dim, out_dim, 5, 2,
                                   padding=2, output_padding=1, bias=False),
                #csdn上搜一搜反卷积定义反卷积.(c+2*2-5)/2+1 = （c+1）/2， output_padding=1的取值决定了c的值
                nn.BatchNorm2d(out_dim),               #批归一化
                nn.ReLU())
        self.l1 = nn.Sequential(
            nn.Linear(in_dim, dim * 8 * 4 * 4, bias=False),
            nn.BatchNorm1d(dim * 8 * 4 * 4),
            nn.ReLU())
        self.l2_5 = nn.Sequential(
            dconv_bn_relu(dim * 8, dim * 4),        #这里从512*4*4到256*8*8
            dconv_bn_relu(dim * 4, dim * 2),         #这里从256*8*8到128*16*16
            dconv_bn_relu(dim * 2, dim),             #这里从128*16*16到64*32*32
            nn.ConvTranspose2d(dim, 3, 5, 2, padding=2, output_padding=1), #这里转化为3*64*64
            nn.Tanh())                         #激活函数
        self.apply(weights_init)
    def forward(self, x):
        y = self.l1(x)         #这里输出为64*8*4*4
        y = y.view(y.size(0), -1, 4, 4)   #这里将y resize为512*4*4
        y = self.l2_5(y)              #这里经过多次反卷积，具体图像变换见上面
        return y

class Discriminator(nn.Module):                  #这里是判别器
    """
    input (N, 3, 64, 64)
    output (N, )
    """
    def __init__(self, in_dim, dim=64):
        super(Discriminator, self).__init__()
        def conv_bn_lrelu(in_dim, out_dim):
            return nn.Sequential(
                nn.Conv2d(in_dim, out_dim, 5, 2, 2),
                nn.BatchNorm2d(out_dim),
                nn.LeakyReLU(0.2))
        self.ls = nn.Sequential(
            nn.Conv2d(in_dim, dim, 5, 2, 2),         #这里输出为64*32*32
            nn.LeakyReLU(0.2),                   #带泄露的relu
            conv_bn_lrelu(dim, dim * 2),                  #这里输出为128*16*16
            conv_bn_lrelu(dim * 2, dim * 4),           #这里输出为256*8*8
            conv_bn_lrelu(dim * 4, dim * 8),         #这里输出为512*4*4
            nn.Conv2d(dim * 8, 1, 4),               #这里输出为 1*1的值
            nn.Sigmoid())
        self.apply(weights_init)
    def forward(self, x):              #输入为3*64*64
        y = self.ls(x)
        y = y.view(-1)
        return y
