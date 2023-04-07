import math #数学运算库
import torch.nn as nn #torch神经网络库
import collections as c

#残差网络进行特征提取

class DarkNet(nn.Module):
    def __init__(self,layers): #初始化变量
        super(DarkNet,self).__init__()
        self.chan = 32 #定义32通道数
        self.conv1 = nn.Conv2d(3,self.chan,(3,3),stride=1,padding=1,bias=False)
        #定义3x3卷积核 此时变化为 416,416,3->416,416,32
        self.Bn1 = nn.BatchNorm2d(self.chan)#标准化操作
        self.Relu1 = nn.LeakyReLU(0.1)#Relu激活函数（激活函数的系数怎么选择？）

        #初始化残差块
        #416,416,32->208,208,64
        self.layer1 = self.__make_layers([32,64],layers[0])
        #208,208,64->104,104,128
        self.layer2 = self.__make_layers([64,128],layers[1])
        #104,104,128->52,52,256
        self.layer3 = self.__make_layers([128, 256], layers[2])
        #52,52,256->26,26,512
        self.layer4 = self.__make_layers([256, 512], layers[3])
        #26,26,512->13,13,1024
        self.layer5 = self.__make_layers([512, 1024], layers[4])

        self.layer_filters = [64,128,256,512,1024]

        # 权值初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    #在每一个layer内，先进行下采样，再进行残差块的堆叠
    def __make_layers(self,planes,blocks):
        layers = []
        #下采样
        #大小为3x3，步长为2
        layers.append("dn_Conv",nn.Conv2d(self.inplanes,planes[1],kernel_size=3,stride=2,padding=1,bias=False))
        layers.append("dn_BN",nn.BatchNorm2d(planes[1]))
        layers.append("dn_Relu",nn.LeakyReLU(0.1))
        #加入残差结构
        self.inplanes = planes[1]
        for i in range(0,blocks):
            layers.append("residual_{}".format(i),RsBlock(self.inplanes,planes))
        return nn.Sequential(c.OrderedDict(layers))

    def forward(self,x):
        x = self.conv1(x)
        x = self.Bn1(x)
        x = self.Relu1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        out3 = self.layer3(x)
        out4 = self.layer4(out3)
        out5 = self.layer5(out4)

        return out3,out4,out5


#残差块
class RsBlock(nn.Module):
    def __init__(self,inplanes,planes):
        super(RsBlock,self).__init__()
        #构建残差块
        #先用1x1卷积核后用3x3卷积核的原因是减少参数量
        self.conv1 = nn.Conv2d(inplanes,planes[0],kernel_size=1,stride=1,padding=0,bias=False)
        self.BN1 = nn.BatchNorm2d(planes[0])
        self.Relu1 = nn.LeakyReLU(0.1)

        self.conv2 = nn.Conv2d(planes[0],planes[1],kernel_size=3,stride=1,padding=1,bias=False)
        self.BN2 = nn.BatchNorm2d(planes[1])
        self.Relu2 = nn.LeakyReLU(0.1)

    #构建残差结构，分为两部分。
    #一部分接收正常的卷积、标准化、激活函数。
    #另一部分不经过任何处理保留下来，并在最后两部分相加。
    def forward(self,x):
        res = x

        out = self.conv1(x)
        out = self.BN1(out)
        out = self.Relu1(out)

        out = self.conv2(out)
        out = self.BN2(out)
        out += res

        out = self.Relu2(out)
        return out

def darknet53():
    model = DarkNet([1,2,8,8,4])
    return model