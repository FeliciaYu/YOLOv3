import torch.nn as nn #torch神经网络库
import collections as c
import Darknet

#通过提取出来的特征获得预测结果

class Yolov3(nn.Module):
    #定义
    def __init__(self,anchors_mask,num_classes):
        super(Yolov3,self).__init__()

        #生成Darknet-53的三个主干模型
        #shape分别是：
        #52,52,256 中间层
        #26,26,512 中下层
        #13,13,1024 底层
        self.backbone = Darknet.darknet53()
        #out_filter = [64,128,256,512,1024]
        out_filter = self.backbone.layer_filters

        #第三个参数：3*（num_classes+1+4) 3为先验框个数，1判断先验框内部是否真实地包含着物体，4是调整的参数
        #底层
        self.last_layer0 =self.__make_last_layers([512,1024],out_filter[-1],len(anchors_mask[0]) * (num_classes + 5))
        #中下层
        self.last_layer1_conv = conv2d(512,256,1)
        self.last_layer1_upSam = nn.Upsample(scale_factor=2,mode='nearest')
        self.last_layer1 = self.__make_last_layers([256,512],out_filter[-2]+256,len(anchors_mask[1]) * (num_classes + 5))
        #中间层
        self.last_layer2_conv = conv2d(256,128,1)
        self.last_layer2_upSam = nn.Upsample(scale_factor=2,mode='nearest')
        self.last_layer2 = self.__make_last_layers([128,256],out_filter[-3]+128,len(anchors_mask[2]) * (num_classes + 5))

    def forward(self,x):
        #获取三个有效特征层，分别是中间层、中下层、底层
        x2,x1,x0 = self.backbone(x)
        #底层
        x0_branch = self.last_layer0(x0)[:5]
        #out0 = [conv2d(filter_list[0],filter_list[1],3),nn.Conv2d(filter_list[1],filter_out,1,1,0,bias=False)]
        #基于voc数据集的shape为(batch_size,52,52,75)
        out0 = self.last_layer0(x0_branch)[5:]
        #堆叠
        x1_in = self.last_layer1_conv(x0_branch)
        x1_in = self.last_layer1_upSam(x1_in)
        x1_in = nn.cat([x1_in,x1],1)
        #中下层
        x1_branch = self.last_layer1(x1_in)[:5]
        #out1(与out0同理)
        #基于voc数据集的shape为(batch_size,26,26,75)
        out1 = self.last_layer1(x1_branch)[5:]
        #堆叠
        x2_in = self.last_layer2_conv(x1_branch)
        x2_in = self.last_layer2_upSam(x2_in)
        x2_in = nn.cat([x2_in,x2],1)
        #中间层
        # out2(与out1同理)
        # 基于voc数据集的shape为(batch_size,13,13,75)
        out2 = self.last_layer2(x2_in)[5:]

        return out0,out1,out2







def conv2d(filter_in,filter_out,kernelsize):
    pad = (kernelsize - 1) // 2 if kernelsize else 0
    return nn.Sequential(c.OrderedDict([
        ("conv",nn.Conv2d(filter_in,filter_out,kernelsize,stride=1,padding=pad,bias=False)),
        ("BN",nn.BatchNorm2d(filter_out)),
        ("Relu1",nn.LeakyReLU(0.1))
    ]))

#共有七个卷积，前五个用于特征提取，后两个用于预测结果
def __make_last_layers(filter_list,filter_in,filter_out):
    s = nn.Sequential(
        conv2d(filter_in,filter_list[0],1),
        conv2d(filter_list[0],filter_list[1],3),
        conv2d(filter_list[1],filter_list[0],1),
        conv2d(filter_list[0],filter_list[1],3),
        conv2d(filter_list[1],filter_list[0],1),

        conv2d(filter_list[0],filter_list[1],3),
        nn.Conv2d(filter_list[1],filter_out,1,1,0,bias=False)
    )
    return s