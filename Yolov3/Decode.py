import torch
import torch.nn as nn
from torchvision.ops import nms
import numpy as np

class DecodeBox:
    def __init__(self,anchors,num_classes,input_shape,anchors_mask=[[6,7,8],[3,4,5],[0,1,2]]):
        super(DecodeBox,self).__init__()
        self.anchors = anchors
        self.num_classes = num_classes
        self.bbox_attrs = 5+num_classes
        self.input_shape = input_shape
        self.anchors_mask = anchors_mask

# 输入的shape有三个，分别是：
# 中间层out2：(batch_side,52,52,255)
# 中下层out1：(batch_side,26,26,255)
# 底层out0：(batch_side,13,13,255)
    def decode_box(self,inputs):
        outputs = []
        for i ,input in enumerate(inputs):
            # 输入的图像的batchside,h和w
            batch_size = input.size(0)
            input_height = input.size(2)
            #out0 = 13,out1 = 26,out2 = 52
            input_width = input.size(3)
            # out0 = 13,out1 = 26,out2 = 52

            stride_h = self.input_shape[0]/input_height
            stride_w = self.input_shape[1]/input_width

            # 获取相对特征层大小的scaled_anchors
            scaled_anchors = [(anchor_width/stride_w,anchor_height/stride_h) for anchor_width,anchor_height in self.anchors[self.anchors_mask[i]]]

            #prediction[batch_size,len(self.anchors[i]),input_width,self.bbox_attrs,input_height]
            prediction = input.view(batch_size,len(self.anchors[i]),
                                    self.bbox_attrs,input_height.input_width).permute(0,1,3,4,2).contiguous()

            #先验框中心位置的调整参数
            x = nn.sigmoid(prediction[...,0])
            y = nn.sigmoid(prediction[...,1])
            #先验框的宽高调整参数
            w = prediction[...,2]
            h = prediction[...,3]
            #获得置信度，是否有物体
            conf = nn.sigmoid(prediction[...,4])

            #种类置信度
            pred_cls = nn.sigmoid(prediction[..., 5:])

            FloatTensor = nn.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
            LongTensor = nn.cuda.LongTensor if x.is_cuda else torch.LongTensor

            #生成网格，先验框中心，网格左上角
            grid_x = nn.linspace(0, input_width - 1, input_width).repeat(input_height, 1).repeat(
                batch_size * len(self.anchors_mask[i]), 1, 1).view(x.shape).type(FloatTensor)
            grid_y = nn.linspace(0, input_height - 1, input_height).repeat(input_width, 1).t().repeat(
                batch_size * len(self.anchors_mask[i]), 1, 1).view(y.shape).type(FloatTensor)

            #按照网格格式生成先验框的宽高
            anchor_w = FloatTensor(scaled_anchors).index_select(1, LongTensor([0]))
            anchor_h = FloatTensor(scaled_anchors).index_select(1, LongTensor([1]))
            anchor_w = anchor_w.repeat(batch_size, 1).repeat(1, 1, input_height * input_width).view(w.shape)
            anchor_h = anchor_h.repeat(batch_size, 1).repeat(1, 1, input_height * input_width).view(h.shape)

            #利用预测结果对先验框进行调整
            #先调整先验框的中心，从先验框中心向右下角偏移
            #再调整先验框的宽高
            pred_boxes = FloatTensor(prediction[..., :4].shape)
            pred_boxes[..., 0] = x.data + grid_x
            pred_boxes[..., 1] = y.data + grid_y
            pred_boxes[..., 2] = nn.exp(w.data) * anchor_w
            pred_boxes[..., 3] = nn.exp(h.data) * anchor_h

            #将输出结果归一化成小数的形式
            _scale = nn.Tensor([input_width, input_height, input_width, input_height]).type(FloatTensor)
            output = nn.cat((pred_boxes.view(batch_size, -1, 4) / _scale,
                                conf.view(batch_size, -1, 1), pred_cls.view(batch_size, -1, self.num_classes)), -1)
            outputs.append(output.data)
        return outputs

#选取正确的先验框
    def yolo_correct_boxes(self,box_xy,box_wh,input_shape,image_shape,letterbox_image):

        box_yx = box_xy[...,::-1]
        box_hw = box_wh[...,::-1]
        input_shape = np.array(input_shape)
        image_shape = np.array(image_shape)

        if letterbox_image:
            #offset是图像有效区域相对于图像左上角的偏移情况
            #new_shape指的是宽高缩放情况
            new_shape = np.round(image_shape*np.min(input_shape/image_shape))
            offset = (input_shape - new_shape)/2./input_shape
            scale = input_shape/new_shape

            box_yx =(box_yx-offset)*scale
            box_hw *= scale

        box_mins = box_yx - (box_hw / 2.)
        box_maxes = box_yx + (box_hw / 2.)
        boxes = np.concatenate([box_mins[..., 0:1], box_mins[..., 1:2], box_maxes[..., 0:1], box_maxes[..., 1:2]],
                               axis=-1)
        boxes *= np.concatenate([image_shape, image_shape], axis=-1)
        return boxes

#非极大值抑制
    def non_max_suppression(self, prediction, num_classes, input_shape, image_shape, letterbox_image, conf_thres=0.5, nms_thres=0.4):
        #prediction[batch_size,num_anchors,85]

        box_corner = prediction.new(prediction.shape)
        box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
        box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
        box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
        box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
        prediction[:, :, :4] = box_corner[:, :, :4]

        output = [None for _ in range(len(prediction))]
        for i,image_pred in enumerate(prediction):
            #对种类预测部分取max
            #class_conf[num_anchors,1] 种类置信度
            #class_pred [num_anchors,1] 种类
            class_conf,class_pred = nn.max(image_pred[:,5:5+num_classes],1,keepdim = True)

            #利用置信度进行第一轮筛选
            conf_mask = (image_pred[:,4]*class_conf[:,0] >=conf_thres).squeeze()

            #根据置信度进行预测结果的筛选
            image_pred = image_pred[conf_mask]
            class_conf = class_conf[conf_mask]
            class_pred = class_pred[conf_mask]
            if not image_pred.size(0):
                continue

            detections = nn.cat((image_pred[:, :5], class_conf.float(), class_pred.float()), 1)

            #获得预测结果中包含的所有种类
            unique_labels = detections[:,-1].cpu().unique()

            if prediction.is_cuda:
                unique_labels = unique_labels.cuda()
                detections = detections.cuda()

            for c in unique_labels:
                #获得某一类得分筛选后全部的预测结果
                detections_class = detections[detections[:,-1] == c]

                #非极大抑制
                keep = nms(
                    detections_class[:,:4],
                    detections_class[:,4]*detections_class[:,5],
                    nms_thres
                )
                max_detections = detections_class[keep]

                output[i] = max_detections if output[i] is None else torch.cat((output[i], max_detections))

            if output[i] is not None:
                output[i] = output[i].cpu().numpy()
                box_xy, box_wh = (output[i][:, 0:2] + output[i][:, 2:4]) / 2, output[i][:, 2:4] - output[i][:, 0:2]
                output[i][:, :4] = self.yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape, letterbox_image)

        return output


