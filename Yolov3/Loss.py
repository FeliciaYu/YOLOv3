import torch
import torch.nn as nn
import math
import numpy as np

class YOLOlloss(nn.Module):
    def __init__(self,anchors,num_classes,input_shape, cuda, anchors_mask = [[6,7,8], [3,4,5], [0,1,2]]):
        super(YOLOlloss,self).__init__()

        self.anchors = anchors
        self.num_classes = num_classes
        self.bbox_attrs = 5+num_classes
        self.input_shape = input_shape
        self.anchors_mask = anchors_mask

        self.ignore_threshold = 0.7
        self.cuda = cuda

    def clip_by_tensor(self,t,t_min,t_max):
        t = t.float()

        #小于t_min的变为min
        result = (t>=t_min).float() * t + (t<t_min).float() *t_min

        #大于t_max的变为max
        result = (result<=t_max).float()*result + (t>t_max).float()*t_max

        return result

    def MSE(self,pred,target):
        return math.pow(target-pred,2)

    def BCE(self, pred, target):
        epsilon = 1e-7
        pred = self.clip_by_tensor(pred, epsilon, 1.0 - epsilon)
        output = - target * nn.log(pred) - (1.0 - target) * nn.log(1.0 - pred)
        return output

    def forward(self,L,input,targets = None):
        #L表示当前输入进来的有效特征层是第几个有效特征层
        #input的shape为bs, 3*(5+num_classes), 13, 13
        #             bs, 3*(5+num_classes), 26, 26
        #             bs, 3*(5+num_classes), 52, 52
        #target表示的是真实框
        batch_size = input.size(0)
        in_h = input.size(2)
        in_w = input.size(3)

        #步长
        stride_h = self.input_shape[0]/in_h
        stride_w = self.input_shape[1]/in_w
        #相对特征层大小的scaled_anchors
        scaled_anchors = [(a_w / stride_w, a_h / stride_h) for a_w, a_h in self.anchors]

        #prediction[batch_size,len(self.anchors_mask[L]),in_w,self.bbox_attrs,in_h]
        prediction = input.view(batch_size, len(self.anchors_mask[L]), self.bbox_attrs, in_h, in_w).permute(0, 1, 3, 4,2).contiguous()

        #先验框中心调整
        x = nn.sigmoid(prediction[..., 0])
        y = nn.sigmoid(prediction[..., 1])

        #先验框宽高
        w = prediction[..., 2]
        h = prediction[..., 3]

        #获得置信度
        conf = nn.sigmoid(prediction[...,4])
        #种类置信度
        pred_cls =nn.sigmoid(prediction[...,5:])
        y_true, noobj_mask, box_loss_scale = self.get_target(L, targets, scaled_anchors, in_h, in_w)
        #将预测结果进行解码，判断预测结果和真实值的重合程度
        #如果重合程度过大则忽略，因为这些特征点属于预测比较准确的特征点
        #作为负样本不合适
        noobj_mask, pred_boxes = self.get_ignore(L, x, y, h, w, targets, scaled_anchors, in_h, in_w, noobj_mask)

        if self.cuda:
            y_true = y_true.type_as(x)
            noobj_mask = noobj_mask.type_as(x)
            box_loss_scale = box_loss_scale.type_as(x)
            #box_loss_scale是真实框宽高的乘积，宽高均在0-1之间，因此乘积也在0-1之间。
            #2-宽高的乘积代表真实框越大，比重越小，小框的比重更大。
        box_loss_scale = 2 - box_loss_scale

        loss = 0
        obj_mask = y_true[..., 4] == 1
        n = nn.sum(obj_mask)
        if n != 0:
            if self.giou:
                #计算预测结果和真实结果的giou
                giou = self.box_giou(pred_boxes, y_true[..., :4]).type_as(x)
                loss_loc = nn.mean((1 - giou)[obj_mask])
            else:
                #计算中心偏移情况的loss，使用BCELoss效果好一些
                loss_x = nn.mean(self.BCE(x[obj_mask], y_true[..., 0][obj_mask]) * box_loss_scale[obj_mask])
                loss_y = nn.mean(self.BCE(y[obj_mask], y_true[..., 1][obj_mask]) * box_loss_scale[obj_mask])
                #计算宽高调整值的loss
                loss_w = nn.mean(self.MSE(w[obj_mask], y_true[..., 2][obj_mask]) * box_loss_scale[obj_mask])
                loss_h = nn.mean(self.MSE(h[obj_mask], y_true[..., 3][obj_mask]) * box_loss_scale[obj_mask])
                loss_loc = (loss_x + loss_y + loss_h + loss_w) * 0.1

            loss_cls = nn.mean(self.BCE(pred_cls[obj_mask], y_true[..., 5:][obj_mask]))
            loss += loss_loc * self.box_ratio + loss_cls * self.cls_ratio

        loss_conf = nn.mean(self.BCE(conf, obj_mask.type_as(conf))[noobj_mask.bool() | obj_mask])
        loss += loss_conf * self.balance[L] * self.obj_ratio
        # if n != 0:
        #print(loss_loc * self.box_ratio, loss_cls * self.cls_ratio, loss_conf * self.balance[l] * self.obj_ratio)
        return loss

    def get_target(self,L, targets, anchors, in_h, in_w):
        #计算图片的数量
        batch_size = len(targets)
        #作用：选取哪些先验框不包含物体
        noobj_mask = nn.ones(batch_size, len(self.anchors_mask[L]), in_h, in_w)
        box_loss_scale = nn.zeros(batch_size,len(self.anchors_mask[L]),in_h,in_w)
        y_true = nn.zeros(batch_size, len(self.anchors_mask[L]), in_h, in_w, self.bbox_attrs, requires_grad=False)
        for b in range(batch_size):
            if len(targets[b]) == 0:
                continue
            batch_target = nn.zeros_like(targets[b])
            #计算出正样本在特征层上的中心点
            batch_target[:, [0, 2]] = targets[b][:, [0, 2]] * in_w
            batch_target[:, [1, 3]] = targets[b][:, [1, 3]] * in_h
            batch_target[:, 4] = targets[b][:, 4]
            batch_target = batch_target.cpu()
            #将真实框转换一个形式
            #num_true_box, 4
            gt_box = nn.FloatTensor(nn.cat((nn.zeros((batch_target.size(0), 2)), batch_target[:, 2:4]), 1))
            #将先验框转换一个形式
            #9, 4
            anchor_shapes = nn.FloatTensor(nn.cat((nn.zeros((len(anchors), 2)), nn.FloatTensor(anchors)), 1))
            #计算交并比
            #self.calculate_iou(gt_box, anchor_shapes) = [num_true_box, 9]每一个真实框和9个先验框的重合情况
            #best_ns:
            #[每个真实框最大的重合度max_iou, 每一个真实框最重合的先验框的序号]
            best_ns = nn.argmax(self.calculate_iou(gt_box, anchor_shapes), dim=-1)

            for t, best_n in enumerate(best_ns):
                if best_n not in self.anchors_mask[L]:
                    continue
                #判断这个先验框是当前特征点的哪一个先验框
                k = self.anchors_mask[L].index(best_n)
                #获得真实框属于哪个网格点
                i = nn.floor(batch_target[t, 0]).long()
                j = nn.floor(batch_target[t, 1]).long()
                #取出真实框的种类
                c = batch_target[t, 4].long()

                #noobj_mask代表无目标的特征点
                noobj_mask[b, k, j, i] = 0
                #tx、ty代表中心调整参数的真实值
                if not self.giou:
                    #tx、ty代表中心调整参数的真实值
                    y_true[b, k, j, i, 0] = batch_target[t, 0] - i.float()
                    y_true[b, k, j, i, 1] = batch_target[t, 1] - j.float()
                    y_true[b, k, j, i, 2] = math.log(batch_target[t, 2] / anchors[best_n][0])
                    y_true[b, k, j, i, 3] = math.log(batch_target[t, 3] / anchors[best_n][1])
                    y_true[b, k, j, i, 4] = 1
                    y_true[b, k, j, i, c + 5] = 1
                else:
                    #tx、ty代表中心调整参数的真实值
                    y_true[b, k, j, i, 0] = batch_target[t, 0]
                    y_true[b, k, j, i, 1] = batch_target[t, 1]
                    y_true[b, k, j, i, 2] = batch_target[t, 2]
                    y_true[b, k, j, i, 3] = batch_target[t, 3]
                    y_true[b, k, j, i, 4] = 1
                    y_true[b, k, j, i, c + 5] = 1
                #用于获得xywh的比例
                #大目标loss权重小，小目标loss权重大
                box_loss_scale[b, k, j, i] = batch_target[t, 2] * batch_target[t, 3] / in_w / in_h
        return y_true, noobj_mask, box_loss_scale


