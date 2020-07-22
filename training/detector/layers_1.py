import numpy as np

import torch
from torch import nn
import math

class SpatialAttention3d(nn.Module):
    def __init__(self, channel):
        super(SpatialAttention3d, self).__init__()
        self.squeeze = nn.Conv3d(channel, 1, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # print(x.shape)
        z = self.squeeze(x)
        z = self.sigmoid(z)
        return x * z


class GAB(nn.Module):
    def __init__(self, input_dim, reduction=4):
        super(GAB, self).__init__()
        self.global_avgpool = nn.AdaptiveAvgPool3d(1)
        self.conv1 = nn.Conv3d(input_dim, input_dim // reduction, kernel_size=1, stride=1)
        self.conv2 = nn.Conv3d(input_dim // reduction, input_dim, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        z = self.global_avgpool(x)
        z = self.relu(self.conv1(z))
        z = self.sigmoid(self.conv2(z))
        return x * z

class SCse(nn.Module):
    def __init__(self, dim):
        super(SCse, self).__init__()
        self.satt = SpatialAttention3d(dim)
        self.catt = GAB(dim)

    def forward(self, x):
        return self.catt(x)
        # return self.satt(x) + self.catt(x)

class PostRes2d(nn.Module):
    def __init__(self, n_in, n_out, stride = 1):
        super(PostRes2d, self).__init__()
        self.conv1 = nn.Conv2d(n_in, n_out, kernel_size = 3, stride = stride, padding = 1)
        self.bn1 = nn.BatchNorm2d(n_out)
        self.relu = nn.ReLU(inplace = True)
        self.conv2 = nn.Conv2d(n_out, n_out, kernel_size = 3, padding = 1)
        self.bn2 = nn.BatchNorm2d(n_out)

        if stride != 1 or n_out != n_in:
            self.shortcut = nn.Sequential(
                nn.Conv2d(n_in, n_out, kernel_size = 1, stride = stride),
                nn.BatchNorm2d(n_out))
        else:
            self.shortcut = None

    def forward(self, x):
        residual = x
        if self.shortcut is not None:
            residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += residual
        out = self.relu(out)
        return out
    
class PostRes(nn.Module):
    def __init__(self, n_in, n_out, stride = 1):
        super(PostRes, self).__init__()
        self.conv1 = nn.Conv3d(n_in, n_out, kernel_size = 3, stride = stride, padding = 1)
        self.bn1 = nn.BatchNorm3d(n_out)
        self.relu = nn.ReLU(inplace = True)
        self.conv2 = nn.Conv3d(n_out, n_out, kernel_size = 3, padding = 1)
        self.bn2 = nn.BatchNorm3d(n_out)
        self.scse1 = SCse(n_out)
        self.scse2 = SCse(n_out)

        if stride != 1 or n_out != n_in:
            self.shortcut = nn.Sequential(
                nn.Conv3d(n_in, n_out, kernel_size = 1, stride = stride),
                nn.BatchNorm3d(n_out))
        else:
            self.shortcut = None

    def forward(self, x):
        residual = x
        if self.shortcut is not None:
            residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.scse1(out)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.scse2(out)
        out = self.bn2(out)
        
        out += residual
        out = self.relu(out)
        return out

class Rec3(nn.Module):
    def __init__(self, n0, n1, n2, n3, p = 0.0, integrate = True):
        super(Rec3, self).__init__()
        
        self.block01 = nn.Sequential(
            nn.Conv3d(n0, n1, kernel_size = 3, stride = 2, padding = 1),
            nn.BatchNorm3d(n1),
            nn.ReLU(inplace = True),
            nn.Conv3d(n1, n1, kernel_size = 3, padding = 1),
            nn.BatchNorm3d(n1))

        self.block11 = nn.Sequential(
            nn.Conv3d(n1, n1, kernel_size = 3, padding = 1),
            nn.BatchNorm3d(n1),
            nn.ReLU(inplace = True),
            nn.Conv3d(n1, n1, kernel_size = 3, padding = 1),
            nn.BatchNorm3d(n1))
        
        self.block21 = nn.Sequential(
            nn.ConvTranspose3d(n2, n1, kernel_size = 2, stride = 2),
            nn.BatchNorm3d(n1),
            nn.ReLU(inplace = True),
            nn.Conv3d(n1, n1, kernel_size = 3, padding = 1),
            nn.BatchNorm3d(n1))
 
        self.block12 = nn.Sequential(
            nn.Conv3d(n1, n2, kernel_size = 3, stride = 2, padding = 1),
            nn.BatchNorm3d(n2),
            nn.ReLU(inplace = True),
            nn.Conv3d(n2, n2, kernel_size = 3, padding = 1),
            nn.BatchNorm3d(n2))
        
        self.block22 = nn.Sequential(
            nn.Conv3d(n2, n2, kernel_size = 3, padding = 1),
            nn.BatchNorm3d(n2),
            nn.ReLU(inplace = True),
            nn.Conv3d(n2, n2, kernel_size = 3, padding = 1),
            nn.BatchNorm3d(n2))
        
        self.block32 = nn.Sequential(
            nn.ConvTranspose3d(n3, n2, kernel_size = 2, stride = 2),
            nn.BatchNorm3d(n2),
            nn.ReLU(inplace = True),
            nn.Conv3d(n2, n2, kernel_size = 3, padding = 1),
            nn.BatchNorm3d(n2))
 
        self.block23 = nn.Sequential(
            nn.Conv3d(n2, n3, kernel_size = 3, stride = 2, padding = 1),
            nn.BatchNorm3d(n3),
            nn.ReLU(inplace = True),
            nn.Conv3d(n3, n3, kernel_size = 3, padding = 1),
            nn.BatchNorm3d(n3))

        self.block33 = nn.Sequential(
            nn.Conv3d(n3, n3, kernel_size = 3, padding = 1),
            nn.BatchNorm3d(n3),
            nn.ReLU(inplace = True),
            nn.Conv3d(n3, n3, kernel_size = 3, padding = 1),
            nn.BatchNorm3d(n3))

        self.relu = nn.ReLU(inplace = True)
        self.p = p
        self.integrate = integrate

    def forward(self, x0, x1, x2, x3):
        if self.p > 0 and self.training:
            coef = torch.bernoulli((1.0 - self.p) * torch.ones(8))
            out1 = coef[0] * self.block01(x0) + coef[1] * self.block11(x1) + coef[2] * self.block21(x2)
            out2 = coef[3] * self.block12(x1) + coef[4] * self.block22(x2) + coef[5] * self.block32(x3)
            out3 = coef[6] * self.block23(x2) + coef[7] * self.block33(x3)
        else:
            out1 = (1 - self.p) * (self.block01(x0) + self.block11(x1) + self.block21(x2))
            out2 = (1 - self.p) * (self.block12(x1) + self.block22(x2) + self.block32(x3))
            out3 = (1 - self.p) * (self.block23(x2) + self.block33(x3))

        if self.integrate:
            out1 += x1
            out2 += x2
            out3 += x3

        return x0, self.relu(out1), self.relu(out2), self.relu(out3)

def hard_mining(neg_output, neg_labels, num_hard):
    _, idcs = torch.topk(neg_output, min(num_hard, len(neg_output)))
    neg_output = torch.index_select(neg_output, 0, idcs)
    neg_labels = torch.index_select(neg_labels, 0, idcs)
    return neg_output, neg_labels

class base_Loss(nn.Module):
    def __init__(self, num_hard = 0):
        super(base_Loss, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.classify_loss = nn.BCELoss()
        self.regress_loss = nn.SmoothL1Loss()
        self.num_hard = num_hard

    def forward(self, pos_output,pos_labels,neg_output,neg_labels, train = True):
        batch_size = pos_labels.size(0)
        if batch_size == 0: 
            batch_size=1
        if self.num_hard > 0 and train:
            neg_output, neg_labels = hard_mining(neg_output, neg_labels, self.num_hard * batch_size)
        neg_prob = self.sigmoid(neg_output)

        #classify_loss = self.classify_loss(
         #   torch.cat((pos_prob, neg_prob), 0),
          #  torch.cat((pos_labels[:, 0], neg_labels + 1), 0))
        if len(pos_output)>0:
            pos_prob = self.sigmoid(pos_output[:, 0])
            pz, ph, pw, pd = pos_output[:, 1], pos_output[:, 2], pos_output[:, 3], pos_output[:, 4]
            lz, lh, lw, ld = pos_labels[:, 1], pos_labels[:, 2], pos_labels[:, 3], pos_labels[:, 4]

            regress_losses = [
                self.regress_loss(pz, lz),
                self.regress_loss(ph, lh),
                self.regress_loss(pw, lw),
                self.regress_loss(pd, ld)]
            regress_losses_data = [l.item() for l in regress_losses]
            classify_loss = 0.5 * self.classify_loss(
            pos_prob, pos_labels[:, 0]) + 0.5 * self.classify_loss(
            neg_prob, neg_labels + 1)
            pos_correct = (pos_prob.data >= 0.5).sum()
            pos_total = len(pos_prob)

        else:
            regress_losses = [0,0,0,0]
            classify_loss =  0.5 * self.classify_loss(
            neg_prob, neg_labels + 1)
            pos_correct = 0
            pos_total = 0
            regress_losses_data = [0,0,0,0]
        classify_loss_data = classify_loss.item()
        #print(classify_loss_data)
        loss = classify_loss
        for regress_loss in regress_losses:
            loss += regress_loss

        neg_correct = (neg_prob.data < 0.5).sum()
        neg_total = len(neg_prob)

        return [loss, classify_loss_data] + regress_losses_data + [pos_correct, pos_total, neg_correct, neg_total]

class Loss(nn.Module):
    def __init__(self, num_hard = 0):
        super(Loss, self).__init__()
        self.num_hard = num_hard
        self.base_Loss1 = base_Loss(self.num_hard)
        self.base_Loss2 = base_Loss(self.num_hard)
        self.base_Loss3 = base_Loss(self.num_hard)
        self.base_Loss4 = base_Loss(self.num_hard)
    def forward(self, output1,output2,output3,output4, labels, train=True):
        output1 = output1.view(-1, 5)
        output2 = output2.view(-1, 5)
        output3 = output3.view(-1, 5)
        output4 = output4.view(-1, 5)
        labels = labels.view(-1, 5)

        pos_idcs1 = labels[:, 0] == 1
        pos_idcs1 = pos_idcs1.unsqueeze(1).expand(pos_idcs1.size(0), 5)
        pos_output1 = output1[pos_idcs1].view(-1, 5)
        pos_labels1 = labels[pos_idcs1].view(-1, 5)
        pos_labels1[:,0]=1

        neg_idcs1 = labels[:, 0] == -1
        neg_output1 = output1[:, 0][neg_idcs1]
        neg_labels1 = labels[:, 0][neg_idcs1]

        pos_idcs2 = labels[:, 0] == 2
        pos_idcs2 = pos_idcs2.unsqueeze(1).expand(pos_idcs2.size(0), 5)
        pos_output2 = output2[pos_idcs2].view(-1, 5)
        pos_labels2 = labels[pos_idcs2].view(-1, 5)
        pos_labels2[:,0]=1

        neg_idcs2 = labels[:, 0] == -1
        neg_output2 = output2[:, 0][neg_idcs2]
        neg_labels2 = labels[:, 0][neg_idcs2]

        pos_idcs3 = labels[:, 0] == 3
        pos_idcs3 = pos_idcs3.unsqueeze(1).expand(pos_idcs3.size(0), 5)
        pos_output3 = output3[pos_idcs3].view(-1, 5)
        pos_labels3 = labels[pos_idcs3].view(-1, 5)
        pos_labels3[:,0]=1

        neg_idcs3 = labels[:, 0] == -1
        neg_output3 = output3[:, 0][neg_idcs3]
        neg_labels3 = labels[:, 0][neg_idcs3]

        pos_idcs4 = labels[:, 0] == 4
        pos_idcs4 = pos_idcs4.unsqueeze(1).expand(pos_idcs4.size(0), 5)
        pos_output4 = output4[pos_idcs4].view(-1, 5)
        pos_labels4 = labels[pos_idcs4].view(-1, 5)
        pos_labels4[:,0]=1

        neg_idcs4 = labels[:, 0] == -1
        neg_output4 = output4[:, 0][neg_idcs4]
        neg_labels4 = labels[:, 0][neg_idcs4]

        loss1, classify_loss_data1, regress_losses_data1_1, regress_losses_data1_2, regress_losses_data1_3, regress_losses_data1_4, pos_correct1, pos_total1, neg_correct1, neg_total1 = self.base_Loss1(pos_output1,pos_labels1,neg_output1,neg_labels1,train)
        loss2, classify_loss_data2, regress_losses_data2_1, regress_losses_data2_2, regress_losses_data2_3, regress_losses_data2_4, pos_correct2, pos_total2, neg_correct2, neg_total2 = self.base_Loss2(pos_output2,pos_labels2,neg_output2,neg_labels2,train)
        loss3, classify_loss_data3, regress_losses_data3_1, regress_losses_data3_2, regress_losses_data3_3, regress_losses_data3_4, pos_correct3, pos_total3, neg_correct3, neg_total3 = self.base_Loss3(pos_output3,pos_labels3,neg_output3,neg_labels3,train)
        loss4, classify_loss_data4, regress_losses_data4_1, regress_losses_data4_2, regress_losses_data4_3, regress_losses_data4_4, pos_correct4, pos_total4, neg_correct4, neg_total4 = self.base_Loss4(pos_output4,pos_labels4,neg_output4,neg_labels4,train)
        loss = (loss1 + loss2 + loss3 + loss4)/4
        classify_loss_data = (classify_loss_data1 + classify_loss_data2 + classify_loss_data3 + classify_loss_data4)/4
        pos_correct = pos_correct1 + pos_correct2 + pos_correct3 + pos_correct4
        pos_total = pos_total1 + pos_total2 + pos_total3 + pos_total4
        neg_correct = neg_correct1 + neg_correct2 + neg_correct3 + neg_correct4
        neg_total = neg_total1 + neg_total2 + neg_total3 + neg_total4
        data0 = (regress_losses_data1_1 + regress_losses_data2_1 + regress_losses_data3_1 + regress_losses_data4_1)/4
        data1 = (regress_losses_data1_2 + regress_losses_data2_2 + regress_losses_data3_2 + regress_losses_data4_2)/4
        data2 = (regress_losses_data1_3 + regress_losses_data2_3 + regress_losses_data3_3 + regress_losses_data4_3)/4
        data3 = (regress_losses_data1_4 + regress_losses_data2_4 + regress_losses_data3_4 + regress_losses_data4_4)/4
        regress_losses_data = [data0,data1,data2,data3]
        return [loss, classify_loss_data] + regress_losses_data + [pos_correct, pos_total, neg_correct, neg_total]


        

def sigmoid(x):
  return 1.0 / (1.0 + np.exp(-x))
class GetPBB(object):
    def __init__(self, config):
        self.stride = config['stride']
        self.anchors = np.asarray(config['anchors'])
        self.conf_th = config['conf_th']
        self.nms_th = config['nms_th']

    def __call__(self, output,thresh = -3, ismask=False):
        stride = self.stride
        anchors = self.anchors
        output = np.copy(output)
        offset = (float(stride) - 1) / 2
        output_size = output.shape
        oz = np.arange(offset, offset + stride * (output_size[0] - 1) + 1, stride)
        oh = np.arange(offset, offset + stride * (output_size[1] - 1) + 1, stride)
        ow = np.arange(offset, offset + stride * (output_size[2] - 1) + 1, stride)
        
        output[:, :, :, :, 1] = oz.reshape((-1, 1, 1, 1)) + output[:, :, :, :, 1] * anchors.reshape((1, 1, 1, -1))
        output[:, :, :, :, 2] = oh.reshape((1, -1, 1, 1)) + output[:, :, :, :, 2] * anchors.reshape((1, 1, 1, -1))
        output[:, :, :, :, 3] = ow.reshape((1, 1, -1, 1)) + output[:, :, :, :, 3] * anchors.reshape((1, 1, 1, -1))
        output[:, :, :, :, 4] = np.exp(output[:, :, :, :, 4]) * anchors.reshape((1, 1, 1, -1))
        mask = output[..., 0] > thresh
        xx,yy,zz,aa = np.where(mask)

       
        output = output[xx,yy,zz,aa]
        output_ = output[output[:, 0] >= self.conf_th] 
        bboxes = nms(output_, self.nms_th)
        #print(bboxes) 

        if ismask:
            return output,[xx,yy,zz,aa],bboxes
        else:
            return output
def nms(output, nms_th):
    if len(output) == 0:
        return output

    output = output[np.argsort(-output[:, 0])]
    bboxes = [output[0]]
    
    for i in np.arange(1, len(output)):
        bbox = output[i]
        flag = 1
        for j in range(len(bboxes)):
            if iou(bbox[1:5], bboxes[j][1:5]) >= nms_th:
                flag = -1
                break
        if flag == 1:
            bboxes.append(bbox)
    
    bboxes = np.asarray(bboxes, np.float32)
    return bboxes

def iou(box0, box1):
    
    r0 = box0[3] / 2
    s0 = box0[:3] - r0
    e0 = box0[:3] + r0

    r1 = box1[3] / 2
    s1 = box1[:3] - r1
    e1 = box1[:3] + r1

    overlap = []
    for i in range(len(s0)):
        overlap.append(max(0, min(e0[i], e1[i]) - max(s0[i], s1[i])))

    intersection = overlap[0] * overlap[1] * overlap[2]
    union = box0[3] * box0[3] * box0[3] + box1[3] * box1[3] * box1[3] - intersection
    return intersection / union

def acc(pbb, lbb, conf_th, nms_th, detect_th):
    pbb = pbb[pbb[:, 0] >= conf_th] 
    pbb = nms(pbb, nms_th)

    tp = []
    fp = []
    fn = []
    l_flag = np.zeros((len(lbb),), np.int32)
    for p in pbb:
        flag = 0
        bestscore = 0
        for i, l in enumerate(lbb):
            score = iou(p[1:5], l)
            if score>bestscore:
                bestscore = score
                besti = i
        if bestscore > detect_th:
            flag = 1
            if l_flag[besti] == 0:
                l_flag[besti] = 1
                tp.append(np.concatenate([p,[bestscore]],0))
            else:
                fp.append(np.concatenate([p,[bestscore]],0))
        if flag == 0:
            fp.append(np.concatenate([p,[bestscore]],0))
    for i,l in enumerate(lbb):
        if l_flag[i]==0:
            score = []
            for p in pbb:
                score.append(iou(p[1:5],l))
            if len(score)!=0:
                bestscore = np.max(score)
            else:
                bestscore = 0
            if bestscore<detect_th:
                fn.append(np.concatenate([l,[bestscore]],0))

    return tp, fp, fn, len(lbb)    

def topkpbb(pbb,lbb,nms_th,detect_th,topk=30):
    conf_th = 0
    fp = []
    tp = []
    while len(tp)+len(fp)<topk:
        conf_th = conf_th-0.2
        tp, fp, fn, _ = acc(pbb, lbb, conf_th, nms_th, detect_th)
        if conf_th<-3:
            break
    tp = np.array(tp).reshape([len(tp),6])
    fp = np.array(fp).reshape([len(fp),6])
    fn = np.array(fn).reshape([len(fn),5])
    allp  = np.concatenate([tp,fp],0)
    sorting = np.argsort(allp[:,0])[::-1]
    n_tp = len(tp)
    topk = np.min([topk,len(allp)])
    tp_in_topk = np.array([i for i in range(n_tp) if i in sorting[:topk]])
    fp_in_topk = np.array([i for i in range(topk) if sorting[i] not in range(n_tp)])
#     print(fp_in_topk)
    fn_i =       np.array([i for i in range(n_tp) if i not in sorting[:topk]])
    newallp = allp[:topk]
    if len(fn_i)>0:
        fn = np.concatenate([fn,tp[fn_i,:5]])
    else:
        fn = fn
    if len(tp_in_topk)>0:
        tp = tp[tp_in_topk]
    else:
        tp = []
    if len(fp_in_topk)>0:
        fp = newallp[fp_in_topk]
    else:
        fp = []
    return tp, fp , fn
