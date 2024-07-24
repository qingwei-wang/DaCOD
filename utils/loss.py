from ctypes import Union
import torch
import torch.nn as nn
import torch.nn.functional as F

#     iou loss

class IOU(nn.Module):
    def __init__(self):
        super(IOU,self).__init__()

    
    def _iou(self,pred,gt):
        pred = torch.sigmoid(pred)
        inter = (pred * gt).sum(dim = (2 , 3))
        union = (pred + gt).sum(dim=(2, 3)) - inter
        iou = 1 - (inter / union)
        
        return iou.mean()

    def forward(self, pred, gt):
        return self._iou(pred, gt)


#   structure loss

class structure_loss(nn.Module):
    def __init__(self):
        super(structure_loss,self).__init__()

    def _structure_loss(self,pred,gt):
        weit = 1 + 5 * torch.abs(F.avg_pool2d(gt, kernel_size=31, stride=1, padding=15) - gt)
        wbce = F.binary_cross_entropy_with_logits(pred, gt, reduce='none')
        wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

        pred = torch.sigmoid(pred)
        inter = ((pred * gt) * weit).sum(dim=(2, 3))
        union = ((pred + gt) * weit).sum(dim=(2, 3))
        wiou = 1 - (inter) / (union - inter)
        return (wbce + wiou).mean()

    def forward(self, pred, gt):
        return self._structure_loss(pred, gt)