import torch
import torch.nn as nn
import torch.nn.functional as F

from models.vgg import vgg_sg as vgg
from losses import CE_DiceLoss,DiceLoss,FocalLoss2d

class WeightCrossEntropy(nn.Module):

    def __init__(self,size_average=True):
        super(WeightCrossEntropy,self).__init__()
        self.size_average = size_average

    def forward(self,pred,target):

        mask = torch.zeros(2,requires_grad=False).cuda()
        num_positive = torch.sum((target==1)).float()
        num_negative = torch.sum((target==0)).float()
        mask[1] = num_positive / (num_negative + num_positive)
        mask[0] = num_negative / (num_positive + num_negative)

        # mask[mask==1] = 1.0 * num_negative /(num_negative + num_positive)
        # mask[mask==0] = 1.0 * num_positive /(num_negative + num_positive)

        loss = F.cross_entropy(pred,target,weight=mask)

        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()



class OneShotModel(nn.Module):

    def __init__(self):
        super(OneShotModel, self).__init__()

        self.netB = vgg.vgg16(pretrained=False, use_decoder=True)

        self.classifier_6 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, dilation=1,  padding=1),  
            nn.ReLU(inplace=True)
        )
        self.exit_layer = nn.Conv2d(128, 2, kernel_size=1, padding=1)

        # self.bce_logits_func = nn.CrossEntropyLoss()
        # self.bce_logits_func = WeightCrossEntropy()
        self.bce_logits_func = FocalLoss2d()
        self.cos_similarity_func = nn.CosineSimilarity()
        # self.triplelet_func = nn.TripletMarginLoss(margin=2.0)

    def forward(self, support_img, support_mask, query_img):
        '''
        support_mask :[1,H,W]
        support_img : [3,H,W]
        query_img : [3,H,W]
        '''
        outA_pos, _ = self.netB(support_img)

        _, mask_w, mask_h = support_mask.size()
        outA_pos = F.interpolate(outA_pos, size=(mask_w, mask_h), mode='bilinear',align_corners=True)
        vec_pos = torch.sum(torch.sum(outA_pos * support_mask, dim=3), dim=2)/torch.sum(support_mask) #batch_size=1,[1,512]
        outB, outB_side= self.netB(query_img) #[1,512,64,64]

        vec_pos = vec_pos.unsqueeze(dim=2).unsqueeze(dim=3)
        tmp_seg = self.cos_similarity_func(outB, vec_pos) #[1,64,64]

        exit_feat_in = outB_side * tmp_seg.unsqueeze(dim=1)
        outB_side_6 = self.classifier_6(exit_feat_in)
        out = self.exit_layer(outB_side_6)
        # print(out)
        # print(out.size())

        return out
    
    def get_celoss(self,logits,query_label):
        '''
        logits :[2,512,512]
        query_label: [1,512,512]
        '''
        H,W = query_label.size()[-2:]
        out = F.interpolate(logits,size=(W,H),mode='bilinear',align_corners=True)
        celoss = self.bce_logits_func(out,query_label.long())

        return celoss

    def get_pred(self,logits,query_img):
        '''

        :param logits: [1,2,66,66]
        :param query_img: [1,512,512]
        :return: [512,512]
        '''
        H,W = query_img.size()[-2:]
        out = F.interpolate(logits,size=(W,H),mode='bilinear',align_corners=True)

        out_softmax = F.softmax(out, dim=1).squeeze()
        pred = torch.argmax(out_softmax,dim=0)

        return pred
    
    def get_acc(self,pred,label):
        '''

        :param pred: [512,512]
        :param label: [512,512]
        :return:
        '''

        # e = torch.tensor(1e-6,dtype=torch.long).cuda()
        tp = ((label == 1) & (pred == 1)).sum()
        print(tp)

        if torch.sum(pred==1)==0:
            precison = torch.tensor(0,dtype=torch.long)
        else:
            precison = tp/(torch.sum(pred==1))
        print(precison)
        recall = tp /torch.sum(label==1)
        print(recall)

        return precison,recall
        




