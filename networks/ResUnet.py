import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from models.resunet import Encoder


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class UpBasicBlock(nn.Module):
    
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(UpBasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

    def forward(self, x):

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        return out

class _Decoder(nn.Module):

    def __init__(self,block,layers):
        super(_Decoder,self).__init__()
        self.inplanes = 512
       
        self.decoder_1 = self._make_layer(block,256,layers[0])
        self.decoder_2 = self._make_layer(block,128,layers[1])
        self.decoder_3 = self._make_layer(block,64, layers[2])

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self,block,planes,blocks):

        layers = []
        self.inplanes = self.inplanes + planes
        layers.append(block(self.inplanes, planes))
        self.inplanes = planes
        for i in range(1,blocks):
            layers.append(block(planes, planes))

        return nn.Sequential(*layers)
    
    def forward(self, x):

        decoder1 = self.decoder_1(x)
        decoder2 = self.decoder_2(decoder1)
        decoder3 = self.decoder_3(decoder2)
       
        return decoder3

def Decoder(**kwargs):
    """Constructs a model to get Image Embedding.
    """
    model = _Decoder(UpBasicBlock, [2, 2, 2, 2], **kwargs)

    return model


class ConvOut(nn.Module):

    def __init__(self,in_channels,mid_channels,out_channels):
        super(ConvOut,self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels,mid_channels,kernel_size=3,padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(),
            nn.Conv2d(mid_channels,out_channels,kernel_size=1),
            nn.ReLU()
        )
    
    def forward(self,x):

        return self.block(x)


class ResUnet(nn.Module):

    def __init__(self):

        super(ResUnet,self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.conv_out = ConvOut(64,32,2)

    def forward(self,x):
        
        res1,res2,res3,res4 = self.encoder(x)
        up5 = F.interpolate(res4,scale_factor=2,mode='bilinear',align_corners=True)
        dec1 = self.decoder.decoder_1(torch.cat([up5,res3],dim=1))
        up6 = F.interpolate(dec1,scale_factor=2,mode='bilinear',align_corners=True)
        dec2 = self.decoder.decoder_2(torch.cat([up6,res2],dim=1))
        up7 = F.interpolate(dec2,scale_factor=2,mode='bilinear',align_corners=True)
        dec3 = self.decoder.decoder_3(torch.cat([up7,res1],dim=1))
        up8 = F.interpolate(dec3,scale_factor=2,mode='bilinear',align_corners=True)

        out = self.conv_out(up8)

        return out

class ResUnetOneShot(nn.Module):

    def __init__(self):

        super(ResUnetOneShot,self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.conv_out = ConvOut(64,32,2)
        self.cos_similarity_func = nn.CosineSimilarity()
        self.global_pool = nn.AdaptiveAvgPool2d((1,1))
        self.bce_logits_func = nn.CrossEntropyLoss()
    
    def forward(self,support_img,query_img):

        sup_res1,sup_res2,sup_res3,sup_res4 = self.encoder(support_img)
        que_res1,que_res2,que_res3,que_res4 = self.encoder(query_img)

        vec_sup_res1 = self.global_pool(sup_res1)  #[B,C,1,1]
        atten1 = self.cos_similarity_func(vec_sup_res1,que_res1) #[B,H,W]
        res1 = que_res1 * atten1.unsqueeze(dim=1)

        vec_sup_res2 = self.global_pool(sup_res2)
        atten2 = self.cos_similarity_func(vec_sup_res2,que_res2)
        res2 = que_res2 * atten2.unsqueeze(dim=1)

        vec_sup_res3 = self.global_pool(sup_res3)
        atten3 = self.cos_similarity_func(vec_sup_res3,que_res3)
        res3 = que_res3 * atten3.unsqueeze(dim=1)

        vec_sup_res4 = self.global_pool(sup_res4)
        atten4 = self.cos_similarity_func(vec_sup_res4,que_res4)
        res4 = que_res4 * atten4.unsqueeze(dim=1)

        up5 = F.interpolate(res4,scale_factor=2,mode='bilinear',align_corners=True)
        dec1 = self.decoder.decoder_1(torch.cat([up5,res3],dim=1))
        up6 = F.interpolate(dec1,scale_factor=2,mode='bilinear',align_corners=True)
        dec2 = self.decoder.decoder_2(torch.cat([up6,res2],dim=1))
        up7 = F.interpolate(dec2,scale_factor=2,mode='bilinear',align_corners=True)
        dec3 = self.decoder.decoder_3(torch.cat([up7,res1],dim=1))
        up8 = F.interpolate(dec3,scale_factor=2,mode='bilinear',align_corners=True)

        out = self.conv_out(up8)

        return out
    
    def get_celoss(self,logits,query_label):
        '''
        logits :[2,512,512]
        query_label: [1,512,512]
        '''
        celoss = self.bce_logits_func(logits,query_label.long())

        return celoss

    def get_pred(self,logits):
        '''

        :param logits: [1,2,66,66]
        :param query_img: [1,512,512]
        :return: [512,512]
        '''
        # H,W = query_img.size()[-2:]
        # out = F.interpolate(logits,size=(W,H),mode='bilinear',align_corners=True)

        out_softmax = F.softmax(logits, dim=1).squeeze()
        pred = torch.argmax(out_softmax,dim=0)

        return pred


if __name__ == "__main__":

    from torchsummary import summary
    m = ResUnet()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    m.to(device)
    summary(m, input_size=(3,256,256))

    # m = nn.AdaptiveAvgPool2d((5,7))  
    # input = torch.randn(1, 32, 8, 9)  
    # simi = nn.CosineSimilarity()
    # vec = torch.randn(1,32,1,1)
    # output = m(input) 
    # print(output.shape) #[1,32,5,7]
    # simila = simi(vec,input)
    # print(simila.shape)
    


        






        