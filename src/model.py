'''
This is a fully convolutional neural network implementation using pytorch with a few minor architectural 
modifications from the original paper


'''
from torch import nn
import torch
from torchvision.models import swin_v2_t, Swin_V2_T_Weights
import torchvision.transforms as T
from torch.nn import functional as F 
import math
from pathlib import Path
from PIL import Image


class SwinSegOld(nn.Module):

        
    def __init__(self, n_class):        
        super().__init__()

        # pretrained base net extractor swin-v2-t 
        base_model = swin_v2_t(weights=Swin_V2_T_Weights.DEFAULT)
        self.base = base_model.features
        self.permute = base_model.permute
        self.norm = base_model.norm
        
        # returns prediction scores for intermediate layers and final
        self.fm96_proj = nn.Conv2d(96, 256, 1)
        self.fm192_proj = nn.Conv2d(192, 256, 1)
        self.fm384_proj = nn.Conv2d(384, 256, 1)
        self.fm768_proj = nn.Conv2d(768, 256, 1)
        
        # smoothen fusion
        self.smooth4 = nn.Conv2d(256, 256, 3, 1, padding=1)
        self.smooth8 = nn.Conv2d(256, 256, 3, 1, padding=1)
        self.smooth16 = nn.Conv2d(256, 256, 3, 1, padding=1)
        
        # equivalent to the layernorm used in swin
        self.norm256 = nn.GroupNorm(1, 256)
        self.relu = nn.ReLU()

        # 256 -> class size
        self.score4 = nn.Conv2d(256, n_class, 1)
        self.score8 = nn.Conv2d(256, n_class, 1)
        self.score16 = nn.Conv2d(256, n_class, 1)
        self.score32 = nn.Conv2d(256, n_class, 1)
        
        # initialize weights with kaiming
        self.apply(self._init_weights)
    
    def forward(self, x, weights=[1.3, 0.5, 0.3, 0.3]):
        '''
        args:
        - x: tensor(batch_size, channel_size, height, width)
        
        output:
        - tensor(batch_size, class_size, height, width)
        
        '''
        # get img spatial dimensions
        img_res = x.shape[-2:]
        
        
        ''' forward pass through swin backbone'''
        
        x = self.base[0](x)
        fm96 = x = self.base[1](x)        
        x = self.base[2](x)         
        fm192 = x = self.base[3](x)
        x = self.base[4](x)         
        fm384 = x = self.base[5](x)
        x = self.base[6](x)
        x = self.base[7](x)         
        
        ''' norm and rearrange tensor dims'''
        
        fm96 = self.permute(fm96)
        fm192 = self.permute(fm192)
        fm384 = self.permute(fm384)
        fm768 = self.permute(self.norm(x))
        
        ''' feature pyramid network '''
        
        # project to 256 dimensions and normalize
        fm96 = self.relu(self.norm256(self.fm96_proj(fm96)))
        fm192 = self.relu(self.norm256(self.fm192_proj(fm192)))
        fm384 = self.relu(self.norm256(self.fm384_proj(fm384)))
        fm768 = self.relu(self.norm256(self.fm768_proj(fm768)))
        
        fuse16 = self.relu(self.norm256(self.smooth16(fm384 + F.interpolate(fm768, size=fm384.shape[-2:], mode='bilinear', align_corners=False))))
        fuse8 = self.relu(self.norm256(self.smooth8(fm192 + F.interpolate(fuse16, size=fm192.shape[-2:], mode='bilinear', align_corners=False))))
        fuse4 = self.relu(self.norm256(self.smooth4(fm96 + F.interpolate(fuse8, size=fm96.shape[-2:], mode='bilinear', align_corners=False))))
        
        score32 = F.interpolate(self.score32(fm768), size=img_res, mode='bilinear', align_corners=False)
        score16 = F.interpolate(self.score16(fuse16), size=img_res, mode='bilinear', align_corners=False)
        score8 = F.interpolate(self.score8(fuse8), size=img_res, mode='bilinear', align_corners=False)
        score4 = F.interpolate(self.score4(fuse4), size=img_res, mode='bilinear', align_corners=False)        
        
        # deep supervision
        return [
            (score4, weights[0]),
            (score8, weights[1]),
            (score16, weights[2]),
            (score32, weights[3])
        ]
        
        