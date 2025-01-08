'''
This is a fully convolutional neural network implementation using pytorch with a few minor architectural 
modifications from the original paper


'''
from torch import nn
import torch
from torchvision.models import swin_v2_t, Swin_V2_T_Weights
from torch.nn import functional as F 


class SwinSeg(nn.Module):
    def __init__(self, n_class):        
        super().__init__()

        # pretrained base net extractor swin-v2-t 
        base_model = swin_v2_t(weights=Swin_V2_T_Weights.DEFAULT)
        self.base = base_model.features
        self.permute = base_model.permute
        self.norm = base_model.norm
        
        # returns prediction scores for intermediate layers and final
        self.fm96_proj = nn.Sequential(
            nn.Conv2d(96, 48, 1),
            nn.ReLU(),
            nn.Conv2d(48, 96, 1),
        )
        
        self.fm192_proj = nn.Sequential(
            nn.Conv2d(192, 96, 1),
            nn.ReLU(),
            nn.Conv2d(96, 192, 1),
        )
        
        self.fm384_proj = nn.Sequential(
            nn.Conv2d(384, 192, 1),
            nn.ReLU(),
            nn.Conv2d(192, 384, 1),
        )
        
        self.fm768_proj = nn.Sequential(
            nn.Conv2d(768, 384, 1),
            nn.ReLU(),
            nn.Conv2d(384, 768, 1),
        )
        
        # smoothen fusion
        self.smooth4 = nn.Conv2d(96, 96, 3, 1, padding=1)
        self.smooth8 = nn.Conv2d(192, 192, 3, 1, padding=1)
        self.smooth16 = nn.Conv2d(384, 384, 3, 1, padding=1)
        
        # equivalent to the layernorm used in swin
        self.norm768 = nn.GroupNorm(1, 768)
        self.norm384 = nn.GroupNorm(1, 384)
        self.norm192 = nn.GroupNorm(1, 192)
        self.norm96 = nn.GroupNorm(1, 96)
        self.relu = nn.ReLU()
        
        # 256 -> class size
        self.score8 = nn.Conv2d(96, n_class, 1)
        self.score16 = nn.Conv2d(192, n_class, 1)
        self.score32 = nn.Conv2d(384, n_class, 1)
        
        self.gate32 = nn.Sequential(
            nn.Conv2d(1536, 768, 1),
            nn.ReLU(),
            nn.Conv2d(768, 384, 1),
        )
        
        self.gate16 = nn.Sequential(
            nn.Conv2d(768, 384, 1),
            nn.ReLU(),
            nn.Conv2d(384, 192, 1),
        )
        
        self.gate8 = nn.Sequential(
            nn.Conv2d(384, 192, 1),
            nn.ReLU(),
            nn.Conv2d(192, 96, 1),
        )
      
        self.gate4 = nn.Sequential(
            nn.Conv2d(192, 96, 1),
            nn.ReLU(),
            nn.Conv2d(96, n_class, 1),
        )
        
      
        # initialize weights with kaiming
        nn.init.kaiming_uniform_(self.fm96_proj[0].weight)
        nn.init.kaiming_uniform_(self.fm96_proj[2].weight)
        nn.init.kaiming_uniform_(self.fm192_proj[0].weight)
        nn.init.kaiming_uniform_(self.fm192_proj[2].weight)
        nn.init.kaiming_uniform_(self.fm384_proj[0].weight)
        nn.init.kaiming_uniform_(self.fm384_proj[2].weight)
        nn.init.kaiming_uniform_(self.fm768_proj[0].weight)
        nn.init.kaiming_uniform_(self.fm768_proj[2].weight)

        
        nn.init.kaiming_uniform_(self.smooth4.weight)
        nn.init.kaiming_uniform_(self.smooth8.weight)
        nn.init.kaiming_uniform_(self.smooth16.weight)
        
        
        nn.init.kaiming_uniform_(self.gate32[0].weight)
        nn.init.kaiming_uniform_(self.gate32[2].weight)
        nn.init.kaiming_uniform_(self.gate16[0].weight)
        nn.init.kaiming_uniform_(self.gate16[2].weight)
        nn.init.kaiming_uniform_(self.gate8[0].weight)
        nn.init.kaiming_uniform_(self.gate8[2].weight)
        nn.init.kaiming_uniform_(self.gate4[0].weight)
        nn.init.kaiming_uniform_(self.gate4[2].weight)
        
        nn.init.kaiming_uniform_(self.score32.weight)
        nn.init.kaiming_uniform_(self.score16.weight)
        nn.init.kaiming_uniform_(self.score8.weight)
        
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
        
        fm96_raw = self.permute(fm96)
        fm192_raw = self.permute(fm192)
        fm384_raw = self.permute(fm384)
        fm768_raw = self.permute(self.norm(x))
        
        ''' feature pyramid network '''
        
        # project to 384 dimensions and normalize
        fm96 = self.relu(self.norm96(self.fm96_proj(fm96_raw)))
        fm192 = self.relu(self.norm192(self.fm192_proj(fm192_raw)))
        fm384 = self.relu(self.norm384(self.fm384_proj(fm384_raw)))
        fm768 = self.relu(self.norm768(self.fm768_proj(fm768_raw)))
        
        # feature pyramid
        fuse32 = self.relu(self.norm384(self.gate32(torch.cat([fm768, fm768_raw], dim=1))))
        fuse16 = self.smooth16(fm384 + F.interpolate(fuse32, size=fm384.shape[-2:], mode='bilinear', align_corners=False))
        fuse16 = self.relu(self.norm192(self.gate16(torch.cat([fuse16, fm384_raw], dim=1))))
        fuse8 = self.smooth8(fm192 + F.interpolate(fuse16, size=fm192.shape[-2:], mode='bilinear', align_corners=False))
        fuse8 = self.relu(self.norm96(self.gate8(torch.cat([fuse8, fm192_raw], dim=1))))
        fuse4 = self.smooth4(fm96 + F.interpolate(fuse8, size=fm96.shape[-2:], mode='bilinear', align_corners=False))
        fuse4 = self.gate4(torch.cat([fuse4, fm96_raw], dim=1))

        # score: dim -> num_classes
        score32 = F.interpolate(self.score32(fuse32), size=img_res, mode='bilinear', align_corners=False)
        score16 = F.interpolate(self.score16(fuse16), size=img_res, mode='bilinear', align_corners=False)
        score8 = F.interpolate(self.score8(fuse8), size=img_res, mode='bilinear', align_corners=False)
        score4 = F.interpolate(fuse4, size=img_res, mode='bilinear', align_corners=False)
        
        # deep supervision
        return [
            (score4, weights[0]),
            (score8, weights[1]),
            (score16, weights[2]),
            (score32, weights[3])
        ]
        
