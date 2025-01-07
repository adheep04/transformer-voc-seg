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


class SwinSeg(nn.Module):
    def __init__(self, n_class):        
        super().__init__()

        # pretrained base net extractor swin-v2-t 
        base_model = swin_v2_t(weights=Swin_V2_T_Weights.DEFAULT)
        self.base = base_model.features
        self.permute = base_model.permute
        self.norm = base_model.norm
        
        # returns prediction scores for intermediate layers and final
        self.s8_score = nn.Conv2d(192, n_class, 1, 1)
        self.s16_score = nn.Conv2d(384, n_class, 1, 1)
        self.out_score = nn.Conv2d(768, n_class, 1, 1)
        
        # upsampling x2 learned initialized with bilinear interpolation weights shown below
        self.upsample_a = nn.ConvTranspose2d(n_class, n_class, 4, 2, bias=False)
        self.upsample_b = nn.ConvTranspose2d(n_class, n_class, 4, 2, bias=False)
       
        # initialize weights
        self._init_weights(n_class)
        
    
    def _init_weights(self, n_class):
        '''
        initializes bilinear interpolation weights for learned upsample layers
        
        '''
  
        # initialize deconvolution weights like that of bilinear interpolation and have it learnable
        # https://github.com/tnarihi/caffe/commit/4f249a00a29432e0bb6723087ec64187e1506f0f <- used this code to produce the following initialization
        bilinear_interp_weights = torch.tensor(
            [[[[0.0625, 0.1875, 0.1875, 0.0625],
            [0.1875, 0.5625, 0.5625, 0.1875],
            [0.1875, 0.5625, 0.5625, 0.1875],
            [0.0625, 0.1875, 0.1875, 0.0625]]]]
        ).repeat((n_class, n_class, 1, 1))
        
        # duplicate along in_channel and out_channel dimension for size (1, 1, 4, 4) -> (n_class, n_class, 4, 4)
        self.upsample_a.weight.data = bilinear_interp_weights
        self.upsample_b.weight.data = bilinear_interp_weights
        
        nn.init.xavier_uniform_(self.s8_score.weight)
        nn.init.xavier_uniform_(self.s16_score.weight)
        nn.init.xavier_uniform_(self.out_score.weight)
        nn.init.xavier_uniform_(self.local_enhance[0].weight)
        nn.init.xavier_uniform_(self.local_enhance[3].weight)
        
    def forward(self, x):
        '''
        args:
        - x: tensor(batch_size, channel_size, height, width)
        
        output:
        - tensor(batch_size, class_size, height, width)
        
        '''
        # get img spatial dimensions
        input_res = (x.shape[-2], x.shape[-1])
        
        ''' forward pass through most of base net swin'''
        
        x = self.base[0](x)
        # down x4
        x = self.base[1](x)
        # down x2
        x = self.base[2](x)
        skip8 = x = self.base[3](x)
        # down x2
        x = self.base[4](x)
        skip16 = x = self.base[5](x)
        x = self.base[6](x)
        # down x2
        x = self.base[7](x)
        x = self.norm(x)
        x = self.permute(x)
        x = self.out_score(x)
        # up x16
        out32 = F.interpolate(input=x, size=input_res, mode='bilinear')
        
        skip16 = self.permute(skip16)
        skip16 = self.s16_score(skip16)
        x = self.upsample_a(x)
        # up x2
        skip16 = self._expand(skip16, x)
        x = x + skip16
        # up x8
        out16 = F.interpolate(input=x, size=input_res, mode='bilinear')
        
        skip8 = self.permute(skip8)
        skip8 = self.s8_score(skip8)
        x = self.upsample_b(x)
        # up x2
        skip8 = self._expand(skip8, x)
        x = x + skip8
        # up x4
        out8 = F.interpolate(input=x, size=input_res, mode='bilinear')
        
        return (out32 + out16 + out8) / 3
        
    def _expand(self, small, big):
        '''
        expands the slightly smaller tensor to be the size of the bigger
        using bilinear interpolation (big and small are very close in size)
        
        '''
        assert big.shape[2] > small.shape[2] and big.shape[3] > small.shape[3], "big not bigger than small"
    
        return F.interpolate(small, size=(big.shape[2], big.shape[3]), mode='bilinear', align_corners=False)
    



transform = T.Compose([
    # transforms.Resize((224, 224)),  
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])  
])

def resize(img, max_pixels=1000*1000):
    # max is 1024*1024 pixels
    
    # get original image dimensions
    w, h = img.size
    # calculate scale ratio between original image necessary to hit max pixel size
    ratio = math.sqrt(max_pixels / (w * h))
    new_w = int(w * ratio)
    new_h = int(h * ratio)
    # resize height and width by proportional values to maintain aspect ratio
    return T.Resize((new_h, new_w))(img)


img_path = Path(r'data\donker_bergen_sneeuw_5000x5000-wallpaper-5120x2880.jpg')
img_path.exists()
img = Image.open(img_path)
img_r = resize(img)
img_tensor = transform(img_r).unsqueeze(0)

swinseg = SwinSeg(30)
swingseg = swinseg.to(torch.device('cuda'))
img_tensor = img_tensor.to(torch.device('cuda'))
print(img_tensor.shape)
pred = swinseg(img_tensor)
print(pred[2].shape)