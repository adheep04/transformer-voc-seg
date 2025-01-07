from pathlib import Path
from PIL import Image
import numpy as np
import math
from torchvision import transforms
import torch
from model_old import SwinSeg

transform = transforms.Compose([
    # transforms.Resize((224, 224)),  
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
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
    return transforms.Resize((new_h, new_w))(img)


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