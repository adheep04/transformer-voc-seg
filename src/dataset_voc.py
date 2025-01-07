import torch
from torch.utils.data import Dataset
from pathlib import Path
import math
from torchvision import transforms
from config import config
from PIL import Image
import numpy as np


class VOC2012(Dataset):
    CLASSES=('background', 'aeroplane', 'bicycle', 'bird', 'boat',
                 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
                 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep',
                 'sofa', 'train', 'tvmonitor')
    PALETTE=([0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                [0, 64, 128])
    IDS=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20)
    
    def __init__(self, path=r'data/VOCtrainval_11-May-2012/VOCdevkit/VOC2012/', cache_size=200, pin_memory=True, train=True):
        self.train = train
        self.pin_memory = pin_memory
        self.img_dir = path + r'JPEGImages/'
        self.label_dir = path + r'SegmentationClass/'
        
        self.train_list = Path(path + r'ImageSets/Segmentation/train.txt').read_text().splitlines()
        self.val_list = Path(path + r'ImageSets/Segmentation/val.txt').read_text().splitlines()
        
        self.process_data = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225],
            ),     
        ])
        self.cache = {}
        self.cache_size = cache_size
    
    def __getitem__(self, index):
        img_name = self.train_list[index] if self.train else self.val_list[index]
        img_path = self.img_dir + f"{img_name}.jpg"
        label_path = self.label_dir + f"{img_name}.png"

        with Image.open(img_path) as data_img:
            data_img.load()
            data_tensor = self.process_data(data_img)

        with Image.open(label_path) as label_img:
            label_img.load() 
            label_tensor = torch.tensor(np.array(label_img)).long()
            
        if self.pin_memory:
            data_tensor = data_tensor.pin_memory()
            label_tensor = label_tensor.pin_memory()
    
        if len(self.cache) >= self.cache_size:
            self.cache.pop(next(iter(self.cache)))
            
        self.cache[index] = (data_tensor, label_tensor)
        
        return data_tensor, label_tensor
    
    def __len__(self):
        return len(self.train_list) if self.train else len(self.val_list)