import torch
from torch.utils.data import Dataset
from pathlib import Path
import math
from torchvision import transforms
from PIL import Image
import numpy as np

CLASSES=('background', 'aeroplane', 'bicycle', 'bird', 'boat',
                 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
                 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep',
                 'sofa', 'train', 'tvmonitor'),

PALETTE=([0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
            [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
            [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
            [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
            [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
            [0, 64, 128])

IDS=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20)

data = Image.open(r'data\VOCtrainval_11-May-2012\VOCdevkit\VOC2012\JPEGImages\2007_000032.jpg')
label = Image.open(r'data\VOCtrainval_11-May-2012\VOCdevkit\VOC2012\SegmentationClass\2007_000032.png')
label.show()
label_tensor = np.array(label)
count = len(label_tensor[label_tensor == 0])
print(np.unique(label_tensor), count)


def _convert_label(self, label_tensor):
    pass
