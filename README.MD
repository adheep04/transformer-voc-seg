
training for swin-based fpn segmentation model: experienced irrecoverable loss at 2k steps. Pixel accuracy: 0.75, miou: 0.25

![alt text](ss/loss_spike.png)

![alt text](ss/image.png)

## update

tried using boundary refinement module (sobel kernel) to
improve IoU score (0.66 on pascal voc 2012) but i think it may have 
just added noise:

with boundary-module (purple), without (blue)

![alt text](image.png)