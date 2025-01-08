import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from lion_pytorch import Lion
import torch.optim.lr_scheduler as schedule

import numpy as np
from math import inf

import time

from model_restore import SwinSeg
from dataset_voc import VOC2012
from config import config


def train(resume=False, resume_file_path=None):
    ''' training setup '''
    
    # set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # initialize model
    swinseg = SwinSeg(n_class=21)
    # bring it to CUDA:) or cpu if u dont have cuda ig
    swinseg.to(device) 
    # swinseg = torch.compile(swinseg)
    
    # tensorboard
    writer = SummaryWriter(log_dir=config.LOG_DIR / f'run_{config.RUN}')    
    
    if resume:
        state_dict = torch.load(resume_file_path)
        swinseg.load_state_dict(state_dict)
    
    # get datasets
    train_data = VOC2012(train=True)
    val_data = VOC2012(train=False)
    
    # initialize dataloaders
    train_loader = DataLoader(
        train_data,
        batch_size=1,
        num_workers=4,  
        pin_memory=True,
        prefetch_factor=2, 
        persistent_workers=True 
    )

    val_loader = DataLoader(
        dataset = val_data,
        batch_size = 1,
        shuffle = False
    )
    
    
    loss_fn = nn.CrossEntropyLoss(ignore_index=255)
    
    log_step = 20
    global_step = 0  
    
    ''' use 2 x learning_rate for biases (like authors)'''
    
    # initialize lists to hold model parameters
    best_miou = 0
    best_loss = float(inf)
    best_acc = 0
    
    # initialize optimizer 
    optimizer = Lion(
        swinseg.parameters(),
        lr=8e-6, 
        weight_decay=3e-4
    )
    
    scheduler = schedule.CosineAnnealingWarmRestarts(optimizer, T_0=3, T_mult=2, eta_min=1e-6)
    
    ''' training loop '''
    
    swinseg.train()

    try: 
        for epoch in range(config.NUM_EPOCHS):
            
            # run validation for last epoch
            if epoch != 0:
                # run validation
                val_stats = validation(swinseg, val_loader, epoch=epoch-1)
                
                # log stats for tensorboard
                writer.add_scalar('loss/val', val_stats['mean_loss'], epoch)
                writer.add_scalar('mIoU/val', val_stats['mean_iou'], epoch)
                
                # log per-class IoU
                for class_id, iou in enumerate(val_stats['miou_per_class']):
                    writer.add_scalar(f'IoU/class_{class_id}', iou, epoch)

            print(f'starting epoch {epoch}')
            for step, (data, label) in enumerate(train_loader):
                # send data to device
                data = data.to(device)
                label = label.to(device)
                    
                # forward pass  
                outputs = swinseg(data)
                loss = sum([loss_fn(score, label) * weight for score, weight in outputs])
                output = outputs[0][0]
                    
                # calculate gradients
                loss.backward()      

                # gradient clipping
                torch.nn.utils.clip_grad_norm_(swinseg.parameters(), max_norm=3) 

                # update parameters
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                
                # miou is the mean of the class ious
                miou = np.mean(class_iou(output, label)[0])
                acc = pixel_acc(output, label)
                
                best_loss = min(loss, best_loss)
                best_acc = max(best_acc, acc)
                best_miou = max(best_miou, miou)
                
                # log loss
                if step % log_step == 0:
                    print(f'{loss}, {miou}, {acc}')
                    
                    # log loss and miou 
                    writer.add_scalar('Loss/train', loss.item(), global_step)
                    writer.add_scalar('mIoU/train', miou, global_step)
                    writer.add_scalar('accuracy/train', acc, global_step)
                    
                    # log sample predictions
                    if step % (log_step * 10) == 0:
                        pred = output.softmax(dim=1).argmax(dim=1)
                        writer.add_images('predictions', pred.unsqueeze(1).float(), global_step)
                        writer.add_images('ground truth', label.unsqueeze(1).float(), global_step)
                    
                    if step % (log_step * 20) == 0:
                        torch.save({
                            'epoch': epoch,
                            'model_state_dict': swinseg.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'best_miou': best_miou,
                        }, f'checkpoints/per200/run_{config.RUN}_BA_{best_acc}_.pth')
                
                    
    except KeyboardInterrupt:
        print("training interrupted. Saving checkpoint...")
        writer.close()
        torch.save(swinseg.state_dict(), 
            f=config.CHECKPOINT_DIR / f'interrupted_s_dict_{config.RUN}_{int(time.time())}')
        print(f'epoch: {epoch}, step: {step}')
        return
        
    writer.close() 
    torch.save(swinseg.state_dict(),
        f=config.CHECKPOINT_DIR / f'finished_s_dict_{config.RUN}_{int(time.time())}')
    return "training done!"
           
def validation(
    model, 
    val_dataloader, 
    loss_fn=nn.CrossEntropyLoss(ignore_index=255), 
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    epoch=None):
    '''
    run a single round of validation using loss and mean-intersection/union

    args:
    - they r obvious :D
    
    '''
    
    # initialize metric accumulation variables
    ious = np.zeros(21)
    loss = 0
    iou = 0
    acc = 0
    
    # move model to device and set to evalulation mode
    model = model.to(device)
    model.eval() 
    
    print('starting validation round')
    with torch.no_grad():      # no gradient 
        for step, (data, label) in enumerate(val_dataloader):
            
            # move to device
            data = data.to(device)
            label = label.to(device)
            
            # forward pass  
            output = model(data)[0][0]
            step_loss = loss_fn(output, label)
            
            # get metrics
            step_ious, class_ids = class_iou(output, label)
            step_iou = np.mean(step_ious)
            step_acc = pixel_acc(output, label)
                
            # update running average metrics
            loss += (step_loss - loss) / (step + 1) if step != 0 else step_loss
            iou += (step_iou - iou) / (step + 1) if step != 0 else step_iou
            acc += (step_acc - acc) / (step + 1) if step != 0 else acc
            ious[class_ids] += (step_ious - ious[class_ids]) / (step + 1) if step != 0 else step_ious
            
        stats = {
            'mean_loss' : loss,
            'mean_iou' : iou,
            'miou_per_class' : ious,
            'acc' : acc
        }
        
        torch.save(stats, f=config.CHECKPOINT_DIR / f'val_{config.RUN}_{epoch}')
        print(stats)
        return stats       
                         
def class_iou(model_out, label):
        '''
        metric for evaluating image segmentation tasks by dividing
        the intersection area by the union area of a given object in 
        both label and prediction images (measuring overlapp)
        
        args:
        - model_out: tensor shape (1, n_class, h, w)
        - label: tensor shape (1, 1, h, w)
        
        output:
        - (np.array(n_class), mean_iou float)
            - iou per class
        '''
        
        # gets a set of all class labels in the sample
        label_class_ids = label.unique().tolist()
        
        # convert from predictions for all classes to single prediction per pixel
        # (1, n_class, h, w) -> (1, 1, h, w)
        pred = model_out.softmax(dim=1).argmax(dim=1).to(dtype=torch.uint8)
        
        # get set of prediction classes by model
        pred_class_ids = pred.unique().tolist()
        
        # set of all predictions
        class_ids = set(label_class_ids + pred_class_ids)
        class_ids.discard(255) # remove ignore class
        
        # initialize per class iou score list
        scores = []
        
        # iterate through all types
        for id in class_ids:
            # if both pred and label contain type object
            if id in pred_class_ids and id in label_class_ids:
                
                # get boolean masks that are True where the pixel value == the type
                pred_mask = (pred == id)
                label_mask = (label == id)
                
                # get the boolean mask for the union and intersection of pred and label 
                union = pred_mask | label_mask          # using or operator for union
                intersection = pred_mask & label_mask   # using and operator for intersection
                type_iou = float(torch.sum(intersection))/float(torch.sum(union))
                scores.append(type_iou)
            else:
                # if a type is in label but isn't in pred, it's a false positive
                # if a type is in pred but isn't in label, it's a false negative
                # either case it's a 0
                scores.append(0)
        
        return scores, np.array(list(class_ids), dtype=int)

def pixel_acc(model_out, label):
    pred = model_out.softmax(dim=1).argmax(dim=1).to(dtype=torch.uint8)
    assert pred.shape == label.shape
   
    # Create mask excluding background (0) and ignore_index (255)
    mask = (label != 0) & (label != 255)
   
    # Calculate accuracy only on masked pixels
    accuracy = (pred[mask] == label[mask]).float().mean()
   
    return accuracy.item()


if __name__ == '__main__':
    train()
    
    