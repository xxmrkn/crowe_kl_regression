import gc
from tqdm import tqdm
tqdm.pandas()

import torch
import torch.nn as nn

from utils.Parser import get_args
from utils.Configuration import CFG
#from evaluation.EvaluationHelper import EvaluationHelper

def train_one_epoch(model,
                    optimizer,
                    scheduler,
                    criterion,
                    dataloader):

    opt = get_args()

    model.train()
    grad_scaler = torch.cuda.amp.GradScaler()
    
    running_loss = 0.0
    
    pbar = tqdm(enumerate(dataloader, start=1),
                total=len(dataloader),
                desc='Train')
    for step, (inputs, labels) in pbar:         
        inputs = inputs.to(CFG.device)
        labels  = labels.to(CFG.device)
        labels = labels.unsqueeze(1)

        optimizer.zero_grad()
        
        with torch.cuda.amp.autocast(enabled=True):
            outputs = model(inputs)
            loss = criterion(outputs.to(torch.float32), labels.to(torch.float32))

            running_loss += loss.item()
            train_loss = running_loss / step

        #optimizer.zero_grad()
        grad_scaler.scale(loss).backward()
        grad_scaler.step(optimizer)
        grad_scaler.update()

        scheduler.step()

        mem = torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0
        current_lr = optimizer.param_groups[0]['lr']
        pbar.set_postfix(train_loss=f'{train_loss:0.6f}',
                         lr=f'{current_lr:0.6f}',
                         gpu_mem=f'{mem:0.2f} GB')
        torch.cuda.empty_cache()
            
        gc.collect()
    
    return train_loss


def valid_one_epoch(model,
                    optimizer,
                    criterion,
                    dataloader):

    opt = get_args()

    #model.train()# turn ON dropout
    model.eval()# turn OFF dropout

    with torch.no_grad():
        
        dataset_size = 0
        running_loss = 0.0
        
        pbar = tqdm(enumerate(dataloader, start=1),
                    total=len(dataloader),
                    desc='Valid')
        for step, (inputs, labels, image_path, image_id) in pbar:        
            inputs  = inputs.to(CFG.device)
            labels  = labels.to(CFG.device)
            labels = labels.unsqueeze(1)

            batch_size = inputs.size(0)

            outputs = model(inputs)

            loss = criterion(outputs.to(torch.float32), labels.to(torch.float32))
            
            running_loss += loss.item()
            dataset_size += batch_size

            valid_loss = running_loss / step

            mem = torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0
            current_lr = optimizer.param_groups[0]['lr']
            pbar.set_postfix(valid_loss=f'{valid_loss:0.6f}',
                             lr=f'{current_lr:0.6f}',
                             gpu_memory=f'{mem:0.2f} GB')
            torch.cuda.empty_cache()
            
            gc.collect()
        
    return valid_loss