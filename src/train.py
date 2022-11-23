import copy
import csv
import gc
from glob import glob
import os
import pathlib
import time

import pandas as pd
from tqdm import tqdm
tqdm.pandas()
import torch
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

import torch
import torch.nn as nn
from utils.Parser import get_args
from utils.Configuration import CFG
from visualization.VisualizeHelper import visualize_total_image
from dataset.dataset import TrainDataset,TestDataset,get_transforms
from model.select_model import choose_model
from evaluation.EvaluationHelper import EvaluationHelper
from function.compare_acc import compare, compare2
from trainval_one_epoch import train_one_epoch, valid_one_epoch
#from gradCAM.cam2 import visualize_grad_cam

import wandb


def main():

    #preparation training
    opt = get_args()
    CFG.set_seed(opt.seed)

    if torch.cuda.is_available():
        print(f"cuda: {torch.cuda.get_device_name}")
    
    data_df = pd.read_csv(opt.df_path)
    #manage filename
    file_names = []

    p = pathlib.Path(f'../datalist{opt.datalist}/k{opt.fold}').glob('train*.txt')
    for i in p:
        file_names.append(f'k{opt.fold}/'+i.name)

    p = pathlib.Path(f'../datalist{opt.datalist}/k{opt.fold}').glob('valid*.txt')
    for i in p:
        file_names.append(f'k{opt.fold}/'+i.name)
    print(file_names)

    name = []
    for j in range(len(file_names)):
        for i in range(int(opt.fold)):
            if str(i) in file_names[j]:
                name.append(os.path.join(opt.datalist_path+f'/datalist{opt.datalist}', file_names[j]))
    print(name)

    print('image_path2', opt.image_path)

    #run training each fold
    for fold in [i for i in range(int(opt.fold))]:
        print(f'#'*15)
        print(f'### Fold: {fold+1}')
        print(f'#'*15)

        #prepare dataframe for each fold
        with open(name[fold]) as f:
            line = f.read().splitlines()
        with open(name[fold+int(opt.fold)]) as f:
            line2 = f.read().splitlines()
        print(line,line2)

        train_df = data_df[data_df['UID'].isin(line)]
        valid_df = data_df[data_df['UID'].isin(line2)]
        

        train_dataset = TrainDataset(train_df,
                                     transform = get_transforms('train'))
        valid_dataset = TestDataset(valid_df,
                                    transform = get_transforms('valid'))

        train_loader = DataLoader(train_dataset,
                                  batch_size = opt.batch_size, 
                                  num_workers = opt.num_workers,
                                  shuffle = True,
                                  pin_memory = True,
                                  drop_last = False)
        valid_loader = DataLoader(valid_dataset,
                                  batch_size = opt.valid_batch_size, 
                                  num_workers = opt.num_workers,
                                  shuffle = False,
                                  pin_memory = True)


        #create model
        model = choose_model(opt.model)
        #print(model)

        #Metrics
        #criterion = nn.MSELoss()
        criterion = nn.L1Loss()
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr = opt.lr,
                                     weight_decay = opt.wd)
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer,
                                                   T_max = opt.t_max,
                                                   eta_min = opt.min_lr)


        #wandb
        wandb.login(key=CFG.key)
        run = wandb.init(project = f'{opt.sign}{opt.num_classes}class_crowe_kl-regression', 
                         config = {"model_name": opt.model,
                                   "learning_rate": opt.lr,
                                   "fold": opt.fold,
                                   "epochs": opt.epoch,
                                   "image_size": opt.image_size,
                                   "batch_size": opt.batch_size,
                                   "num_workers": opt.num_workers,
                                   "num_classes": opt.num_classes,
                                   "optimizer": opt.optimizer,
                                   "loss": opt.criterion,
                                   "sign": opt.sign},
                         entity = "xxmrkn",
                         name = f"{opt.sign}|{opt.num_classes}class|{opt.model}|{opt.fold}fold"
                                f"|fold-{fold+1}|dim-{opt.image_size}**2|batch-{opt.batch_size}|lr-{opt.lr}")

        wandb.watch(model, log_freq=100)        


        #Training 
        since = time.time()
        best_model_wts = copy.deepcopy(model.state_dict())
        best_loss = 100

        for epoch in range(opt.epoch):
            gc.collect()
            print(f'Epoch {epoch+1}/{opt.epoch}')
            print('-' * 10)

            train_loss = train_one_epoch(model,
                                        optimizer,
                                        scheduler,
                                        criterion,
                                        train_loader)

            valid_loss = valid_one_epoch(model,
                                        optimizer,
                                        criterion,
                                        valid_loader)

            wandb.log({"Train Loss": train_loss, 
                       "Valid Loss": valid_loss,
                       "LR": scheduler.get_last_lr()[0]})

            #display results
            print('#'*50)
            print(f'######## now training...  fold : {fold+1}')
            print('#'*50)

            print(f"Train Loss: {train_loss}")
            print(f"Valid Loss: {valid_loss}")

            print('#'*50)

            if valid_loss < best_loss:
                best_loss = valid_loss
                #print(f'Loss {best_loss} --> {valid_loss}')
                tgt = f'{opt.result_path}/{opt.sign}/weights/{opt.model}'
                os.makedirs(tgt, exist_ok=True)

                torch.save(model.state_dict(),
                        tgt + f'/{opt.sign}_datalist{opt.datalist}_'
                        f'fold{opt.fold}{fold+1}_{opt.epoch}epoch_weights.pth')
                print('Save Model !')
        run.finish()

        #display wandb webpage link
        print(f"wandb website ------> {run.url}")

        #remove wandb files
        print(os.path.isdir('wandb'))
        #shutil.rmtree("wandb")

        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed//3600}h {time_elapsed//60}m {time_elapsed%60:.2f}s')

    return model


if __name__ == '__main__':
    main()
