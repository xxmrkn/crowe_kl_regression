import argparse
import os
import pickle
import pathlib
import time
import itertools

import numpy as np
import pandas as pd
#import seaborn as sns
import matplotlib.pyplot as plt

import torch 
import torch.nn as nn
import torch.optim as optim
from torch.cuda import amp
from torchvision import models
from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm
from dataset.dataset import TestDataset
tqdm.pandas()

from utils.Parser import get_args
import dataset.dataset as dataset
from model.select_model import choose_model
from utils.Configuration import CFG
from evaluation.EvaluationHelper import EvaluationHelper
from visualization.VisualizeHelper import plot_uncertainty_mcdropout


def main():
    N = [str(i) for i in range(8,23)] #datalist8-datalist22 total=15
    #N = ['8']
    print(N)
    opt = get_args()
    
    for k in N:
        
        since = time.time()
        #CFG.set_seed(CFG.seed)
        data_df = pd.read_csv(opt.df_path)

        file_names = []

        p = pathlib.Path(f'../datalist{k}/k4').glob('test*.txt')
        for i in p:
            file_names.append(f'k4/'+i.name)
        #print(file_names)

        name = []
        for j in range(len(file_names)):
            for i in range(4):
                if str(i) in file_names[j]:
                    name.append(os.path.join(opt.datalist_path+f"/datalist{k}/", file_names[j]))
        #print(name)

        model_file = []
        p_w = pathlib.Path(f'{opt.result_path}/{opt.sign}/weights/{opt.model}/').glob(f'{opt.sign}_datalist{opt.datalist}*.pth')
        for i in p_w:
            model_file.append(f'{opt.result_path}/{opt.sign}/weights/{opt.model}/{i.name}')
        print(model_file)

        total_predicts = []
        predicts = [[] for _ in range(opt.num_sampling)]
        predicts_float = [[] for _ in range(opt.num_sampling)]
        total_confusion_matrix = 0

        for fold,i in enumerate(model_file): #fold毎の.pthのループ

            model = choose_model(opt.model)
            model.load_state_dict(torch.load(i))

            for j in range(opt.num_sampling): #各foldをnum_inferenceの回数ループ(MCdropout)

                with open(name[fold]) as f:
                    line = f.read().splitlines()

                valid_df = data_df[data_df['UID'].isin(line)]

                valid_dataset = TestDataset(valid_df,
                                            transform=dataset.get_transforms('valid'))                                                           

                valid_loader = DataLoader(valid_dataset,
                                          batch_size=opt.batch_size, 
                                          num_workers=opt.num_workers,
                                          shuffle=False,
                                          pin_memory=True)

                print(f'fold {fold+1}, iter {j+1}')
                #print(line)#foldに含まれるK番号の一覧
                #print(len(line))

                #inferenece
                model.train()# turn ON/OFF dropout
                #model.eval()
                with torch.no_grad():
                        
                    pbar = tqdm(enumerate(valid_loader, start=1),
                                total=len(valid_loader), 
                                desc='Valid ',
                                disable=True)
                    for step, (inputs, labels, image_path, image_id) in pbar:                              
                        inputs = inputs.to(CFG.device)
                        labels = labels.tolist()

                        outputs = model(inputs)
                        outputs2 = outputs.tolist()

                        outputs2 = list(itertools.chain.from_iterable(outputs2))
                        outputs = EvaluationHelper.threshold_config(outputs)
                        outputs = list(itertools.chain.from_iterable(outputs))

                        acc_score = EvaluationHelper.acc(outputs,labels)
                        total_confusion_matrix += EvaluationHelper.conf_mtrx(outputs,labels)
                        
                        predicts[j].extend(outputs)
                        predicts_float[j].extend(outputs2)

                        torch.cuda.empty_cache()

                        if j == opt.num_sampling-1:
                            CFG.true.extend(labels)
                            CFG.path_list.extend(image_id)
                            CFG.fold_id.extend([fold+1]*len(labels))
        
        exact_acc, oneneighbor_acc = EvaluationHelper.total_acc(total_confusion_matrix,
                                                                sum(sum(total_confusion_matrix)))
        #print(exact_acc,oneneighbor_acc)
        #print(predicts_float)
        #print(len(CFG.path_list), len(CFG.true), len(CFG.fold_id))
        
        total = np.array(predicts_float)
        #print(f'total {total}')

        pre_avg = np.mean(total,axis=0)
        pre_var = np.var(total,axis=0)
        #print(f'pre_avg {pre_avg} pre_avg shape {len(pre_avg)}')
        #print(f'pre_var {pre_var} pre_var {len(pre_var)}')

        out = EvaluationHelper.threshold_config_for_inf(pre_avg)
        #print(f'out {out}')
        #print(f'CFG.true {CFG.true}')

        difference = np.array(out) - np.array(CFG.true)
        #print('diff',difference)

        plot_uncertainty_mcdropout(pre_var,
                                   difference,
                                   CFG.path_list,
                                   CFG.fold_id,
                                   opt.model,
                                   opt.num_sampling,
                                   pre_avg,
                                   out,
                                   CFG.true,
                                   k)

        time_elapsed = time.time() - since
        print(f'Inference complete in {time_elapsed//3600}h {time_elapsed//60}m {time_elapsed%60:.2f}s')

        CFG.difference = []
        CFG.probability = []
        CFG.path_list = []
        CFG.fold_id = []
        CFG.true = []
        CFG.var = []
        CFG.mean = []
        CFG.total_predict = []
        CFG.total_correct = []

if __name__ == '__main__':
    main()