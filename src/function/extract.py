import os
import time
import pickle
import pathlib

import argparse

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

import dataset.dataset as dataset
from model.select_model import choose_model
from utils.Configuration import CFG
from visualization.VisualizeHelper import plot_uncertainty, plot_uncertainty_mcdropout


def main():

    activations_df = []
    _activations, _inputs  = {}, {}
    def get_activation(name):
        def hook(model, _input, output):
            _activations[name] = output.cpu().detach().numpy()
            _inputs[name] = _input[0].cpu().detach().numpy()
        return hook 


    #N = [str(i) for i in range(8,23)
    N = ['8']
    print(N)
    for k in N:
        parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        #parser.add_argument('--datalist', type=str, default='8', help="Datalist Number 8 to 19")
        parser.add_argument('--num_sampling', type=int, default=50, help="Number of Sampling")
        parser.add_argument('--models', type=str, default='VisionTransformer_Base16', help="Model name")
        parser.add_argument('--dir', type=str, default='20200910_results', help="Directory name")
        
        opt = parser.parse_args()

        since = time.time()
        #print(f'datalist number={i}')
        print(f"datalist={k}")
        print(f'num_sampling={opt.num_sampling}')
        print(f'model name={opt.models}')
        print(f'dir={opt.dir}')

        #CFG.set_seed(CFG.seed)
        data_df = pd.read_csv(CFG.df_path)
        #print(data_df)

        file_names = []

        p = pathlib.Path(f'../datalist{k}/k{CFG.n_fold}').glob('test*.txt')
        for i in p:
            file_names.append(f'k{CFG.n_fold}/'+i.name)


        name = []
        for j in range(len(file_names)):
            for i in range(CFG.n_fold):
                if str(i) in file_names[j]:
                    name.append(os.path.join(CFG.fold_path+f"datalist{k}/", file_names[j]))
        print(name)


        model_file = []
        p_w = pathlib.Path(f'{CFG.model_path}/{opt.models}/').glob(f'0908_train{int(k)-7}7*.pth')
        for i in p_w:
            model_file.append(f'{CFG.model_path}/{opt.models}/{i.name}')
        print(model_file)


        total_predicts = []
        predicts = [[] for _ in range(opt.num_sampling)]
    
        for fold,i in enumerate(model_file): #fold毎の.pthのループ

            model = choose_model(opt.models)
            model.load_state_dict(torch.load(i))
            #print(list(model.children())[-2].ln)
            print(f'model file name : {i}', )


            for j in range(opt.num_sampling): #各foldをnum_inferenceの回数ループ(MCdropout)


                with open(name[fold]) as f:
                    line = f.read().splitlines()

                valid_df = data_df[data_df['UID'].isin(line)]

                valid_dataset = TestDataset(valid_df,
                                            transform=dataset.get_transforms('valid'))                                                           

                valid_loader = DataLoader(valid_dataset,
                                          batch_size=1, 
                                          num_workers=CFG.num_workers,
                                          shuffle=False,
                                          pin_memory=True)

                print(f'fold {fold+1}, iter {j+1}')
                print(line)#foldに含まれるK番号の一覧
                print(len(line))

                h = list(model.children())[-2].ln.register_forward_hook(hook=get_activation('ln')) #for VisionTransformer_Base16
                #h = model.classifier[3].register_forward_hook(hook=get_activation('Linear')) #for VGG16
                #h = list(model.children())[-2].ln.register_forward_hook(hook=get_activation('ln')) #for DenseNet161

                #print(model)
                # print(_activations)
                # print(_inputs)

                #inferenece
                model.train()# turn ON/OFF dropout
                #model.eval()

                activations_per_image = {}
                predicts_per_image = {}   
                labels_per_image = {}

                with torch.no_grad():
                        
                    pbar = tqdm(enumerate(valid_loader, start=1), total=len(valid_loader), desc='Valid ')
                    for step, (inputs, labels, image_path) in pbar:                              
                        inputs  = inputs.to(CFG.device)
                        labels  = labels.to(CFG.device)
                        #print(image_path)

                        outputs  = model.forward(inputs)
                        #outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        # print(f'inputs shape {inputs.shape}')
                        # print(f'outputs shape {outputs.shape}')
                        # print(f'preds shape {preds.shape}')
                        

                        # print(_inputs['ln'].shape)
                        #print(_inputs[0].shape)
                        # print(_inputs['ln'][0].shape)
                        # print(_inputs['ln'])
                        #print(inputs)

                        #print(_inputs['ln'][:,0])
                        #print(_inputs['ln'][:,0].shape)
                        
         
                        activations_per_image[os.path.basename(image_path[0])] = _inputs['ln'][:,0][0]
                        predicts_per_image[os.path.basename(image_path[0])] = preds.cpu().item()
                        labels_per_image[os.path.basename(image_path[0])] = labels.cpu().item()
                        # print(_activations['ln'].shape)
                        # print(_activations['ln'][0].shape)
                        # print(_activations['ln'])
                        # print(_activations)
                        #print(outputs)
                        #print(preds)
                        print(predicts_per_image)
                        print(labels_per_image)
                        #print('######################')

                        #print(activations_per_image['/win/salmon/user/masuda/project/vit_kl_crowe/20220511_DRR_with_Crowe_KL/DRR_AP/K1917_right.png'].shape)
                        #print(activations_per_image)
                        # n = nn.Softmax(dim=1)
                        # outputs = n(outputs)
                        
                        #print(outputs)
                        predicts[j].extend(outputs.tolist())

                        torch.cuda.empty_cache()

                        if j == opt.num_sampling-1:
                            CFG.true.extend(CFG.to_numpy(labels).numpy())
                            CFG.path_list.extend(image_path)
                            CFG.fold_id.extend([fold+1]*len(labels))
            h.remove()

            act_df = pd.DataFrame(activations_per_image).transpose()
            print(act_df)
            #act_df['fold'] = fold
            act_df['predict'] = pd.Series(predicts_per_image)
            act_df['label'] = pd.Series(labels_per_image)
            activations_df.append(act_df)

        

                #iteration = j
            #print(predicts)

        activations_df = pd.concat(activations_df,axis=0)
        activations_df.columns = ['act%04d' % x for x in range(1,len(_inputs['ln'][:,0][0])+1)] + ['predict'] + ['labels']
        activations_df.to_csv('activations.csv')
        
        total = np.array(predicts)
        total

        # print(CFG.path_list)
        # print(CFG.true)
        # print(CFG.fold_id)
        # print(len(CFG.path_list))
        # print(len(CFG.true))
        # print(len(CFG.fold_id))

        pre_avg = np.mean(total,axis=0)
        pre_var = np.var(total,axis=0)
        print(f'pre_avg {pre_avg}')
        # print(pre_avg.shape)
        print(f'pre_var {pre_var}')
        # print(pre_var.shape)

        #一旦処理としてここで切って、平均から最大値の推定値とラベルを求める
        
        # with open(f'{CFG.results_path}/{opt.models}/mean/datalist{i}_avg_{CFG.num_classes}class_{CFG.epochs}epoch_iter{opt.num_sampling}.txt','wb') as f:
        #     pickle.dump(pre_avg,f)
        # with open(f'{CFG.results_path}/{opt.models}/variance/datalist{i}_var_{CFG.num_classes}class_{CFG.epochs}epoch_iter{opt.num_sampling}.txt','wb') as f:
        #     pickle.dump(pre_var,f)

        print(type(CFG.true))
        for i in range(pre_avg.shape[0]):
            pred = np.argmax(pre_avg[i])
            CFG.var.append(pre_var[i][pred])
            CFG.mean.append(pre_avg[i][pred])
            #print(CFG.true[i])
            CFG.difference.append(pred - CFG.true[i])
            CFG.total_predict.append(pred)

        #print(CFG.total_predict)
        #print(CFG.true)
        #print(CFG.difference)
        #print(len(CFG.difference))
        #print(len(CFG.probability),len(CFG.difference),len(CFG.path_list),len(CFG.fold_id))  
        #                    actual_label = CFG.labels_dict[actual.item()]
        #print(CFG.labels_dict[CFG.total_predict.item()])                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          +1

        # plot_uncertainty_mcdropout(CFG.var,
        #                             CFG.difference,
        #                             CFG.path_list,
        #                             CFG.fold_id,
        #                             opt.models,
        #                             opt.num_sampling,
        #                             opt.dir,
        #                             CFG.mean,
        #                             CFG.total_predict,
        #                             CFG.true,
        #                             k)

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