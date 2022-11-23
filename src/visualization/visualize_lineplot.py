import pickle
import argparse

import seaborn as sns
import matplotlib.pyplot as plt

import pandas as pd

from Configuration import CFG

def main():
    
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dir', type=str, default='new_mcdropout', help="Choose directory")
    
    opt = parser.parse_args()
    print(f'dir = {opt.dir}')

    # total_uncertainty1 = pd.read_csv(f'{CFG.results_path}/{opt.dir}/total/{CFG.sign}_total_{opt.dir}{CFG.num_classes}class_lineplot1_{CFG.epochs}epoch_iter100_list.csv')
    # total_uncertainty2 = pd.read_csv(f'{CFG.results_path}/{opt.dir}/total/{CFG.sign}_total_{opt.dir}{CFG.num_classes}class_lineplot2_{CFG.epochs}epoch_iter100_list.csv')
    #Y:\masuda\project\vit_kl_crowe\20220821_results\new_mcdropout\total\0823_train_total_new_mcdropout7class_lineplot1_200epoch_iter100_list.csv
    total_uncertainty2 = pd.read_csv(f'Y:\\masuda\\project\\vit_kl_crowe\\20220821_results\\new_mcdropout\\total\\0823_train_total_new_mcdropout7class_lineplot2_200epoch_iter100_list.csv')
    total_uncertainty1 = pd.read_csv(f'Y:\\masuda\\project\\vit_kl_crowe\\20220821_results\\new_mcdropout\\total\\0823_train_total_new_mcdropout7class_lineplot1_200epoch_iter100_list.csv')

    print(total_uncertainty1)
    #print(total_uncertainty2)

    plt.figure(figsize=(12, 7))
    sns.set_style('whitegrid')
    sns.set_palette("Set1",3)    
    sns.pointplot(x="Number of Sampling",
                    y="Accuracy (Normal / 1Neighbor)",
                    #y="Variance",
                    hue="Model",
                    data=total_uncertainty1)
                    #palette=sns.color_palette("Set2"),)
    sns.set_palette("Set1",3)
    sns.pointplot(x="Number of Sampling",
                    y="Accuracy (Normal / 1Neighbor)",
                    #y="Variance",
                    hue="Model",
                    data=total_uncertainty2,
                    linestyles='--',
                    markers=',')

    # sns.fig.set_figheight(8)
    # sns.fig.set_figwidth(13)

    #sns.set(ylim=(0.0, 1.0))
    
    #new_labels = [f"Correct (n={num_1})",f"1 Neighbor (n={num_2})",f"Others (n={num_3})"]
    #new_labels = ["ViT_B16","VGG16","DenseNet161"]
    #               "ViT_B16(1Neighbor)","VGG16(1Neighbor)","DenseNet161(1Neighbor)"]

    #plt.savefig(f'{CFG.results_path}/{opt.dir}/total/{CFG.sign}_total_{opt.dir}_lineplot.png')
    plt.savefig(f'0823_train_total_new_mcdropout_lineplot.png')

if __name__ == '__main__':
    main()