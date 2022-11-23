import pickle
import argparse

import seaborn as sns
import matplotlib.pyplot as plt

import pandas as pd

from Configuration import CFG

def main():
    
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--sampling', type=int, default=1, help="Number of sampling")
    #parser.add_argument('--dir', type=str, default='new_mcdropout', help="Choose directory")
    parser.add_argument('--mode', type=str, default='Variance', help="Choose mode Probability or Variance")
    
    opt = parser.parse_args()
    print(f'sampling = {opt.sampling}')
    #print(f'dir = {opt.dir}')
    print(f'mode = {opt.mode}')
    
    total_uncertainty = pd.read_csv(f'/win/salmon/user/masuda/project/vit_kl_crowe/utils/new_datalist8_7class_uncertainty_boxplot_200epoch_iter50.csv')

    print(total_uncertainty)
    
    # for i in ['Exact','1 Neighbor','Others']:
    #     if i =='Correct':
    #         num_1 = len(total_uncertainty[total_uncertainty.Uncertainty==i])
    #     elif i =='1 Neighbor':
    #         num_2 = len(total_uncertainty[total_uncertainty.Uncertainty==i])                                   
    #     else:
    #         num_3 = len(total_uncertainty[total_uncertainty.Uncertainty==i])
    sns.set(font_scale=1.5)
    sns.set_style('whitegrid')
    
    if opt.mode == 'Probability':
        plt.figure(figsize=(40, 40))
        g = sns.catplot(x="Model",
                        y="Probability",
                        #y="Variance",
                        hue="Uncertainty",
                        data=total_uncertainty,
                        kind="box",
                        legend_out=False,
                        #palette=sns.color_palette("Set2"),
                    )
        g.set(ylim=(0.0, 1.0))

    elif opt.mode == 'Variance':
        plt.figure(figsize=(40, 40))
        g = sns.catplot(x="Model",
                        #y="Probability",
                        y="Variance",
                        hue="Uncertainty",
                        data=total_uncertainty,
                        kind="box",
                        legend_out=False,
                        #palette=sns.color_palette("Set2"),
                    )
        g.set(ylim=(0.0, 0.02))

    else:
        raise('No Exist !')
    g.fig.set_figheight(10)
    g.fig.set_figwidth(8)

    
    #new_labels = [f"Correct (n={num_1})",f"1 Neighbor (n={num_2})",f"Others (n={num_3})"]
    #new_labels = [f"Correct (p<0.05)",f"1 Neighbor (p<0.05)",f"Others (p<0.05)"]
    new_labels = [f"Exact",f"1 Neighbor",f"Others"]
    
    for t, l in zip(g._legend.texts, new_labels):
        t.set_text(l)

    #g.savefig(f'{CFG.results_path}/{opt.dir}/total/{CFG.sign}_{opt.mode}_totalnormal{opt.sampling}sampling.png')
    #g.savefig(f'mi_fold15pattern_iter50.png')
    g.savefig(f'mi_slide_fold2.png')


if __name__ == '__main__':
    main()