import os
import random

import torch
import numpy as np

from utils.Parser import get_args

class CFG:
    opt = get_args()
    
    # base_path = '/win/salmon/user/masuda/project/crowe_kl_regression'
    # dataset_path2 = '/win/salmon/user/masuda/project/makedrr/75_Normal-DRR_944_masuda'
    # dataset_path = '/win/salmon/user/masuda/project/vit_kl_crowe/20220511_DRR_with_Crowe_KL'
    # weights_path = f'/win/salmon/user/masuda/project/crowe_kl_regression/report/{opt.sign}/weights'
    # datalist_path = f'/win/salmon/user/masuda/project'
    # #image_path = dataset_path + "/DRR_AP"
    # image_path = dataset_path2 + "/DRR_AP"
    # csv_path = dataset_path + "/20220511_OsakaHip_TwoSide_KL_Crowe.csv"
    # df_path = dataset_path + '/20220919_data_df.csv'
    # results_path = f'/win/salmon/user/masuda/project/crowe_kl_regression/report/{opt.sign}'

    key = '6a2084b93da3087a45bcccf07f48f2bffa7b2b0f'

    scheduler = 'CosineAnnealinglr'

    difference = []
    probability = []
    path_list = []
    fold_id = []
    true = []
    var = []
    mean = []
    total_predict = []
    total_correct = []

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    if opt.num_classes == 1:
        labels_dict = {0:'Crowe=1,KL=1',
                       1:'Crowe=1,KL=2',
                       2:'Crowe=1,KL=3',
                       3:'Crowe=1,KL=4',
                       4:'Crowe=2,KL=4',
                       5:'Crowe=3,KL=4',
                       6:'Crowe=4,KL=4'}

        labels = [0,1,2,3,4,5,6]

        labels_name = ['1,1',
                       '1,2',
                       '1,3',
                       '1,4',
                       '2,4',
                       '3,4',
                       '4,4']

        labels_index = ['Crowe=1,KL=1',
                        'Crowe=1,KL=2',
                        'Crowe=1,KL=3',
                        'Crowe=1,KL=4',
                        'Crowe=2,KL=4',
                        'Crowe=3,KL=4',
                        'Crowe=4,KL=4']
    
    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu()

    def set_seed(seed = opt.seed):
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        # When running on the CuDNN backend, two further options must be set
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Set a fixed value for the hash seed
        os.environ['PYTHONHASHSEED'] = str(seed)
        print('> SEEDING DONE')
