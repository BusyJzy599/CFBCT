from __future__ import print_function, division
import math
import os
import pdb
import pickle
import re
import hashlib
import h5py
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler

import torch
from torch.utils.data import Dataset

from utils.utils import generate_split, nth,string_hash
from .dataset_survival_tcga import Generic_WSI_Survival_Dataset


class Generic_MIL_Biomarker_Dataset(Generic_WSI_Survival_Dataset):
    def __init__(self, data_dir, mode: str='omic', **kwargs):
        super(Generic_MIL_Biomarker_Dataset, self).__init__(**kwargs)
        self.data_dir = data_dir
        self.mode = mode
        self.use_h5 = False


    def load_from_h5(self, toggle):
        self.use_h5 = toggle

    def __getitem__(self, idx):
        case_id = self.slide_data['case_id'][idx]
        label = self.slide_data['disc_label'][idx]
        event_time = self.slide_data[self.label_col][idx]
        c = self.slide_data['censorship'][idx]
        slide_ids = self.patient_dict[case_id]
        omic_len=len(self.omic_names)

        if type(self.data_dir) == dict:
            source = self.slide_data['oncotree_code'][idx]
            data_dir = self.data_dir[source]
        else:
            data_dir = self.data_dir
        
        if not self.use_h5:
            if self.data_dir:
                if self.mode == 'path':
                    path_features = []
                
                    path_features = []
                    with h5py.File(os.path.join(data_dir,'path_h5', f"{case_id}.h5"), 'r') as hf:
                        path_features = torch.FloatTensor(hf['data'][:])
                    return (path_features, torch.zeros((1,1)), label, event_time, c)
                elif self.mode == 'omic':
                    genomic_features = torch.tensor(self.genomic_features.iloc[idx])
                    return (torch.zeros((1,1)), genomic_features, label, event_time, c)
                elif self.mode == 'coattn':
                    path_features = []
                    with h5py.File(os.path.join(data_dir,'path_h5', f"{case_id}.h5"), 'r') as hf:
                        path_features = torch.FloatTensor(hf['data'][:])
                    
                    omics=[]
                    for o in range(omic_len):
                        omics.append(torch.tensor(self.genomic_features[self.omic_names[o]].iloc[idx]))
                    return (path_features,omics, label, event_time, c)
                        
                        
            
                else:
                    raise NotImplementedError('Mode [%s] not implemented.' % self.mode)
                ### <--
            else:
                return slide_ids, label, event_time, c
