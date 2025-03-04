from collections import OrderedDict
from os.path import join
import pdb

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.model_utils import *
from torch.nn import ReLU, ELU


##########################
#### Genomic FC Model ####
##########################
class SNN(nn.Module):
    def __init__(self, omic_input_dim: int, model_size_omic: str='small', n_classes: int=4):
        super(SNN, self).__init__()
        self.n_classes = n_classes
        self.size_dict_omic = {'small': [256, 256, 256, 256], 'big': [1024, 1024, 1024, 256]}
        
        ### Constructing Genomic SNN
        hidden = self.size_dict_omic[model_size_omic]
        fc_omic = [SNN_Block(dim1=omic_input_dim, dim2=hidden[0])]
        for i, _ in enumerate(hidden[1:]):
            fc_omic.append(SNN_Block(dim1=hidden[i], dim2=hidden[i+1], dropout=0.25))
        self.fc_omic = nn.Sequential(*fc_omic)
        self.classifier = nn.Linear(hidden[-1], n_classes)
        init_max_weights(self)


    def forward(self, **kwargs):
        x = kwargs['x_omic'].view(1,-1)
        features = self.fc_omic(x)

        logits = self.classifier(features)
        Y_hat = torch.topk(logits, 1, dim=1)[1]
        hazards = torch.sigmoid(logits)
        S = torch.cumprod(1 - hazards, dim=1)
        return {
            "hazards":hazards,
            "fusion":None,
            "S":S,
            "Y_hat":Y_hat,
        }

        

    def relocate(self):
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

            if torch.cuda.device_count() > 1:
                device_ids = list(range(torch.cuda.device_count()))
                self.fc_omic = nn.DataParallel(self.fc_omic, device_ids=device_ids).to('cuda:0')
            else:
                self.fc_omic = self.fc_omic.to(device)


            self.classifier = self.classifier.to(device)


class MLP(nn.Module):
    def __init__(
        self, 
        omic_input_dim,
        n_classes=4, 
        projection_dim = 512, 
        dropout = 0.1, 
        ):
        super(MLP, self).__init__()
        
        # self
        self.projection_dim = projection_dim

        self.net = nn.Sequential(
            nn.Linear(omic_input_dim, projection_dim//2), ReLU(), nn.Dropout(dropout),
            nn.Linear(projection_dim//2, projection_dim//2), ReLU(), nn.Dropout(dropout)
        ) 

        self.to_logits = nn.Sequential(
                nn.Linear(projection_dim//2, n_classes)
            )

    def forward(self, **kwargs):
        self.cuda()

        #---> unpack
        data_omics = kwargs['x_omic'].view(1,-1)
        
        #---> project omics data to projection_dim/2
        data = self.net(data_omics) #[B, n]

        #---->predict
        logits = self.to_logits(data) #[B, n_classes]
        Y_hat = torch.topk(logits, 1, dim=1)[1]
        hazards = torch.sigmoid(logits)
        S = torch.cumprod(1 - hazards, dim=1)
        return {
            "hazards":hazards,
            "fusion":None,
            "S":S,
            "Y_hat":Y_hat,
        }

