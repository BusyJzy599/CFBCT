
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.model_utils import *


################################
### Deep Sets Implementation ###
# Manzil Zaheer, Satwik Kottur, Siamak Ravanbakhsh, Barnabas Poczos, Ruslan Salakhutdinov, and Alexander Smola. Deep sets. Advances in Neural Information Processing Systems, 2017.
################################
class MQP_surv(nn.Module):
    def __init__(self, model_size_wsi='small',dropout=0.25, n_classes=4):
        super(MQP_surv, self).__init__()
        self.size_dict_path = {"small": [768, 256,256]}

        # FC Layer over WSI bag
        size = self.size_dict_path[model_size_wsi]
        fc = [nn.Linear(size[0], size[1]), nn.ReLU()]
        fc.append(nn.Dropout(0.25))
        self.wsi_net = nn.Sequential(*fc)

        self.pool=MQP(dim=size[-1])

        self.GAP = Attn_Net_Gated(L=size[-1], D=size[-1], dropout=dropout, n_classes=1)
        self.gap_rho = nn.Sequential(
            *[nn.Linear(size[-1], size[-1]), nn.ReLU(), nn.Dropout(dropout)])

        ### Classifier
        self.classifier = nn.Linear(size[-1], n_classes)


    def forward(self, **kwargs):
        x_path = kwargs['x_path']
        
        h_path_bag = self.wsi_net(x_path).unsqueeze(1) ### path embeddings are fed through a FC layer
        h_path_bag=self.pool(h_path_bag).transpose(0, 1)
        
        A_path, h_path_bag_gap = self.GAP(h_path_bag)
        A_path = F.softmax(A_path, dim=1)
        h_path_bag_gap = torch.bmm(A_path.transpose(-2, -1), h_path_bag_gap).transpose(0,1)
        h=self.gap_rho(h_path_bag_gap).squeeze()
        
       
        ### Survival Layer
        logits = self.classifier(h).unsqueeze(0)
        Y_hat = torch.topk(logits, 1, dim = 1)[1]
        hazards = torch.sigmoid(logits)
        S = torch.cumprod(1 - hazards, dim=1)
        
        
        return {
            "hazards":hazards,
            "fusion":h,
            "S":S,
            "Y_hat":Y_hat,
        }