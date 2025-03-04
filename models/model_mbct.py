import torch
from torch import linalg as LA
import torch.nn.functional as F
import torch.nn as nn

from models.model_utils import *

class MBCT_Surv(nn.Module):
    def __init__(self,
                 fusion='concat',
                 omic_sizes=[100, 200, 300, 400, 500, 600], 
                 n_classes=4,
                 num_layers=1,
                 model_size_wsi: str='small',
                 model_size_omic: str='small', 
                 pool:str="MQP",
                 dropout=0.25):
        super(MBCT_Surv, self).__init__()
        self.fusion = fusion
        self.omic_sizes = omic_sizes
        self.n_classes = n_classes
        self.size_dict_WSI = {"small": [768, 256, 256], "big": [768, 512, 384]}
        self.size_dict_omic = {'small': [256, 256], 'big': [1024, 1024, 1024, 256]}

        # FC Layer over WSI bag
        size = self.size_dict_WSI[model_size_wsi]
        fc = [nn.Linear(size[0], size[1]), nn.ReLU()]
        fc.append(nn.Dropout(0.25))
        self.wsi_net = nn.Sequential(*fc)


        ### Constructing Genomic SNN
        hidden = self.size_dict_omic[model_size_omic]
        sig_networks = []
        for input_dim in omic_sizes:
            fc_omic = [SNN_Block(dim1=input_dim, dim2=hidden[0])]
            for i, _ in enumerate(hidden[1:]):
                fc_omic.append(SNN_Block(dim1=hidden[i], dim2=hidden[i+1], dropout=0.25))
            sig_networks.append(nn.Sequential(*fc_omic))
        self.sig_networks = nn.ModuleList(sig_networks)
        
        ### Constructing Dual Co-Attention
        self.dca=TCoAttn(size[-1],size[-1],nhead=8,num_layers=1,dropout=dropout,pool=pool)

        # Path Transformer + Attention Head
        path_encoder_layer = nn.TransformerEncoderLayer(
            d_model=256, nhead=8, dim_feedforward=512, dropout=dropout, activation='relu')
        self.path_transformer = nn.TransformerEncoder(
            path_encoder_layer, num_layers=num_layers)
        self.path_attention_head = Attn_Net_Gated(
            L=size[2], D=size[2], dropout=dropout, n_classes=1)
        self.path_rho = nn.Sequential(
            *[nn.Linear(size[2], size[2]), nn.ReLU(), nn.Dropout(dropout)])

        # Omic Transformer + Attention Head
        omic_encoder_layer = nn.TransformerEncoderLayer(
            d_model=256, nhead=8, dim_feedforward=512, dropout=dropout, activation='relu')
        self.omic_transformer = nn.TransformerEncoder(
            omic_encoder_layer, num_layers=num_layers)
        self.omic_attention_head = Attn_Net_Gated(
            L=size[2], D=size[2], dropout=dropout, n_classes=1)
        self.omic_rho = nn.Sequential(
            *[nn.Linear(size[2], size[2]), nn.ReLU(), nn.Dropout(dropout)])

        ### Fusion Layer
        if self.fusion == 'concat':
            self.mm = nn.Sequential(*[nn.Linear(256*2, size[2]), nn.ReLU(), nn.Linear(size[2], size[2]), nn.ReLU()])
        elif self.fusion == 'bilinear':
            self.mm = BilinearFusion(dim1=256, dim2=256, scale_dim1=8, scale_dim2=8, mmhid=256)
        else:
            self.mm = None

        ### Classifier
        self.classifier = nn.Linear(size[2], n_classes)
        
    def forward(self, **kwargs):
        x_path = kwargs['x_path']
        x_omic = [kwargs['x_omic%d' % i] for i in range(1,7)]   
        
        h_path_bag = self.wsi_net(x_path).unsqueeze(1) ### path embeddings are fed through a FC layer

        h_omic = [self.sig_networks[idx].forward(sig_feat) for idx, sig_feat in enumerate(x_omic)] ### each omic signature goes through it's own FC layer
        h_omic_bag = torch.stack(h_omic).unsqueeze(1) ### omic embeddings are stacked (to be used in co-attention)

        # dca
        dca=self.dca(h_omic_bag,h_path_bag)

        # Path
        h_path_trans = self.path_transformer(torch.cat([dca['h_path_bag_trans'],dca['h_path_bag_trans_1']],dim=0))
        A_path, h_path = self.path_attention_head(h_path_trans.squeeze(1))
        A_path = torch.transpose(A_path, 1, 0)
        h_path = torch.mm(F.softmax(A_path, dim=1), h_path)
        h_path = self.path_rho(h_path).squeeze()

        # Omic
        h_omic_trans = self.omic_transformer(dca['h_omic_bag_trans'])
        A_omic, h_omic = self.omic_attention_head(h_omic_trans.squeeze(1))
        A_omic = torch.transpose(A_omic, 1, 0)
        h_omic = torch.mm(F.softmax(A_omic, dim=1), h_omic)
        h_omic = self.omic_rho(h_omic).squeeze()

        if self.fusion == 'bilinear':
            h = self.mm(h_path.unsqueeze(dim=0), h_omic.unsqueeze(dim=0)).squeeze()
        elif self.fusion == 'concat':
            h = self.mm(torch.cat([h_path, h_omic], axis=0))


        ### Survival Layer
        logits = self.classifier(h).unsqueeze(0)
        Y_hat = torch.topk(logits, 1, dim = 1)[1]
        hazards = torch.sigmoid(logits)
        S = torch.cumprod(1 - hazards, dim=1)
        
        attention_scores = {'coattn': None, 'path': A_path, 'omic': A_omic}
        
        return {
            "hazards":hazards,
            "fusion":h,
            "h_path":h_path,
            "h_omic":h_omic,
            "S":S,
            "Y_hat":Y_hat,
            "MQP":dca['h_path_pool'],
            "attention_scores":dca['path_attention']
        }
