import torch
from torch import linalg as LA
import torch.nn.functional as F
import torch.nn as nn

from models.model_utils import *





#############################
### CFBCT Implementation ###
#############################

class GAPPool(torch.nn.Module):
    def __init__(self,  dim1=256,dim2=256,dropout=0.25,n_classes=6,rho='mlp'):
        super(GAPPool, self).__init__()
        self.p_rho=nn.Sequential(
            *[nn.Linear(dim1, dim1), nn.ReLU(), nn.Dropout(dropout)])
        self.GAP = Attn_Net_Gated(L=dim1, D=dim2, dropout=dropout, n_classes=n_classes)
        if rho=='linear':
            self.gap_rho = nn.Sequential(
                *[nn.Linear(dim2, dim2), nn.ReLU(), nn.Dropout(dropout)])
        elif rho=='mlp':
            self.gap_rho = MLP_Block(dim2,projection_dim=512,dropout=0.1)

    def forward(self, x):  # （b,n,c）
        x=self.p_rho(x.transpose(0, 1))
        A_path, h_path_bag_gap = self.GAP(x)
        A_path = F.softmax(A_path, dim=1)
        h_path_bag_gap = torch.bmm(A_path.transpose(-2, -1), h_path_bag_gap).transpose(0,1)
        h_path_bag_gap=self.gap_rho(h_path_bag_gap)
        return h_path_bag_gap
   




class MQP(torch.nn.Module):
    def __init__(self,  dim=256,dropout=0.25):
        super(MQP, self).__init__()
        self.p_rho=nn.Sequential(
            *[nn.Linear(dim, dim), nn.ReLU(), nn.Dropout(dropout)])
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.maxpool = nn.AdaptiveMaxPool1d(1)
        self.rho=MLP_Block(dim,projection_dim=512,dropout=0.1)


    def forward(self, x):  # （b,n,c）
        x=self.p_rho(x.transpose(0, 1))
        x_max= self.maxpool(x.transpose(1, 2)).transpose(-2, -1)
        x_mean=self.avgpool(x.transpose(1, 2)).transpose(-2, -1)
        

        x_std=torch.std(x,dim=1,keepdim=True,unbiased=False)
        x_var=torch.var(x,dim=1,keepdim=True,unbiased=False)

        x_l=torch.quantile(x,0.25,dim=1,keepdim=True)
        x_h=torch.quantile(x,0.75,dim=1,keepdim=True)

       
        out=self.rho(torch.cat([x_max,x_mean,x_std,x_var,x_l,x_h]))
        return out
    
 
class TCoAttn(nn.Module):
    def __init__(self,
                 dim1=256,
                 dim2=256,
                 nhead=8,
                 num_layers=1,
                 top_percent=1.0,
                 pool="MQP",
                 dropout=0.25):
        super(TCoAttn, self).__init__()
        self.top_percent=top_percent
        omic_SA = nn.TransformerEncoderLayer(
            d_model=dim1, nhead=nhead, dim_feedforward=512, dropout=dropout, activation='relu')
        self.omic_SA = nn.TransformerEncoder(omic_SA, num_layers=num_layers)

        path_SA = nn.TransformerEncoderLayer(
            d_model=dim2, nhead=nhead, dim_feedforward=512, dropout=dropout, activation='relu')
        self.path_SA = nn.TransformerEncoder(path_SA, num_layers=num_layers)
        path_SA_1 = nn.TransformerEncoderLayer(
            d_model=dim2, nhead=nhead, dim_feedforward=512, dropout=dropout, activation='relu')
        self.path_SA_1 = nn.TransformerEncoder(path_SA_1, num_layers=num_layers)

        self.coattn_1 = MultiheadAttention(embed_dim=dim1, num_heads=1)
        self.coattn_2 = MultiheadAttention(embed_dim=dim2, num_heads=1)
        self.coattn_3 = MultiheadAttention(embed_dim=dim2, num_heads=1)

        if pool=="MQP":
            self.pool=MQP(dim=dim1)
            
    def top_feat(self,path_A,path_all):
        sum_attn=torch.sum(path_A.squeeze(),dim=0)
        total_elements = sum_attn.size(0)
        top_k = int(self.top_percent * total_elements)
        sum_attn, top_idx = torch.topk(sum_attn, top_k)
        path_all=torch.index_select(path_all,dim=0,index=top_idx)
        _, top_idx= torch.topk(sum_attn, int(self.top_percent*top_k),largest=False)
        path_all=torch.index_select(path_all,dim=0,index=top_idx)
        return path_all
    
    def forward(self, omic,path_all):
        # coatten1
        h_path_coattn, path_A = self.coattn_1(omic, path_all, path_all)
        h_path_bag_trans=self.path_SA(h_path_coattn)
        # pool
        path_all=self.top_feat(path_A,path_all)
        path_gap=self.pool(path_all)
        h_path_coattn_1, path_A_1=self.coattn_3(omic, path_gap, path_gap)
        h_path_bag_trans_1=self.path_SA_1(h_path_coattn_1)
        # coatten2
        h_omic_coattn, omic_A = self.coattn_2(path_gap, omic, omic)
        h_omic_bag_trans=self.omic_SA(h_omic_coattn)
        
        

        return {
            'path_attention':path_A,
            "h_path_coattn":h_path_coattn,
            # "h_path_coattn_1":h_path_coattn_1,
            # "h_omic_coattn":h_omic_coattn,
            "h_path_pool":path_gap,
            "h_path_bag_trans":h_path_bag_trans,
            "h_path_bag_trans_1":h_path_bag_trans_1,
            "h_omic_bag_trans":h_omic_bag_trans,
        }


class CFBCT_Surv(nn.Module):
    def __init__(self,
                 fusion='concat',
                 omic_sizes=[100, 200, 300, 400, 500, 600], 
                 n_classes=4,
                 num_layers=1,
                 model_size_wsi: str='small',
                 model_size_omic: str='small', 
                 pool:str="MQP",
                 p_branch:bool=True,
                 end_cls: bool=True,
                 eps=1e-7,
                 dropout=0.25):
        super(CFBCT_Surv, self).__init__()
        self.n_classes = n_classes
        self.eps=eps
        self.p_branch=p_branch
        self.end_cls=end_cls
        self.cfusion=True
        self.fusion = fusion
        self.constant_g = nn.Parameter(torch.tensor(0.0))
        
        self.omic_sizes = omic_sizes
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

        # gene branch
        self.g_net = nn.Sequential(
            GAPPool(hidden[-1],hidden[-1],dropout=dropout,n_classes=1,rho='linear'),
            nn.Linear(hidden[-1], n_classes),
        )
        if self.end_cls:
            self.g_cls=nn.Linear(n_classes, n_classes)
        # path branch
        if self.p_branch:
            self.p_net = nn.Sequential(
                GAPPool(hidden[-1],hidden[-1],dropout=dropout,n_classes=1,rho='linear'),
                nn.Linear(hidden[-1], n_classes),                
            )
            if self.end_cls:
                self.p_cls=nn.Linear(n_classes, n_classes)

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
        # 
        if self.fusion == 'bilinear':
            h = self.mm(h_path.unsqueeze(dim=0), h_omic.unsqueeze(dim=0)).squeeze()
        elif self.fusion == 'concat':
            h = self.mm(torch.cat([h_path, h_omic], axis=0))
        ### Survival Layer
        logits = self.classifier(h).unsqueeze(0)

        # Gene branch+Path branch
        gene_feature=grad_mul_const(h_omic_bag.clone(),0.0) # don't backpropagate
        g_h=self.g_net(gene_feature).view(1,-1)
        if self.p_branch:
            path_feature=grad_mul_const(dca['h_path_pool'].clone(),0.0) # don't backpropagate
            # path_feature=grad_mul_const(torch.cat([dca['h_path_coattn'].clone(),dca['h_path_pool'].clone()],dim=0),0.0) # don't backpropagate
            p_h=self.p_net(path_feature).view(1,-1)
        else:
            p_h=None
        # both g, k and p are the facts
        z_gkp = self.fusion_cf(logits, g_h, p_h, g_fact=True,  k_fact=True, p_fact=True,cfc=kwargs['cfc']) # te
        z_g = self.fusion_cf(logits, g_h, p_h, g_fact=True,  k_fact=False, p_fact=False,cfc=kwargs['cfc']) # nie
        logits_cf = z_gkp - float(kwargs['cfc'][1])*z_g

        
        # end classify
        if self.end_cls:
            g_hazard=self.g_cls(g_h)
            if self.p_branch:
                p_hazard=self.p_cls(p_h)
            else:
                p_hazard=None
        # NDE
        if self.p_branch:
            z_nde=self.fusion_cf(logits.clone().detach(),
                                 g_h.clone().detach(), 
                                 p_h.clone().detach(), 
                                 g_fact=True,  k_fact=False, p_fact=False,cfc=kwargs['cfc'])
        else:
            z_nde=self.fusion_cf(logits.clone().detach(),
                                 g_h.clone().detach(), 
                                 None, 
                                 g_fact=True,  k_fact=False, p_fact=False,cfc=kwargs['cfc'])
        return {
            # hazards
            "hazards":self.get_hazard(z_gkp)[1],
            "gp_hazards":self.get_hazard(logits)[1],
            "g_hazards":self.get_hazard(g_hazard)[1],
            "p_hazards":self.get_hazard(p_hazard)[1] if self.p_branch else None,
            "cf_hazards":self.get_hazard(logits_cf)[1],
            # S
            "S":self.get_hazard(z_gkp)[2],
            "gp_S":self.get_hazard(logits)[2],
            "g_S":self.get_hazard(g_hazard)[2],
            "p_S":self.get_hazard(p_hazard)[2] if self.p_branch else None,
            "cf_S":self.get_hazard(logits_cf)[2],
            # nde
            'z_nde' :z_nde,
            # 
            "z_gkp":z_gkp,
            "z_g":z_g,
            "attention_scores":dca['path_attention']
        }
        
    def get_hazard(self,logits):
        Y_hat = torch.topk(logits, 1, dim = 1)[1]
        hazards = torch.sigmoid(logits)
        S = torch.cumprod(1 - hazards, dim=1)
        return Y_hat,hazards,S
    
    
    def fusion_cf(self, z_k, z_g, z_p, g_fact=False, k_fact=False, p_fact=False,cfc=None):
        z_k, z_g, z_p = self.transform(z_k, z_g, z_p, g_fact, k_fact, p_fact)
        if self.cfusion:
            z = float(cfc[0])*z_k+float(cfc[1])*z_g
            if self.p_branch:
                z = float(cfc[0])*z_k+float(cfc[1])*z_g+float(cfc[2])*z_p

        return z
    def transform(self, z_k, z_g, z_p, g_fact=False, k_fact=False, p_fact=False):  

        if not k_fact:
            z_k = self.constant_g * torch.ones_like(z_k)

        if not g_fact:
            z_g = self.constant_g * torch.ones_like(z_g)
        if self.p_branch:
            if not p_fact:
                z_p = self.constant_g * torch.ones_like(z_p)
                
        return z_k, z_g, z_p
