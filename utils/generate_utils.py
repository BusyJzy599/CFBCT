from argparse import Namespace
import os

import torch.nn as nn
import torch
import numpy as np
import math
from collections import defaultdict
from sksurv.metrics import concordance_index_censored
from sklearn.metrics.pairwise import cosine_similarity
from utils.utils import temperature_scaled_softmax


class Monitor_CIndex:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 20
            stop_epoch (int): Earliest epoch possible for stopping
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
        """
        self.best_score = None

    def __call__(self, val_cindex, model, ckpt_name:str='checkpoint.pt'):

        score = val_cindex

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model, ckpt_name)
        elif score > self.best_score:
            self.best_score = score
            self.save_checkpoint(model, ckpt_name)
        else:
            pass

    def save_checkpoint(self, model, ckpt_name):
        '''Saves model when validation loss decrease.'''
        torch.save(model.state_dict(), ckpt_name)

def generate_noise_dict(lengths):
    sample_dict={}
    for i in range(lengths):
        sample_dict[i]=[
            0.0, # cindex
            0.0  # loss
        ]
    return sample_dict,[]

def np_softmax(x):
    f_x = np.exp(x) / np.sum(np.exp(x))
    return f_x


def update_branch_weight(epoch,args,cfc,ema=0.9,**kwargs):
    train_event_times,train_censorships,train_risk_scores_gp,train_risk_scores_g,train_risk_scores_p=kwargs['train_event_times'],kwargs['train_censorships'],kwargs['train_risk_scores_gp'],kwargs['train_risk_scores_g'],kwargs['train_risk_scores_p']
    # cindex
    train_gp_cindex=concordance_index_censored((1-train_censorships).astype(bool), train_event_times, train_risk_scores_gp, tied_tol=1e-08)[0]
    train_g_cindex=concordance_index_censored((1-train_censorships).astype(bool), train_event_times, train_risk_scores_g, tied_tol=1e-08)[0]
    train_p_cindex=concordance_index_censored((1-train_censorships).astype(bool), train_event_times, train_risk_scores_p, tied_tol=1e-08)[0]
    # # cindex
    #### dynamic staregy 2
    if args.p_branch:
        cindexs=torch.FloatTensor([train_gp_cindex,train_g_cindex,train_p_cindex])
    else:
        cindexs=torch.FloatTensor([train_gp_cindex,train_g_cindex])
    ema=args.ema
    new_cfc=ema*cfc+(1-ema)*temperature_scaled_softmax(cindexs,temperature=args.tmp)
    
    with open(os.path.join(args.writer_dir, 'log.txt'), 'a') as f:
        f.write(f"cfc:{new_cfc}"+'\n')
    f.close()
    return new_cfc
    

def argtopK(matrix, K, axis=0,reverse=True):
    """
    perform topK based on np.argsort
    :param matrix: to be sorted
    :param K: select and sort the top K items
    :param axis: dimension to be sorted.
    :return:
    """
    full_sort = np.argsort(matrix, axis=axis)
    lens=full_sort.shape[0]
    rg=np.arange(lens)
    if reverse:
        return full_sort.take(rg[lens-K:], axis=axis)        
    else:
        return full_sort.take(rg[:K], axis=axis)


def generate_loss_fn(args):
    if args.task_type == 'survival':
        if args.bag_loss == 'ce_surv':
            from utils.utils import CrossEntropySurvLoss
            loss_fn = {
                "surv_loss":CrossEntropySurvLoss(alpha=args.alpha_surv)
                }
        elif args.bag_loss == 'nll_surv':
            from utils.utils import NLLSurvLoss
            if args.model_type=='cfbct':
                loss_fn = {
                    "surv_loss": NLLSurvLoss(alpha=args.alpha_surv),
                    "surv_loss_g": NLLSurvLoss(alpha=args.alpha_surv),
                    "surv_loss_p": NLLSurvLoss(alpha=args.alpha_surv),
                    "kl_loss":nn.KLDivLoss(reduction='batchmean'),
                    }
            elif args.model_type=='mbct':
                loss_fn = {
                    "surv_loss": NLLSurvLoss(alpha=args.alpha_surv),
                }
            else:
                loss_fn = {
                    "surv_loss": NLLSurvLoss(alpha=args.alpha_surv),    
                }

        elif args.bag_loss == 'cox_surv':
            from utils.utils import CoxSurvLoss
            loss_fn = {
                "surv_loss": CoxSurvLoss()
                }
        else:
            raise NotImplementedError
    elif args.task_type == 'HRD':
        from utils.utils import CrossEntropySurvLoss
        loss_fn = {
             "surv_loss":CrossEntropySurvLoss(alpha=args.alpha_surv)
             }
    else:
        raise NotImplementedError
    if args.reg_type == 'omic':
        from utils.utils import l1_reg_all
        reg_fn = l1_reg_all
    elif args.reg_type == 'pathomic':
        from utils.utils import l1_reg_modules
        reg_fn = l1_reg_modules
    else:
        reg_fn = None
    return loss_fn,reg_fn


def generate_model(args):
    model_dict = {"dropout": args.drop_out, 'n_classes': args.n_classes}
    args.fusion = None if args.fusion == 'None' else args.fusion

    if args.model_type =='snn':
        from models.model_genomic import SNN
        model_dict = {'omic_input_dim': args.omic_input_dim, 'model_size_omic': args.model_size_omic, 'n_classes': args.n_classes}
        model = SNN(**model_dict)
    elif args.model_type == 'mlp':
        from models.model_genomic import MLP
        model_dict = {'omic_input_dim': args.omic_input_dim,  'n_classes': args.n_classes}
        model = MLP(**model_dict)
    elif args.model_type == 'mmlp':
        from models.model_genomic import MaskedOmics
        model_dict = {'omic_input_dim': args.omic_input_dim, 'df_comp': args.omic_sizes,'n_classes': args.n_classes}
        model = MaskedOmics(**model_dict)
    elif args.model_type == 'deepset':
        from models.model_mil import MIL_Sum_FC_surv
        model_dict = {'fusion': None, 'n_classes': args.n_classes}
        model = MIL_Sum_FC_surv(**model_dict)  
    elif args.model_type == 'amil':
        from models.model_mil import MIL_Attention_FC_surv
        model_dict = {'fusion': None, 'n_classes': args.n_classes}
        model = MIL_Attention_FC_surv(**model_dict)  
    elif args.model_type == 'amisl':
        from models.model_mil import MIL_Cluster_FC_surv
        model_dict = {'fusion': None, 'n_classes': args.n_classes}
        model = MIL_Cluster_FC_surv(**model_dict)  
    elif args.model_type == 'tmil':
        from models.model_mil import TMIL
        model_dict = {'fusion': None, 'n_classes': args.n_classes}
        model = TMIL(**model_dict)  

    elif args.model_type == 'clam-sb':
        from models.model_mil import CLAM_SB
        model_dict = {'n_classes': args.n_classes}
        model = CLAM_SB(**model_dict)  
    elif args.model_type == 'clam-mb':
        from models.model_mil import CLAM_MB
        model_dict = {'n_classes': args.n_classes}
        model = CLAM_MB(**model_dict)  
    elif args.model_type == 'mqp':
        from models.model_mqp import MQP_surv
        model_dict = {'n_classes': args.n_classes}
        model = MQP_surv(**model_dict)  
    elif args.model_type == 'mcat':
        from models.model_mcat import MCAT_Surv
        model_dict = {'fusion': args.fusion, 'omic_sizes': args.omic_sizes, 'n_classes': args.n_classes}
        model = MCAT_Surv(**model_dict)
    elif args.model_type == 'cfbct':
        from models.model_cfbct import CFBCT_Surv
        model_dict = {'fusion': args.fusion, 'omic_sizes': args.omic_sizes, 'n_classes': args.n_classes,'pool':args.pooling,'p_branch':args.p_branch}
        model = CFBCT_Surv(**model_dict)  
    elif args.model_type == 'mbct':
        from models.model_mbct import MBCT_Surv
        model_dict = {'fusion': args.fusion, 'omic_sizes': args.omic_sizes, 'n_classes': args.n_classes}
        model = MBCT_Surv(**model_dict)  


        
    else:
        raise NotImplementedError
    if hasattr(model, "relocate"):
        model.relocate()
    else:
        model = model.cuda()
    return model