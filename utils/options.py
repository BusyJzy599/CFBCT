from __future__ import print_function
import argparse
import os
import sys
isDebug = True if sys.gettrace() else False


def process_args():
    ### Training settingsq
    parser = argparse.ArgumentParser(
        description='Configurations for Survival Analysis on TCGA Data.')
    ### MB
    parser.add_argument('--log_data',    default=False, action='store_true', help='Log data using tensorboard')
    

    ### Parameters
    parser.add_argument('--organ',           type=str, default="TCGA")

    parser.add_argument('--b',           type=float, default=0.0)
    parser.add_argument('--W_k',           type=float, default=1.0)
    parser.add_argument('--ema',           type=float, default=0.9)
    parser.add_argument('--tmp',           type=float, default=0.1)
    parser.add_argument('--pooling',     type=str, default="MQP")
    parser.add_argument('--p_branch',    type=bool, default=True)
    parser.add_argument('--mode',            type=str, choices=['omic', 'path', 'coattn','coattn_mb'],
                        default="coattn", help='Specifies which modalities to use / collate function in dataloader.')
    
    ### Checkpoint + Misc. Pathing Parameters
    parser.add_argument('--data_root_dir',   type=str, default=f'' ,
                        help='Data directory to WSI features (extracted via CLAM')
    parser.add_argument('--seed', 			 type=int, default=6,
                        help='Random seed for reproducible experiment (default: 1)')
    parser.add_argument('--k', 			     type=int, default=5,
                        help='Number of folds (default: 5)')
    parser.add_argument('--k_start',		 type=int, default=-1,
                        help='Start fold (Default: -1, last fold)')
    parser.add_argument('--k_end',			 type=int, default=-1,
                        help='End fold (Default: -1, first fold)')
    parser.add_argument('--results_dir',     type=str, default='./results',
                        help='Results directory (Default: ./results)')
    parser.add_argument('--which_splits',    type=str, default='5foldcv',
                        help='Which splits folder to use in ./splits/ (Default: ./splits/5foldcv')
    parser.add_argument('--split_dir',       type=str, default=f'',
                        help='Which cancer type within ./splits/<which_splits> to use for training. Used synonymously for "task" (Default: tcga_blca)')
    parser.add_argument('--overwrite',     	 action='store_true', default=False,
                        help='Whether or not to overwrite experiments (if already ran)')
    parser.add_argument('--load_model',        action='store_true',
                        default=False, help='whether to load model')
    parser.add_argument('--save_pkl',        action='store_true',
                        default=True, help='whether to load model')
    parser.add_argument('--path_load_model', type=str,
                        default='/path/to/load', help='path of ckpt for loading')
    parser.add_argument('--start_epoch',              type=int,
                        default=0, help='start_epoch.')

    ### Model Parameters.
    parser.add_argument('--model_type',      type=str, 
                        default="cfbct", help='Type of model')
    parser.add_argument('--fusion',          type=str, choices=[
                        'None', 'concat'], default='concat', help='Type of fusion. (Default: concat).')
    parser.add_argument('--apply_sig',		 action='store_true', default=True,
                        help='Use genomic features as signature embeddings.')
    parser.add_argument('--apply_sigfeats',  action='store_true',
                        default=False, help='Use genomic features as tabular features.')
    parser.add_argument('--drop_out',        action='store_true',
                        default=True, help='Enable dropout (p=0.25)')
    parser.add_argument('--model_size_wsi',  type=str,
                        default='small', help='Network size of AMIL model')
    parser.add_argument('--model_size_omic', type=str,
                        default='small', help='Network size of SNN model')

    ### Optimizer Parameters + Survival Loss Function
    parser.add_argument('--opt',             type=str,
                        choices=['adam', 'sgd'], default='adam')
    parser.add_argument('--batch_size',      type=int, default=1,
                        help='Batch Size (Default: 1, due to varying bag sizes)')
    parser.add_argument('--gc',              type=int,
                        default=32, help='Gradient Accumulation Step.')
    parser.add_argument('--max_epochs',      type=int, default=20,
                        help='Maximum number of epochs to train (default: 20)')
    parser.add_argument('--lr',				 type=float, default=2e-4,
                        help='Learning rate (default: 0.0002)')
    parser.add_argument('--bag_loss',        type=str, choices=['svm', 'ce', 'ce_surv', 'nll_surv',
                        'cox_surv'], default='nll_surv', help='slide-level classification loss function (default: nll_surv)')
    parser.add_argument('--label_frac',      type=float, default=1.0,
                        help='fraction of training labels (default: 1.0)')
    parser.add_argument('--bag_weight',      type=float, default=0.7,
                        help='clam: weight coefficient for bag-level loss (default: 0.7)')
    parser.add_argument('--reg', 			 type=float, default=1e-5,
                        help='L2-regularization weight decay (default: 1e-5)')
    parser.add_argument('--alpha_surv',      type=float, default=0.0,
                        help='How much to weigh uncensored patients')
    parser.add_argument('--reg_type',        type=str, choices=['None', 'omic', 'pathomic'],
                        default='None', help='Which network submodules to apply L1-Regularization (default: None)')
    parser.add_argument('--lambda_reg',      type=float, default=1e-4,
                        help='L1-Regularization Strength (Default 1e-4)')
    parser.add_argument('--weighted_sample', action='store_true',
                        default=True, help='Enable weighted sampling')
    parser.add_argument('--early_stopping',  action='store_true',
                        default=False, help='Enable early stopping')
    ### MOTCat Parameters
    parser.add_argument('--bs_micro',      type=int, default=256,
                    help='The Size of Micro-batch (Default: 256)') ### new
    parser.add_argument('--ot_impl', 			 type=str, default='pot-uot-l2',
                    help='impl of ot (default: pot-uot-l2)') ### new
    parser.add_argument('--ot_reg', 			 type=float, default=0.1,
                    help='epsilon of OT (default: 0.1)')
    parser.add_argument('--ot_tau', 			 type=float, default=0.5,
                    help='tau of UOT (default: 0.5)')
    args = parser.parse_args()
    if isDebug:
        args.log_data=False
        
    return args
