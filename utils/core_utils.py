from argparse import Namespace
import os

import numpy as np
import torch

from dataset.dataset_generic import save_splits
from utils.utils import get_optim, get_split_loader,get_train_loader
from utils.generate_utils import *


def train(datasets: tuple, cur: int, args: Namespace,recorder):
    """   
        train for a single fold
    """
    print('\nTraining Fold {}!'.format(cur))
    args.writer_dir = os.path.join(args.results_dir, str(cur))
    if not os.path.isdir(args.writer_dir):
        os.mkdir(args.writer_dir)
    if args.log_data:
        from tensorboardX import SummaryWriter
        writer = SummaryWriter(args.writer_dir, flush_secs=15)
    else:
        writer = None

    print('\nInit train/val/test splits...', end=' ')
    train_split, val_split = datasets
    # save_splits(datasets, ['train', 'val'], os.path.join(args.results_dir, 'splits_{}.csv'.format(cur)))
    print('Done!')
    print("Training on {} samples".format(len(train_split)))
    print("Validating on {} samples".format(len(val_split)))

    print('\nInit loss function...', end=' ')
    loss_fn,reg_fn=generate_loss_fn(args=args)
    print('Done!')
    
    print('\nInit Model...', end=' ')
    model=generate_model(args=args)
    # if args.model_type=='cmta':
    #     model= nn.DataParallel(model)
    if args.load_model:
        model.load_state_dict(torch.load(args.path_load_model))
    print('Done!')
    
    print('\nInit optimizer ...', end=' ')
    optimizer = get_optim(model, args)
    print('Done!')

    print('\nInit Loaders...', end=' ')
    train_loader = get_split_loader(train_split, training=True, testing = False, 
        weighted = args.weighted_sample, mode=args.mode, batch_size=args.batch_size)
    val_loader = get_split_loader(val_split,  testing = False, mode=args.mode, batch_size=args.batch_size)
    
    print('Done!')
    
    print('\nSetup Validation C-Index Monitor...', end=' ')
    monitor_cindex = Monitor_CIndex()
    print('Done!')

    latest_c_index = 0.
    max_c_index = 0.
    epoch_max_c_index = 0
    W_k=args.W_k
    if args.p_branch:
        cfc = torch.FloatTensor([W_k,(1-W_k)/2,(1-W_k)/2])
    else:
        cfc = torch.FloatTensor([W_k,(1-W_k)])

    best_val_dict = {}
    recorder.info("running with {} {}".format(args.model_type, args.mode))
    sample_dict,noise_list=generate_noise_dict(len(train_split))
    for epoch in range(args.start_epoch,args.max_epochs):
        
        if args.mode == 'coattn':
            if args.model_type=='cfbct':
                from trainer.cf_trainer import train_loop_survival_cf, validate_survival_cf
                train_event_times,train_censorships,train_risk_scores_gp,train_risk_scores_g,train_risk_scores_p=train_loop_survival_cf(cur,epoch, model, train_loader, optimizer, writer, loss_fn,reg_fn,args,cfc)
                val_latest, c_index_val,eval_event_times,eval_censorships,eval_risk_scores_gp,eval_risk_scores_g,eval_risk_scores_p = validate_survival_cf(cur,epoch,model, val_loader, writer, loss_fn, reg_fn,args,cfc)
                cfc=update_branch_weight(epoch=epoch,args=args,cfc=cfc,
                                         train_event_times=train_event_times,
                                         train_censorships=train_censorships,
                                         train_risk_scores_gp=train_risk_scores_gp,
                                         train_risk_scores_g=train_risk_scores_g,
                                         train_risk_scores_p=train_risk_scores_p,
                                        )
            elif args.model_type=='motcat':
                from trainer.mt_trainer import train_loop_survival_coattn_mt, validate_survival_coattn_mt
                train_loop_survival_coattn_mt(epoch, args.bs_micro, model, train_loader, optimizer, args.n_classes, writer, loss_fn, reg_fn, args.lambda_reg, args.gc, args)
                val_latest, c_index_val, stop = validate_survival_coattn_mt(cur, epoch, args.bs_micro, model, val_loader, args.n_classes, None, monitor_cindex, writer, loss_fn, reg_fn, args.lambda_reg, args.results_dir, args)   
            elif args.model_type in ['mcat','cmta','mbct','porpoise','survpath','ponet','mgct']:
                from trainer.coattn_trainer import train_loop_survival_coattn, validate_survival_coattn
                train_loop_survival_coattn(epoch, model, train_loader, optimizer, writer, loss_fn,reg_fn,args)
                val_latest, c_index_val = validate_survival_coattn(epoch,model, val_loader, writer, loss_fn, reg_fn, args)
            else:
                raise NotImplementedError
        elif args.mode == 'omic':
            from trainer.omic_trainer import train_survival, validate_survival
            train_survival(epoch, model, train_loader, optimizer, writer, loss_fn,reg_fn,args)
            val_latest, c_index_val = validate_survival(epoch,model, val_loader, writer, loss_fn, reg_fn, args)
        elif args.mode == 'path':
            from trainer.path_trainer import train_survival, validate_survival
            train_survival(epoch, model, train_loader, optimizer, writer, loss_fn,reg_fn,args)
            val_latest, c_index_val = validate_survival(epoch,model, val_loader, writer, loss_fn, reg_fn, args)
        elif args.mode == 'cluster':
            from trainer.cluster_trainer import train_survival, validate_survival
            train_survival(epoch, model, train_loader, optimizer, writer, loss_fn,reg_fn,args)
            val_latest, c_index_val = validate_survival(epoch,model, val_loader, writer, loss_fn, reg_fn, args)
        if c_index_val > max_c_index:
            max_c_index = c_index_val
            epoch_max_c_index = epoch
            save_name = 's_{}_checkpoint'.format(cur)
            if args.load_model and os.path.isfile(os.path.join(
                args.results_dir, save_name+".pt".format(cur))):
                save_name+='_load'
            if args.save_ckp:
                torch.save(model.state_dict(), os.path.join(
                    args.results_dir, save_name+".pt".format(cur)))
            best_val_dict = val_latest


        with open(os.path.join(args.writer_dir, 'log.txt'), 'a') as f:
            f.write("** Best C-Index %.4f (Epoch: %d) CF-c: %s **"%(max_c_index,epoch_max_c_index,str(cfc))+"\n")
        f.close()

    if args.log_data:
        writer.close()
    print_results = {'result': (max_c_index, epoch_max_c_index)}
    recorder.info("================= summary of fold {} ====================".format(cur))
    recorder.info("result: {:.4f}".format(max_c_index))
    with open(os.path.join(args.writer_dir, 'log.txt'), 'a') as f:
        f.write('result: {:.4f}, epoch: {}\n'.format(max_c_index, epoch_max_c_index))

    return best_val_dict, print_results
