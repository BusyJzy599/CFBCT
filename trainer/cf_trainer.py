import numpy as np
import torch
import torch.nn.functional as F

from utils.utils import temperature_scaled_softmax,caculate_risk,np_softmax
import os
from sksurv.metrics import concordance_index_censored


def boundary(loss,b):
    return (loss-b).abs()+b

def train_loop_survival_cf(cur,epoch, model, loader, optimizer, writer=None, loss_fn=None, reg_fn=None,args=None,cfc=None):   
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    model.train()
    train_loss_surv, train_loss = 0., 0.

    print('\n')
    all_loss = np.zeros((len(loader)))
    all_risk_scores = np.zeros((len(loader)))
    all_risk_scores_cf = np.zeros((len(loader)))
    all_risk_scores_gp = np.zeros((len(loader)))
    all_risk_scores_g = np.zeros((len(loader)))
    all_risk_scores_p = np.zeros((len(loader)))
    
    all_censorships = np.zeros((len(loader)))
    all_event_times = np.zeros((len(loader)))
    
    slide_ids = loader.dataset.slide_data['slide_id']

    z_gkps = []
    z_gs = []

    
    for batch_idx, (data_WSI, data_omic, label, event_time, c) in enumerate(loader):

        data_WSI = data_WSI.to(device)
        data_omic1 = data_omic[0][0].type(torch.FloatTensor).to(device)
        data_omic2 = data_omic[0][1].type(torch.FloatTensor).to(device)
        data_omic3 = data_omic[0][2].type(torch.FloatTensor).to(device)
        data_omic4 = data_omic[0][3].type(torch.FloatTensor).to(device)
        data_omic5 = data_omic[0][4].type(torch.FloatTensor).to(device)
        data_omic6 = data_omic[0][5].type(torch.FloatTensor).to(device)
        label = label.type(torch.LongTensor).to(device)
        c = c.type(torch.FloatTensor).to(device)

        output = model(cfc=cfc,x_path=data_WSI, x_omic1=data_omic1, x_omic2=data_omic2, x_omic3=data_omic3, x_omic4=data_omic4, x_omic5=data_omic5, x_omic6=data_omic6)
        loss1=loss_fn['surv_loss'](hazards=output['hazards'], S=output['S'], Y=label, c=c)
        loss2=loss_fn['surv_loss_g'](hazards=output['g_hazards'], S=output['g_S'], Y=label, c=c)
        if args.p_branch:
            loss3=loss_fn['surv_loss_p'](hazards=output['p_hazards'], S=output['p_S'], Y=label, c=c)
        else:
            loss3=0

        logits_rubi=output['hazards']
        nde = output['z_nde']
        p_te = torch.nn.functional.softmax(logits_rubi, -1).clone().detach()
        p_nde = torch.nn.functional.softmax(nde, -1)
        kl_loss = - p_te*p_nde.log()    
        kl_loss = kl_loss.sum(1).mean() 

        
        loss =boundary(loss1+loss2+loss3,args.b)+kl_loss
    
        loss_value = loss.item()

        if reg_fn is None:
            loss_reg = 0
        else:
            loss_reg = reg_fn(model) * args.lambda_reg
        ###
        z_gkps.append(output['z_gkp'])
        z_gs.append(output['z_g'])
        ###
        all_loss[batch_idx]=loss_value
        all_risk_scores_cf[batch_idx] = -torch.sum(output['cf_S'], dim=1).detach().cpu().numpy()
        # all_risk_scores_gp[batch_idx] = -torch.sum(output['gp_S'], dim=1).detach().cpu().numpy()
        all_risk_scores_gp[batch_idx] = -torch.sum(output['S'], dim=1).detach().cpu().numpy()
        all_risk_scores_g[batch_idx] = -torch.sum(output['g_S'], dim=1).detach().cpu().numpy()
        if args.p_branch:
            all_risk_scores_p[batch_idx] = -torch.sum(output['p_S'], dim=1).detach().cpu().numpy()
        risk = -torch.sum(output['S'], dim=1).detach().cpu().numpy()
        all_risk_scores[batch_idx] = risk
        all_censorships[batch_idx] = c.item()
        all_event_times[batch_idx] = event_time

        train_loss_surv += loss_value
        train_loss += loss_value + loss_reg



        if (batch_idx + 1) % 100 == 0:
            train_batch_str = 'batch {}, loss: {:.4f}, label: {}, event_time: {:.4f}, risk: {:.4f}'.format(
                batch_idx, loss_value, label.item(), float(event_time), float(risk))
            with open(os.path.join(args.writer_dir, 'log.txt'), 'a') as f:
                f.write(train_batch_str+'\n')
            f.close()
            print(train_batch_str)
        loss = loss / args.gc + loss_reg
        loss.backward()

        if (batch_idx + 1) % args.gc == 0: 
            optimizer.step()
            optimizer.zero_grad()

    # calculate loss and error for epoch
    train_loss_surv /= len(loader)
    train_loss /= len(loader)
    c_index_train = concordance_index_censored((1-all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)[0]
    ####
    c_index_train_gp = concordance_index_censored((1-all_censorships).astype(bool), all_event_times, all_risk_scores_gp, tied_tol=1e-08)[0]
    c_index_train_cf = concordance_index_censored((1-all_censorships).astype(bool), all_event_times, all_risk_scores_cf, tied_tol=1e-08)[0]
    c_index_train_g = concordance_index_censored((1-all_censorships).astype(bool), all_event_times, all_risk_scores_g, tied_tol=1e-08)[0]
    if args.p_branch:
        c_index_train_p = concordance_index_censored((1-all_censorships).astype(bool), all_event_times, all_risk_scores_p, tied_tol=1e-08)[0]
    
    train_epoch_str = 'Epoch: {}, train_loss_surv: {:.4f}, train_loss: {:.4f}, train_c_index: {:.4f}'.format(
        epoch, train_loss_surv, train_loss, c_index_train)
    print(train_epoch_str)
    with open(os.path.join(args.writer_dir, 'log.txt'), 'a') as f:
        f.write(train_epoch_str+'\n')
    f.close()
   

    if writer:
        writer.add_scalar('train/loss_surv', train_loss_surv, epoch)
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/c_index', c_index_train, epoch)
    return (all_event_times,all_censorships,all_risk_scores_gp,all_risk_scores_g,all_risk_scores_p)


def validate_survival_cf(cur,epoch, model, loader, writer=None, loss_fn=None, reg_fn=None, args=None,cfc=None):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu") 

    model.eval()
    val_loss_surv, val_loss = 0., 0.
    all_loss = np.zeros((len(loader)))
    all_risk_scores = np.zeros((len(loader)))
    all_censorships = np.zeros((len(loader)))
    all_event_times = np.zeros((len(loader)))
    all_risk_scores_gp = np.zeros((len(loader)))
    all_risk_scores_g = np.zeros((len(loader)))
    all_risk_scores_p = np.zeros((len(loader)))
    z_gkps,z_gs=[],[]
    slide_ids = loader.dataset.slide_data['slide_id']
    patient_results = {}

    for batch_idx, (data_WSI, data_omic, label, event_time, c) in enumerate(loader):

        data_WSI = data_WSI.cuda()
        data_omic1 = data_omic[0][0].type(torch.FloatTensor).to(device)
        data_omic2 = data_omic[0][1].type(torch.FloatTensor).to(device)
        data_omic3 = data_omic[0][2].type(torch.FloatTensor).to(device)
        data_omic4 = data_omic[0][3].type(torch.FloatTensor).to(device)
        data_omic5 = data_omic[0][4].type(torch.FloatTensor).to(device)
        data_omic6 = data_omic[0][5].type(torch.FloatTensor).to(device)
        label = label.type(torch.LongTensor).cuda()
        c = c.type(torch.FloatTensor).cuda()

        slide_id = slide_ids.iloc[batch_idx]

        with torch.no_grad():
            output = model(cfc=cfc,x_path=data_WSI, x_omic1=data_omic1, x_omic2=data_omic2, x_omic3=data_omic3, x_omic4=data_omic4, x_omic5=data_omic5, x_omic6=data_omic6)

        loss = loss_fn['surv_loss'](hazards=output['cf_hazards'], S=output['cf_S'], Y=label, c=c, alpha=0)
        loss_value = loss.item()

        if reg_fn is None:
            loss_reg = 0
        else:
            loss_reg = reg_fn(model) * args.lambda_reg
        ###
        all_risk_scores_gp[batch_idx] = -torch.sum(output['S'], dim=1).detach().cpu().numpy()
        all_risk_scores_g[batch_idx] = -torch.sum(output['g_S'], dim=1).detach().cpu().numpy()
        if args.p_branch:
            all_risk_scores_p[batch_idx] = -torch.sum(output['p_S'], dim=1).detach().cpu().numpy()
        ###
        z_gkps.append(output['z_gkp'])
        z_gs.append(output['z_g'])
        risk = -torch.sum(output['cf_S'], dim=1).cpu().numpy()
        all_loss[batch_idx]=loss_value
        all_risk_scores[batch_idx] = risk
        all_censorships[batch_idx] = c.cpu().numpy()
        all_event_times[batch_idx] = event_time
        patient_results.update({slide_id: {'slide_id': np.array(slide_id), 'risk': risk, 'disc_label': label.item(), 'survival': event_time.item(), 'censorship': c.item()}})

        val_loss_surv += loss_value
        val_loss += loss_value + loss_reg


    val_loss_surv /= len(loader)
    val_loss /= len(loader)
    c_index = concordance_index_censored((1-all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)[0]
    val_epoch_str = "val c-index: {:.4f}".format(c_index)
    print(val_epoch_str)
    with open(os.path.join(args.writer_dir, 'log.txt'), 'a') as f:
        f.write(val_epoch_str+'\n')
    ### 
    cf_cidx=np.zeros((20))
    with open(os.path.join(args.writer_dir, 'log.txt'), 'a') as f:
            f.write(f"="*30+'\n')

    if writer:
        writer.add_scalar('val/loss_surv', val_loss_surv, epoch)
        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('val/c-index', c_index, epoch)

   

    return patient_results, c_index,all_event_times,all_censorships,all_risk_scores_gp,all_risk_scores_g,all_risk_scores_p