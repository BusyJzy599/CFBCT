import numpy as np
import torch
import torch.nn.functional as F

from utils.utils import temperature_scaled_softmax,caculate_risk,np_softmax
import os
from sksurv.metrics import concordance_index_censored


def boundary(loss,b):
    return (loss-b).abs()+b

def train_loop_survival_coattn(epoch, model, loader, optimizer,  writer=None, loss_fn=None, reg_fn=None, args=None):   
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    model.train()
    train_loss_surv, train_loss = 0., 0.

    print('\n')
    all_risk_scores = np.zeros((len(loader)))
    all_censorships = np.zeros((len(loader)))
    all_event_times = np.zeros((len(loader)))
    case_ids = loader.dataset.slide_data['case_id']
    for batch_idx, (data_WSI, data_omic, label, event_time, c) in enumerate(loader):

        data_WSI = data_WSI.to(device)
        label = label.type(torch.LongTensor).to(device)
        c = c.type(torch.FloatTensor).to(device)

        input_args={'x_path':data_WSI}
        
        if args.model_type=='ponet':
            x_omix=[]
            for i in range(len(data_omic[0])):
                x_omix.append(data_omic[0][i].type(torch.FloatTensor).to(device))
            input_args['x_omic'] = x_omix
            input_args['x_cluster'] = None
        else:
            for i in range(len(data_omic[0])):
                input_args['x_omic%s' % str(i+1)] = data_omic[0][i].type(torch.FloatTensor).to(device)

        output = model(**input_args)


        loss = loss_fn['surv_loss'](hazards=output['hazards'], S=output['S'], Y=label, c=c)
        

        loss_value = loss.item()

        if reg_fn is None:
            loss_reg = 0
        else:
            loss_reg = reg_fn(model) * args.lambda_reg

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
        loss = boundary(loss,args.b) / args.gc + loss_reg

        loss.backward()

        if (batch_idx + 1) % args.gc == 0: 
            optimizer.step()
            optimizer.zero_grad()

    # calculate loss and error for epoch
    train_loss_surv /= len(loader)
    train_loss /= len(loader)
    c_index_train = concordance_index_censored((1-all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)[0]
    
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

def validate_survival_coattn( epoch, model, loader,  writer=None, loss_fn=None, reg_fn=None, args=None):
    model.eval()
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu") 

    val_loss_surv, val_loss = 0., 0.
    all_risk_scores = np.zeros((len(loader)))
    all_censorships = np.zeros((len(loader)))
    all_event_times = np.zeros((len(loader)))

    slide_ids = loader.dataset.slide_data['slide_id']
    patient_results = {}

    for batch_idx, (data_WSI, data_omic, label, event_time, c) in enumerate(loader):

        data_WSI = data_WSI.cuda()
       
        label = label.type(torch.LongTensor).cuda()
        c = c.type(torch.FloatTensor).cuda()

        slide_id = slide_ids.iloc[batch_idx]
        data_WSI = data_WSI.to(device)
        label = label.type(torch.LongTensor).to(device)
        c = c.type(torch.FloatTensor).to(device)

        input_args={'x_path':data_WSI}
        if args.model_type=='ponet':
            x_omix=[]
            for i in range(len(data_omic[0])):
                x_omix.append(data_omic[0][i].type(torch.FloatTensor).to(device))
            input_args['x_omic'] = x_omix
            input_args['x_cluster'] = None
        else:
            for i in range(len(data_omic[0])):
                input_args['x_omic%s' % str(i+1)] = data_omic[0][i].type(torch.FloatTensor).to(device)
        with torch.no_grad():
            output = model(**input_args)

        loss = loss_fn['surv_loss'](hazards=output['hazards'], S=output['S'], Y=label, c=c, alpha=0)
        loss_value = loss.item()

        if reg_fn is None:
            loss_reg = 0
        else:
            loss_reg = reg_fn(model) * args.lambda_reg

        risk = -torch.sum(output['S'], dim=1).cpu().numpy()
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
    if writer:
        writer.add_scalar('val/loss_surv', val_loss_surv, epoch)
        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('val/c-index', c_index, epoch)



    return patient_results, c_index