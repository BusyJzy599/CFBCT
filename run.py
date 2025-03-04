from __future__ import print_function
from utils.options import process_args
import utils.options as options
import os
import sys
from timeit import default_timer as timer
import numpy as np
from dataset.dataset_survival_tcga import Generic_MIL_Survival_Dataset
from dataset.dataset_survival_cptac import Generic_MIL_Survival_CPTAC_Dataset
from dataset.dataset_biomarker import Generic_MIL_Biomarker_Dataset
from utils.file_utils import save_pkl
from utils.core_utils import train
from utils.utils import *
import torch
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")



args = process_args()


######
 
args=get_custom_exp_code(args)
TASK='survival'
args.task = '_'.join(args.split_dir.split('_')[:2]) + f'_{TASK}'
args.task_type = f'{TASK}'
seed_torch(args.seed)

encoding_size = 768
settings = {'data_root_dir': args.data_root_dir,
            'num_splits': args.k,
            'k_start': args.k_start,
            'k_end': args.k_end,
            'task': args.task,
            'max_epochs': args.max_epochs,
            'results_dir': args.results_dir,
            'lr': args.lr,
            'experiment': args.exp_code,
            'reg': args.reg,
            'label_frac': args.label_frac,
            'bag_loss': args.bag_loss,
            'bag_weight': args.bag_weight,
            'seed': args.seed,
            'model_type': args.model_type,
            'model_size_wsi': args.model_size_wsi,
            'model_size_omic': args.model_size_omic,
            "use_drop_out": args.drop_out,
            'weighted_sample': args.weighted_sample,
            'gc': args.gc,
            'opt': args.opt,

            }

print('\nLoad Dataset')
if 'survival' in args.task:
    args.n_classes = 4
    combined_study = '_'.join(args.task.split('_')[:2])

    if combined_study in ['tcga_blca', 'tcga_brca','tcga_gbmlgg', 'tcga_ucec', 'tcga_luad','tcga_lgg','tcga_gbm',
                          'tcga_coad','tcga_hnsc','tcga_stad']:
        csv_path = './%s/%s_all_clean.csv.zip' % (args.dataset_path, combined_study)

        dataset = Generic_MIL_Survival_Dataset(csv_path=csv_path,
                                            mode=args.mode,
                                            apply_sig=args.apply_sig,
                                            data_dir=args.data_root_dir,
                                            shuffle=False,
                                            seed=args.seed,
                                            print_info=True,
                                            patient_strat=False,
                                            n_bins=4,
                                            label_col='survival_months',
                                            ignore=[])
    elif combined_study in ['cptac_luad','cptac_gbm','cptac_ucec','cptac_lscc']:
        csv_path = './%s/%s_all_clean.csv.zip' % (args.dataset_path, combined_study)
        dataset = Generic_MIL_Survival_CPTAC_Dataset(csv_path=csv_path,
                                        mode=args.mode,
                                        apply_sig=args.apply_sig,
                                        data_dir=args.data_root_dir,
                                        shuffle=False,
                                        seed=args.seed,
                                        print_info=True,
                                        patient_strat=False,
                                        n_bins=4,
                                        label_col='survival_days',
                                        ignore=[])
elif 'HRD' in args.task:
    args.n_classes = 1
    combined_study = '_'.join(args.task.split('_')[:2])
    csv_path = './%s/%s_all_clean.csv.zip' % (args.dataset_path, combined_study)
    dataset = Generic_MIL_Biomarker_Dataset(csv_path=csv_path,
                                           mode=args.mode,
                                           apply_sig=args.apply_sig,
                                           data_dir=args.data_root_dir,
                                           shuffle=False,
                                           seed=args.seed,
                                           print_info=True,
                                           patient_strat=False,
                                           label_col='HRD',
                                           ignore=[])
else:
    raise NotImplementedError





# Creates results_dir Directory.
if not os.path.isdir(args.results_dir):
    os.mkdir(args.results_dir)


exp_code =f"{str(args.exp_code)}_s{args.seed}"

# add logger
args.results_dir = os.path.join(args.results_dir, args.which_splits, args.param_code, exp_code)
recorder=Logger(file_name=f"experiment_{args.exp_code}",logger_path=args.results_dir)

recorder.info("===="*30)
recorder.info(f"Experiment Name:{exp_code}")
recorder.info("===="*30)

if not os.path.isdir(args.results_dir):
    os.makedirs(args.results_dir)
recorder.info(f"logs saved at {args.results_dir}")

if ('summary_latest.csv' in os.listdir(args.results_dir)) and (not args.overwrite):
    recorder.info("Exp Code <%s> already exists! Exiting script." % args.exp_code)
    sys.exit()

# Sets the absolute path of split_dir
args.split_dir = os.path.join('./splits', args.which_splits, args.split_dir)
recorder.info(f"split_dir {args.split_dir}")
assert os.path.isdir(args.split_dir)
settings.update({'split_dir': args.split_dir})
# print options
for item in settings.items():
    recorder.info(item)


recorder.info("################# Settings ###################")
for key, val in settings.items():
    recorder.info("{}:  {}".format(key, val))

def main(args):
    # Create Results Directory
    if not os.path.isdir(args.results_dir):
        os.mkdir(args.results_dir)

    if args.k_start == -1:
        start = 0
    else:
        start = args.k_start
    if args.k_end == -1:
        end = args.k
    else:
        end = args.k_end

    latest_val_cindex = []
    folds = np.arange(start, end)
    
    # Start 5-Fold CV Evaluation.
    run_folds = folds
    summary_all_folds = {}
    for i in folds:
        start_t = timer()
        seed_torch(args.seed)
        args.results_pkl_path = os.path.join(
            args.results_dir, 'split_latest_val_{}_results.pkl'.format(i))
        if os.path.isfile(args.results_pkl_path) and not args.load_model and not args.overwrite:
            recorder.info("Skipping Split %d" % i)
            aim_index = np.where(run_folds == i)[0][0]
            run_folds = np.delete(run_folds, aim_index)
            continue
        # Gets the Train + Val Dataset Loader.
        train_dataset, val_dataset = dataset.return_splits(from_id=False,
                                                           csv_path='{}/splits_{}.csv'.format(args.split_dir, i))
        # train_dataset, val_dataset = dataset.return_splits(from_id=False,
        #                                                    csv_path='{}/splits_t2c.csv'.format(args.split_dir))
        
        recorder.info('training: {}, validation: {}'.format(
            len(train_dataset), len(val_dataset)))
        datasets = (train_dataset, val_dataset)

        ### Specify the input dimension size if using genomic features.
        if 'omic' in args.mode or args.mode == 'cluster' or args.mode == 'graph' or args.mode == 'pyramid':
            args.omic_input_dim = train_dataset.genomic_features.shape[1]
            if args.model_type=='mmlp':
                args.omic_sizes = train_dataset.omic_sizes
                recorder.info(f'Genomic Dimensions {args.omic_sizes}')
            else:
                recorder.info(f"Genomic Dimension {args.omic_input_dim}" )
        elif 'coattn' in args.mode or args.mode == 'coattn_mb':
            args.omic_sizes = train_dataset.omic_sizes
            args.omic_input_dim = sum(train_dataset.omic_sizes)
            recorder.info(f'Genomic Dimensions {args.omic_sizes}')
        else:
            args.omic_input_dim = 0
        
        # Run Train-Val on Survival Task.
        if args.task_type == 'survival':
            summary_results, print_results = train(datasets, i, args,recorder) ### new

        # Write Results for Each Split to PKL
        if args.save_pkl:
            save_pkl(args.results_pkl_path, summary_results)
        summary_all_folds[i] = print_results
        end_t = timer()
        recorder.info('Fold %d Time: %f seconds' % (i, end_t - start_t))

    recorder.info('=============================== summary ===============================')
    result_cindex = []
    for i, k in enumerate(summary_all_folds):
        c_index = summary_all_folds[k]['result'][0]
        recorder.info("Fold {}, C-Index: {:.4f}".format(k, c_index))
        result_cindex.append(c_index)
    result_cindex = np.array(result_cindex)
    recorder.info(f"Experiment:{exp_code}")
    recorder.info("Avg C-Index of {} folds: {:.3f}, stdp: {:.3f}, stds: {:.3f}".format(
        len(summary_all_folds), result_cindex.mean(), result_cindex.std(), result_cindex.std(ddof=1)))

if __name__ == "__main__":
    start = timer()
    results = main(args)
    end = timer()
    recorder.info("finished!")
    recorder.info("end script")
    recorder.info(f'Script Time: {format_time(end - start)}')
