import os
import yaml
import argparse
import copy
import numpy as np
import torch
import torch.nn as nn
from torch.nn import BCELoss, MSELoss, CrossEntropyLoss, BCEWithLogitsLoss
from torch.optim import *
from torch.optim.lr_scheduler import StepLR
from scipy.io import loadmat
import sklearn
from sklearn.model_selection import KFold, GroupKFold, train_test_split

from eventaad.AEC import *
from eventaad.EEGModels import *
from eventaad.dataset import *
import eventaad.loss as L
from eventaad.loss import *
from utils.parallel import *
from utils.utils import pearson_scorer
from utils import logging
logger = logging.getLogger()
import matplotlib.pyplot as plt

global NUM_SBJS
NUM_SBJS = 24
WINDOWS = [1.2]
RIDGE_ALPHAS = tuple(10**(i/2) for i in range(-4,10))

from torchsummary import summary
  
def trainLinearEnvelope(model_config, trainset, testset, windows, sr, eeg_context, step=1.0):
    if isinstance(trainset, list):
        eeg_tr = np.concatenate([tr[0] for tr in trainset])
        attd_env_tr = np.concatenate([tr[1] for tr in trainset])
        unattd_env_tr = np.concatenate([tr[2] for tr in trainset])
        attd_evt_tr = np.concatenate([tr[3] for tr in trainset])
        unattd_evt_tr = np.concatenate([tr[4] for tr in trainset])
        group=0
        for i in range(len(trainset)):
            (_,_,_,_,_, groups_tr) = trainset[i]
            unique_tr, counts = np.unique(groups_tr, return_counts=True)
            for j in range(len(unique_tr)):
                trainset[i][5][groups_tr==unique_tr[j]] = group
                group+=1
        groups_tr = np.concatenate([tr[5] for tr in trainset])        
        #
        eeg_te = np.concatenate([te[0] for te in testset])
        attd_env_te = np.concatenate([te[1] for te in testset])
        unattd_env_te = np.concatenate([te[2] for te in testset])
        attd_evt_te = np.concatenate([te[3] for te in testset])
        unattd_evt_te = np.concatenate([te[4] for te in testset])
        group=0
        for i in range(len(testset)):
            (_,_,_,_,_, groups_te) = testset[i]
            unique_te, counts = np.unique(groups_te, return_counts=True)
            for j in range(len(unique_te)):
                testset[i][5][groups_te==unique_te[j]] = group
                group+=1
        groups_te = np.concatenate([te[5] for te in testset])
    else:
        (eeg_tr, attd_env_tr, unattd_env_tr, attd_evt_tr, unattd_evt_tr, groups_tr) = trainset
        (eeg_te, attd_env_te, unattd_env_te, attd_evt_te, unattd_evt_te, groups_te) = testset
    unique_tr, counts = np.unique(groups_tr, return_counts=True)
    unique_te, counts = np.unique(groups_te, return_counts=True)
    # train
    # cv_gen = GroupKFold(n_splits=5).split(eeg_tr, attd_env_tr, groups=groups_tr)    
    cv_gen = KFold(n_splits=5, shuffle=True, random_state=0)
    model = sklearn.linear_model.RidgeCV(alphas=RIDGE_ALPHAS, fit_intercept=True, scoring=pearson_scorer, cv=cv_gen, gcv_mode=None)
    model.fit(eeg_tr, attd_env_tr)
               
    # evaluate
    train_accs = []
    test_accs = []
    test_accs_evt = []
    step = 1 if int(step*sr)==0 else int(step*sr)
    for w in windows:
        L = int(w*sr)
        score_attn_tr = []
        score_unattn_tr = []
        score_attn_te = []
        score_unattn_te = []
        #
        score_attn_te_evt = []
        score_unattn_te_evt = []

        start = 0
        end = start + L
        while end<=attd_env_tr.shape[0]:
            score_attn_tr.append(pearson_scorer(model, eeg_tr[start:end], attd_env_tr[start:end]))
            score_unattn_tr.append(pearson_scorer(model, eeg_tr[start:end], unattd_env_tr[start:end]))                
            start += step
            end += step
            
        start = 0
        end = start + L
        while end<=attd_env_te.shape[0]:
            score_attn_te.append(pearson_scorer(model, eeg_te[start:end], attd_env_te[start:end]))
            score_unattn_te.append(pearson_scorer(model, eeg_te[start:end], unattd_env_te[start:end]))                
            start += step
            end += step            
        
        for i in attd_evt_te:
            start = i-int((w-eeg_context)*sr/2)
            end = start + L
            if end<=attd_env_te.shape[0]:
                score_attn_te_evt.append(pearson_scorer(model, eeg_te[start:end], attd_env_te[start:end]))
                score_unattn_te_evt.append(pearson_scorer(model, eeg_te[start:end], unattd_env_te[start:end]))
            
        score_attn_tr = np.array(score_attn_tr)   
        score_unattn_tr = np.array(score_unattn_tr)
        score_attn_te = np.array(score_attn_te)
        score_unattn_te = np.array(score_unattn_te)
        score_attn_te_evt = np.array(score_attn_te_evt)
        score_unattn_te_evt = np.array(score_unattn_te_evt)        

        
        train_accs.append((score_attn_tr>=score_unattn_tr).astype(float).mean())
        test_accs.append((score_attn_te>=score_unattn_te).astype(float).mean())        
        test_accs_evt.append((score_attn_te_evt>=score_unattn_te_evt).astype(float).mean())   
            
    logger.info(f'train_accs: {train_accs}')
    logger.info(f'test_accs: {test_accs}') 
    logger.info(f'test_accs_evt: {test_accs_evt}') 
    return (np.array(train_accs), np.array(test_accs_evt))
      
def trainSubjecIndependent(config, jobname):
    setup_params = config['setup']
    name = setup_params['name']
    output_path = os.path.abspath(setup_params['output_path'])
    trainModel = setup_params['trainModel']
    output_path = os.path.join(output_path, name)
    os.makedirs(output_path, exist_ok=True)

    dataset_params = config['dataset']
    data_folder = os.path.expandvars(dataset_params['folder'])
    data_files = dataset_params['pre_processed']
    for i in range(len(data_files)):
        path = os.path.join(data_folder, data_files[i])
        data_files[i] = path
    channels = dataset_params['channels']
    sr = dataset_params['sr']
    NUM_SBJS = dataset_params['num_sbjs']
    all_sbjs = np.array(dataset_params['pretrained_sbjs'])
    from_sbj = dataset_params['from_sbj']
    to_sbj = dataset_params['to_sbj']    
    T = dataset_params['T'] # seconds
    L = int(sr*T) # sample
    max_len = dataset_params['max_len'] # events
    n_streams = dataset_params['n_streams']    
    eeg_context = round(dataset_params['eeg_context']*sr) + 1
    model_params = config['model']
    
    # scaler for dataset
    scaler_path = dataset_params['scaler']['path']
    if scaler_path is not None:
        scaler_path = os.path.expandvars(scaler_path)
        if not os.path.exists(scaler_path):
            logger.info(f'Fitting scaler: {scaler_path}')
            if dataset_params['scaler']['type'] == 'MinMaxScaler':
                feature_range = tuple(dataset_params['scaler']['feature_range'])
                scaler = MinMaxScaler(feature_range=feature_range)
            elif dataset_params['scaler']['type'] == 'RobustScaler':
                scaler = RobustScaler(quantile_range=(5.0, 95.0))
            #
            eeg_all = []
            for s in all_sbjs:
                preload_data = loadmat(data_files[s], squeeze_me=True)
                eeg = np.concatenate(preload_data['eeg'], axis=0)
                eeg_all.append(eeg)
            eeg_all = np.concatenate(eeg_all, axis=0)
            scaler.fit_transform(eeg_all.reshape(-1,1))
            joblib.dump(scaler, scaler_path)
            del eeg_all
    
    train_accs = np.zeros((len(WINDOWS), NUM_SBJS))
    test_accs = np.zeros((len(WINDOWS), NUM_SBJS))
    train_F1 = np.zeros((len(WINDOWS), NUM_SBJS))
    test_F1 = np.zeros((len(WINDOWS), NUM_SBJS))
        
    for s in range(from_sbj, to_sbj):
        logger.info(f'{datetime.now().time().replace(microsecond=0)} --- '
                f'********** cross-training Sbj {s} **********')                
        trainset = []
        testset = []
        trained_sbjs = np.delete(all_sbjs, s)                       
        test_config = copy.deepcopy(dataset_params)
        test_config['pre_processed'] = [dataset_params['pre_processed'][s]]
        train_config = copy.deepcopy(dataset_params)
        train_config['pre_processed'] = [dataset_params['pre_processed'][i] for i in trained_sbjs]            
        for i in range(len(trained_sbjs)):
            preload_data = loadmat(data_files[trained_sbjs[i]], squeeze_me=True)
            trainset.append(getLinearEnvelopeData(config=train_config, loaded_data=preload_data, trial_idxs=None))
            del preload_data            
        preload_data = loadmat(data_files[s], squeeze_me=True)
        testset.append(getLinearEnvelopeData(config=test_config, loaded_data=preload_data, trial_idxs=None))
        del preload_data        
        train_accs[:,s], test_accs[:,s] = trainLinearEnvelope(model_params, trainset, testset, WINDOWS, sr, eeg_context)
        logger.info(f'sbj {s} valid_accs: {train_accs[...,s]}')
        logger.info(f'sbj {s} test_accs: {test_accs[...,s]}')    
    
    return train_accs, test_accs, train_F1, test_F1

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='Linear auditory attention classification based on stimulus envelope.')
    parser.add_argument("-j", "--jobname", type=str, required=True, help="Name of training entity.")
    parser.add_argument("-c", "--configs", type=str, required=True, nargs='+', help="Config file path.")
    parser.add_argument("-v", "--verbose", type=bool, default=False, help="Enable DEBUG verbose mode.")
    args = parser.parse_args()
    jobname = args.jobname
    configs = []
    for p in args.configs:
        with open(os.path.abspath(p)) as file:
            config = yaml.safe_load(file)
            file.close()
            configs.append(config)
    output_path = os.path.abspath(configs[0]['setup']['output_path'])
    logging.setup_logging(verbose=args.verbose, jobname=args.jobname, outpath=output_path)
    logger = logging.getLogger()
    
    all_SI_train_accs = [0]*len(configs)
    all_SI_test_accs = [0]*len(configs)
    all_SI_train_F1 = [0]*len(configs)
    all_SI_test_F1 = [0]*len(configs)
    model_names = []
        
    for i in range(len(configs)):
        model_names.append(configs[i]['model']['tag'])
        all_SI_train_accs[i], all_SI_test_accs[i] , _, _ = trainSubjecIndependent(configs[i], jobname)
        
    all_SI_train_accs = np.array(all_SI_train_accs).round(3)
    all_SI_test_accs = np.array(all_SI_test_accs).round(3)
    all_SI_train_F1 = np.array(all_SI_train_F1).round(3)
    all_SI_test_F1 = np.array(all_SI_test_F1).round(3)

    logger.info(f'all_SI_train_accs: {all_SI_train_accs}')
    logger.info(f'all_SI_test_accs: {all_SI_test_accs}')