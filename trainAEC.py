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

from eventaad.AEC import *
from eventaad.EEGModels import *
from eventaad.dataset import *
import eventaad.loss as L
from eventaad.loss import *
from parallel import *

global NUM_SBJS
NUM_SBJS = 24
    
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
    upsampling = dataset_params['upsampling']
    soft_label = dataset_params['soft_label']
    leave_one_out = dataset_params['leave_one_out']
    min_seed = dataset_params['min_seed']
    max_seed = dataset_params['max_seed']    
    L = dataset_params['L']
    channels = dataset_params['channels']
    channels_erp = dataset_params['channels_erp']
    sr = dataset_params['sr']
    NUM_SBJS = dataset_params['num_sbjs']
    start = dataset_params['start']
    end = dataset_params['end']
    start = int(start*sr/1000) # samples    
    end = int(end*sr/1000) # samples
    L = end - start # samples    

    model_params = config['model']
    
    learning_params = config['learning']
    optimizer_params = learning_params['optimizer']        
    loss_params = learning_params['loss_function'] 
    optimizer_params = learning_params['optimizer']
    running_params = learning_params['running']
    loss_params = learning_params['loss_function']
    threshold = learning_params['threshold']
    nFold = learning_params['nFold']
    
    batch_size = running_params['batch_size']
    num_workers = running_params['num_workers']
    epochs = running_params['epochs']
    parallelization = running_params['parallelization']
    
    print_every = running_params['print_every']
    devices = running_params['device']    
    lr = optimizer_params['lr']
    lr_decay_step = optimizer_params['lr_decay_step']
    lr_decay_gamma = optimizer_params['lr_decay_gamma']  
        
    # model
    lossClass = loss_params['name']
    criterion = eval(lossClass)()
    #
    model = eval(model_params['model_name'])(model_params, sr, start, end, channels, channels_erp, model_params['erp_forcing'], model_params['hybrid_training'])
    optimizer = Adam(model.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=lr_decay_step, gamma=lr_decay_gamma)
    erp_criterion = eval(loss_params['erp_loss'])()
    
    train_accs = np.zeros((nFold))
    test_accs = np.zeros((nFold))
    train_F1 = np.zeros((nFold))
    test_F1 = np.zeros((nFold))  
    thrhs = np.zeros((nFold))
    separated_accs = np.zeros((2, 15, nFold))
    separated_F1 = np.zeros((2, 15, nFold))
    if type(data_files) is list:
        loaded_data = []
        for f in data_files:
            path = os.path.join(data_folder, f)
            loaded_data.append(makeERPdata(path))
        if dataset_params['scaler']['type'] is not None:
            scaler_path = os.path.expandvars(dataset_params['scaler']['path'])
            if os.path.exists(scaler_path):
                scaler = joblib.load(scaler_path)
            else:
                X = []
                for data in loaded_data:
                    new_X = np.concatenate(data['X'])
                    X.append(new_X)
                X = 1e6*np.concatenate(X)
                data_shape = X.shape
                print(f'Raw data shape: {data_shape}')
                X = X.reshape(-1, 1)
                if dataset_params['scaler']['type'] == 'MinMaxScaler':
                    feature_range = tuple(dataset_params['scaler']['feature_range'])
                    scaler = eval(dataset_params['scaler']['type'])(feature_range=feature_range)
                else:
                    scaler = RobustScaler(quantile_range=(5.0, 95.0))   
                scaler.fit_transform(X)
                joblib.dump(scaler, scaler_path)                
                del X
                
    else:
        path = os.path.join(data_folder, data_files)
        loaded_data = makeERPdata(path)  
    pkl_filename = os.path.abspath('splits.pkl')
    if os.path.exists(pkl_filename):
        print(f'loading splits file: {pkl_filename}')
        # loading data:        
        with open(pkl_filename, 'rb') as f:
            splits = pickle.load(f)[0]
    else:
        print(f'creating splits file: {pkl_filename}')
        splits = make_splits(loaded_data, nFold)
        with open(pkl_filename, 'wb') as f:
            pickle.dump([splits], f)

    if parallelization == 'multi-fold':
        # multi-fold parallelization
        train_accs, test_accs, train_F1, test_F1, thrhs, separated_accs, separated_F1 = fold_parallel(devices, np.arange(nFold), loaded_data, scaler, splits, config, jobname)
    else:
        for fold in range(nFold):
            print(f'********** training - Fold {fold} **********')
            mixed_trainset = None
            mixed_validset = None
            mixed_testset = None
            if type(loaded_data) is list:
                trainset, validset, testset = get_mixed_splited_datasets(fold, loaded_data, splits, copy.deepcopy(dataset_params))
                trainset = MixedERPDataset(trainset, scaler)
                validset = MixedERPDataset(validset, scaler)
                testset = MixedERPDataset(testset, scaler)
                print(f'mixed trainset: {len(trainset)}')
                print(f'mixed validset: {len(validset)}')
                print(f'mixed testset: {len(testset)}')
            else:
                trainset, validset, testset = get_splited_datasets(fold, loaded_data, splits, dataset_params)
            
                # dataloader
                trainLoader = DataLoader(dataset=trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
                validLoader = DataLoader(dataset=validset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
                testLoader = DataLoader(dataset=testset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)       
                model_path = os.path.join(output_path, f"{model_params['model_name']}_SI_fold_{fold}.pth")
                if model_params['pretrained'] is not None:
                    model.pretrained = os.path.join(os.path.abspath(model_params['pretrained']), os.path.basename(model_path))
                else:
                    model.pretrained = None
                model.initialize()
                if trainModel:
                    if (type(devices) is list) and (len(devices) > 1):
                        fit_data_parallel(model, criterion, optimizer, scheduler, trainLoader, validLoader, epochs, threshold, devices, model_path=model_path, jobname=f'{jobname}_SI_fold_{fold}')
                    else:                
                        fit(model, criterion, erp_criterion, optimizer, scheduler, trainLoader, validLoader, epochs, threshold, devices, model_path=model_path, jobname=f'{jobname}_SI_fold_{fold}', print_every=1)
                # evaluate
                _,_,_, thrhs[fold] = evaluate(model, validLoader, validset.scaler, devices, criterion, sr, threshold=None, model_path=model_path, jobname=f'{jobname}_SI_fold_{fold}_valid', print_output=False)
                train_loss, train_accs[fold], train_F1[fold], threshold = evaluate(model, trainLoader, trainset.scaler, devices, criterion, sr, threshold=thrhs[fold], model_path=model_path, jobname=f'{jobname}_SI_fold_{fold}_train', print_output=False)
                test_loss, test_accs[fold], test_F1[fold], threshold = evaluate(model, testLoader, testset.scaler, devices, criterion, sr, threshold=thrhs[fold], model_path=model_path, jobname=f'{jobname}_SI_fold_{fold}_test', print_output=False)                
                if type(loaded_data) is list: # single DS evaluation
                    ds_config = copy.deepcopy(dataset_params)
                    ds_config['min_seed'] = 1
                    ds_config['max_seed'] = 1
                    ds_config['name'] = ['ExperimentalERPDataset']
                    _, _, mixed_testset1 = get_mixed_splited_datasets(fold, loaded_data, splits, ds_config)
                    ds_config['name'] = ['ExperimentalERPDataset']
                    ds_config['upsampling'] = False
                    _, _, mixed_testset2 = get_mixed_splited_datasets(fold, loaded_data, splits, ds_config)
                    for ds1, ds2, idx in zip(mixed_testset1, mixed_testset2, range(len(mixed_testset1))):
                        loader1 = DataLoader(dataset=ds1, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
                        loader2 = DataLoader(dataset=ds2, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
                        _, separated_accs[0,idx,fold], separated_F1[0,idx,fold],_ = evaluate(model, loader1, ds1.scaler, devices, criterion, sr, threshold=thrhs[fold], model_path=model_path, jobname=f'{jobname}_SI_ds_{idx}_fold_{fold}', print_output=False)
                        _, separated_accs[1,idx,fold], separated_F1[1,idx,fold],_ = evaluate(model, loader2, ds2.scaler, devices, criterion, sr, threshold=thrhs[fold], model_path=model_path, jobname=f'{jobname}_SI_ds_{idx}_fold_{fold}', print_output=False, weighted=True)          
                        del ds1, ds2                        
            del trainset, validset, testset, mixed_trainset, mixed_validset, mixed_testset
    print(f'train_accs: {train_accs}')
    print(f'test_accs: {test_accs}')
    print(f'train_F1: {train_F1}')
    print(f'test_F1: {test_F1}')
    print(f'thresholds: {thrhs}')
    print(f'dataset accs: {separated_accs}')
    print(f'dataset F1: {separated_F1}')
    
    return train_accs, test_accs, separated_accs, train_F1, test_F1, separated_F1
    
def trainAndCrossValidate(config, jobname):
    setup_params = config['setup']
    name = setup_params['name']
    output_path = os.path.abspath(setup_params['output_path'])
    trainModel = setup_params['trainModel']
    output_path = os.path.join(output_path, name)
    os.makedirs(output_path, exist_ok=True)
    
    dataset_params = config['dataset']
    data_folder = os.path.expandvars(dataset_params['folder'])
    data_files = dataset_params['pre_processed']
    # scaler_path = os.path.expandvars(dataset_params['scaler_path']) if dataset_params['scaler_path'] != None else None
    upsampling = dataset_params['upsampling']
    soft_label = dataset_params['soft_label']
    leave_one_out = dataset_params['leave_one_out']
    min_seed = dataset_params['min_seed']
    max_seed = dataset_params['max_seed']    
    L = dataset_params['L']
    channels = dataset_params['channels']
    channels_erp = dataset_params['channels_erp']
    sr = dataset_params['sr']
    NUM_SBJS = dataset_params['num_sbjs']
    from_sbj = dataset_params['from_sbj']
    to_sbj = dataset_params['to_sbj']
    start = dataset_params['start']
    end = dataset_params['end']
    start = int(start*sr/1000) # samples    
    end = int(end*sr/1000) # samples
    L = end - start # samples    

    model_params = config['model']
    
    learning_params = config['learning']
    optimizer_params = learning_params['optimizer']        
    loss_params = learning_params['loss_function'] 
    optimizer_params = learning_params['optimizer']
    running_params = learning_params['running']
    loss_params = learning_params['loss_function']
    threshold = learning_params['threshold']
    nFold = learning_params['nFold']
    
    batch_size = running_params['batch_size']
    num_workers = running_params['num_workers']
    epochs = running_params['epochs']
    
    print_every = running_params['print_every']
    devices = running_params['device']
    device = devices[0] if (type(devices) is list) else devices
    lr = optimizer_params['lr']
    lr_decay_step = optimizer_params['lr_decay_step']
    lr_decay_gamma = optimizer_params['lr_decay_gamma']    
    # model
    lossClass = loss_params['name']
    criterion = eval(lossClass)()
    #
    model = eval(model_params['model_name'])(model_params, sr, start, end, channels, channels_erp, model_params['erp_forcing'], model_params['hybrid_training'])
    optimizer = Adam(model.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=lr_decay_step, gamma=lr_decay_gamma)
    erp_criterion = eval(loss_params['erp_loss'])()
    
    train_accs = np.zeros(NUM_SBJS)
    test_accs = np.zeros(NUM_SBJS)
    train_F1 = np.zeros(NUM_SBJS)
    test_F1 = np.zeros(NUM_SBJS)
    thrhs = np.zeros(NUM_SBJS)
    separated_accs = np.zeros((2, 15, NUM_SBJS))
    separated_F1 = np.zeros((2, 15, NUM_SBJS))    
    if type(data_files) is list:
        loaded_data = []
        for f in data_files:
            path = os.path.join(data_folder, f)
            loaded_data.append(makeERPdata(path))
        if dataset_params['scaler']['type'] is not None:
            scaler_path = os.path.expandvars(dataset_params['scaler']['path'])
            if os.path.exists(scaler_path):
                scaler = joblib.load(scaler_path)
            else:
                X = []
                for data in loaded_data:
                    new_X = np.concatenate(data['X'])
                    X.append(new_X)
                X = 1e6*np.concatenate(X)
                data_shape = X.shape
                print(f'Raw data shape: {data_shape}')
                X = X.reshape(-1, 1)
                if dataset_params['scaler']['type'] == 'MinMaxScaler':
                    feature_range = tuple(dataset_params['scaler']['feature_range'])
                    scaler = eval(dataset_params['scaler']['type'])(feature_range=feature_range)
                else:
                    scaler = RobustScaler(quantile_range=(5.0, 95.0))   
                scaler.fit_transform(X)
                joblib.dump(scaler, scaler_path)                
                del X
                
    else:
        path = os.path.join(data_folder, data_files)
        loaded_data = makeERPdata(path)
    pkl_filename = os.path.abspath('splits.pkl')        
    print(f'splits filename: {pkl_filename}')
    if os.path.exists(pkl_filename):
        print(f'loading splits file: {pkl_filename}')
        # loading data:        
        with open(pkl_filename, 'rb') as f:
            splits = pickle.load(f)[0]   
    else:
        print(f'creating splits file: {pkl_filename}')
        splits = make_splits(loaded_data, nFold)
        with open(pkl_filename, 'wb') as f:
            pickle.dump([splits], f)
    
    for i in range(from_sbj, to_sbj):            
        print(f'********** training - cross subject {i} **********')
        train_idxs = []
        test_idxs = []
        for j in range(NUM_SBJS):
            if (i%NUM_SBJS)!=j and ((i+1)%NUM_SBJS)!=j:
                train_idxs.append(j)
            else:
                test_idxs.append(j)
        if type(loaded_data) is list:
            trs, vs, ts = get_mixed_splited_datasets(0, loaded_data, splits, copy.deepcopy(dataset_params), train_idxs)
            trainset = MixedERPDataset(trs, scaler)
            validset = MixedERPDataset(vs+ts, scaler)
            trs, vs, ts = get_mixed_splited_datasets(0, loaded_data, splits, copy.deepcopy(dataset_params), test_idxs)
            testset = MixedERPDataset(trs+vs+ts, scaler)
            print(f'mixed trainset: {len(trainset)}')
            print(f'mixed validset: {len(validset)}')
            print(f'mixed testset: {len(testset)}')
        else:
            trainset, vs, ts = get_splited_datasets(0, loaded_data, splits, dataset_params, train_idxs)
            validset = MixedERPDataset([vs,ts], scaler)
            trs, vs, ts = get_splited_datasets(0, loaded_data, splits, dataset_params, test_idxs)
            testset = MixedERPDataset([trs,vs,ts], scaler)
        
            # dataloader
            trainLoader = DataLoader(dataset=trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
            validLoader = DataLoader(dataset=validset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
            testLoader = DataLoader(dataset=testset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)       
            model_path = os.path.join(output_path, f"{model_params['model_name']}_CS_{i}.pth")
            if model_params['pretrained'] is not None:
                model.pretrained = os.path.join(os.path.abspath(model_params['pretrained']), os.path.basename(model_path))
            else:
                model.pretrained = None                
            model.initialize()
            if trainModel:
                fit(model, criterion, erp_criterion, optimizer, scheduler, trainLoader, validLoader, epochs, threshold, device, model_path=model_path, jobname=f'{jobname}_CS_{i}', print_every=1)
            # evaluate
            _,_,_, thrhs[i] = evaluate(model, validLoader, validset.scaler, device, criterion, sr, threshold=None, model_path=model_path, jobname=f'{jobname}_CS_{i}_valid', print_output=False)
            train_loss, train_accs[i], train_F1[i], threshold = evaluate(model, trainLoader, trainset.scaler, device, criterion, sr, threshold=thrhs[i], model_path=model_path, jobname=f'{jobname}_CS_{i}_train', print_output=False)
            test_loss, test_accs[i], test_F1[i], threshold = evaluate(model, testLoader, testset.scaler, device, criterion, sr, threshold=thrhs[i], model_path=model_path, jobname=f'{jobname}_CS_{i}_test', print_output=False)
            if type(loaded_data) is list: # single DS evaluation
                ds_config = copy.deepcopy(dataset_params)
                ds_config['min_seed'] = 1
                ds_config['max_seed'] = 1
                ds_config['name'] = ['ExperimentalERPDataset']
                mixed_trainset, mixed_validset, mixed_testset = get_mixed_splited_datasets(0, loaded_data, splits, ds_config, test_idxs)
                for tr, v, t, idx in zip(mixed_trainset, mixed_validset, mixed_testset, range(len(mixed_validset))):
                    ds = MixedERPDataset([tr,v,t], scaler)
                    loader = DataLoader(dataset=ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
                    _, separated_accs[0,idx,i], separated_F1[0,idx,i],_ = evaluate(model, loader, ds.scaler, device, criterion, sr, threshold=thrhs[i], model_path=model_path, jobname=f'{jobname}_CS_{i}_ds_{idx}', print_output=False)
                    del ds
                ds_config['name'] = ['ExperimentalERPDataset']
                ds_config['upsampling'] = False                    
                mixed_trainset, mixed_validset, mixed_testset = get_mixed_splited_datasets(0, loaded_data, splits, ds_config, test_idxs)
                for tr, v, t, idx in zip(mixed_trainset, mixed_validset, mixed_testset, range(len(mixed_validset))):
                    ds = MixedERPDataset([tr,v,t], scaler)
                    loader = DataLoader(dataset=ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
                    _, separated_accs[1,idx,i], separated_F1[1,idx,i],_ = evaluate(model, loader, ds.scaler, device, criterion, sr, threshold=thrhs[i], model_path=model_path, jobname=f'{jobname}_CS_{i}_ds_{idx}', print_output=False, weighted=True)
                    del ds                        
                del mixed_trainset, mixed_validset, mixed_testset, trs, vs, ts
        del trainset, validset, testset    
        print(f'train_accs: {train_accs}')
        print(f'test_accs: {test_accs}')
        print(f'train_F1: {train_F1}')
        print(f'test_F1: {test_F1}')
        print(f'thresholds: {thrhs}')
        print(f'dataset accs: {separated_accs}')
        print(f'dataset F1: {separated_F1}')
    
    return train_accs, test_accs, separated_accs, train_F1, test_F1, separated_F1
    
def trainSubjecSpecific(config, jobname):
    setup_params = config['setup']
    name = setup_params['name']
    output_path = os.path.abspath(setup_params['output_path'])
    trainModel = setup_params['trainModel']
    output_path = os.path.join(output_path, name)
    os.makedirs(output_path, exist_ok=True)
    
    dataset_params = config['dataset']
    data_folder = os.path.expandvars(dataset_params['folder'])
    data_files = dataset_params['pre_processed']
    # scaler_path = os.path.expandvars(dataset_params['scaler_path']) if dataset_params['scaler_path'] != None else None
    upsampling = dataset_params['upsampling']
    soft_label = dataset_params['soft_label']
    leave_one_out = dataset_params['leave_one_out']
    min_seed = dataset_params['min_seed']
    max_seed = dataset_params['max_seed']    
    L = dataset_params['L']
    channels = dataset_params['channels']
    channels_erp = dataset_params['channels_erp']
    sr = dataset_params['sr']
    NUM_SBJS = dataset_params['num_sbjs']
    from_sbj = dataset_params['from_sbj']
    to_sbj = dataset_params['to_sbj']    
    start = dataset_params['start']
    end = dataset_params['end']
    start = int(start*sr/1000) # samples    
    end = int(end*sr/1000) # samples
    L = end - start # samples    

    model_params = config['model']
    
    learning_params = config['learning']
    optimizer_params = learning_params['optimizer']        
    loss_params = learning_params['loss_function'] 
    optimizer_params = learning_params['optimizer']
    running_params = learning_params['running']
    loss_params = learning_params['loss_function']
    threshold = learning_params['threshold']
    nFold = learning_params['nFold']
    
    batch_size = running_params['batch_size']
    num_workers = running_params['num_workers']
    epochs = running_params['epochs']
    
    print_every = running_params['print_every']
    devices = running_params['device'] 
    device = devices[0] if (type(devices) is list) else devices    
    lr = optimizer_params['lr']
    lr_decay_step = optimizer_params['lr_decay_step']
    lr_decay_gamma = optimizer_params['lr_decay_gamma']    
    # model
    lossClass = loss_params['name']
    criterion = eval(lossClass)()
    model = eval(model_params['model_name'])(model_params, sr, start, end, channels, channels_erp, model_params['erp_forcing'], model_params['hybrid_training'])
    optimizer = Adam(model.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=lr_decay_step, gamma=lr_decay_gamma) 
    erp_criterion = eval(loss_params['erp_loss'])()
    
    train_accs = np.zeros((NUM_SBJS, nFold))
    test_accs = np.zeros((NUM_SBJS, nFold))
    train_F1 = np.zeros((NUM_SBJS, nFold))
    test_F1 = np.zeros((NUM_SBJS, nFold))  
    thrhs = np.zeros((NUM_SBJS, nFold))
    separated_accs = np.zeros((2, 15, NUM_SBJS, nFold))
    separated_F1 = np.zeros((2, 15, NUM_SBJS, nFold)) 
    
    if type(data_files) is list:
        loaded_data = []
        for f in data_files:
            path = os.path.join(data_folder, f)
            loaded_data.append(makeERPdata(path))
        if dataset_params['scaler']['type'] is not None:
            scaler_path = os.path.expandvars(dataset_params['scaler']['path'])
            if os.path.exists(scaler_path):
                scaler = joblib.load(scaler_path)
            else:
                X = []
                for data in loaded_data:
                    new_X = np.concatenate(data['X'])
                    X.append(new_X)
                X = 1e6*np.concatenate(X)
                data_shape = X.shape
                print(f'Raw data shape: {data_shape}')
                X = X.reshape(-1, 1)
                if dataset_params['scaler']['type'] == 'MinMaxScaler':
                    feature_range = tuple(dataset_params['scaler']['feature_range'])
                    scaler = eval(dataset_params['scaler']['type'])(feature_range=feature_range)
                else:
                    scaler = RobustScaler(quantile_range=(5.0, 95.0))   
                scaler.fit_transform(X)
                joblib.dump(scaler, scaler_path)                
                del X                
    else:
        path = os.path.join(data_folder, data_files)
        loaded_data = makeERPdata(path)
    pkl_filename = os.path.abspath('splits.pkl')        
    print(f'splits filename: {pkl_filename}')
    if os.path.exists(pkl_filename):
        print(f'loading splits file: {pkl_filename}')
        # loading data:        
        with open(pkl_filename, 'rb') as f:
            splits = pickle.load(f)[0]  
    else:
        print(f'creating splits file: {pkl_filename}')
        splits = make_splits(loaded_data, nFold)
        with open(pkl_filename, 'wb') as f:
            pickle.dump([splits], f)
    
    for i in range(from_sbj, to_sbj):
        for fold in range(nFold):
            print(f'********** training - Subject {i}, Fold {fold} **********')
            if type(loaded_data) is list:
                trainset, validset, testset = get_mixed_splited_datasets(fold, loaded_data, splits, copy.deepcopy(dataset_params), [i])
                trainset = MixedERPDataset(trainset, scaler)
                validset = MixedERPDataset(validset, scaler)
                testset = MixedERPDataset(testset, scaler)
                print(f'mixed trainset: {len(trainset)}')
                print(f'mixed validset: {len(validset)}')
                print(f'mixed testset: {len(testset)}')
            else:
                trainset, validset, testset = get_splited_datasets(fold, loaded_data, splits, dataset_params, [i])
            
                # dataloader
                trainLoader = DataLoader(dataset=trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
                validLoader = DataLoader(dataset=validset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
                testLoader = DataLoader(dataset=testset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)       
                model_path = os.path.join(output_path, f"{model_params['model_name']}_SS_{i}_fold_{fold}.pth")
                if model_params['pretrained'] is not None:
                    model.pretrained = os.path.join(os.path.abspath(model_params['pretrained']), os.path.basename(model_path))
                else:
                    model.pretrained = None                
                model.initialize()
                if trainModel:
                    fit(model, criterion, erp_criterion, optimizer, scheduler, trainLoader, validLoader, epochs, threshold, device, model_path=model_path, jobname=f'{jobname}_SS_{i}_fold_{fold}', print_every=1)
                # evaluate
                _,_,_, thrhs[i, fold] = evaluate(model, validLoader, validset.scaler, device, criterion, sr, threshold=None, model_path=model_path, jobname=f'{jobname}_SS_{i}_fold_{fold}_valid', print_output=True)
                train_loss, train_accs[i, fold], train_F1[i, fold], threshold = evaluate(model, trainLoader, trainset.scaler, device, criterion, sr, threshold=thrhs[i, fold], model_path=model_path, jobname=f'{jobname}_SS_{i}_fold_{fold}_train', print_output=True)
                test_loss, test_accs[i, fold], test_F1[i, fold], threshold = evaluate(model, testLoader, testset.scaler, device, criterion, sr, threshold=thrhs[i, fold], model_path=model_path, jobname=f'{jobname}_SS_{i}_fold_{fold}_test', print_output=True)
                
                if type(loaded_data) is list: # single DS evaluation
                    ds_config = copy.deepcopy(dataset_params)
                    ds_config['min_seed'] = 1
                    ds_config['max_seed'] = 1
                    ds_config['name'] = ['ExperimentalERPDataset']
                    _, _, mixed_testset1 = get_mixed_splited_datasets(fold, loaded_data, splits, ds_config, [i])
                    ds_config['name'] = ['ExperimentalERPDataset']
                    ds_config['upsampling'] = False
                    _, _, mixed_testset2 = get_mixed_splited_datasets(fold, loaded_data, splits, ds_config, [i])
                    for ds1, ds2, idx in zip(mixed_testset1, mixed_testset2, range(len(mixed_testset1))):
                        loader1 = DataLoader(dataset=ds1, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
                        loader2 = DataLoader(dataset=ds2, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
                        _, separated_accs[0,idx, i, fold], separated_F1[0,idx,i,fold],_ = evaluate(model, loader1, ds1.scaler, devices, criterion, sr, threshold=thrhs[i, fold], model_path=model_path, jobname=f'{jobname}_SI_ds_{idx}_fold_{fold}', print_output=False)
                        _, separated_accs[1,idx, i, fold], separated_F1[1,idx,i,fold],_ = evaluate(model, loader2, ds2.scaler, devices, criterion, sr, threshold=thrhs[i, fold], model_path=model_path, jobname=f'{jobname}_SI_ds_{idx}_fold_{fold}', print_output=False, weighted=True)          
                        del ds1, ds2                      
                
            del trainset, validset, testset
        print(f'train_accs: {np.mean(train_accs, -1, keepdims=False)[i]}')
        print(f'test_accs: {np.mean(test_accs, -1, keepdims=False)[i]}')
        print(f'train_F1: {np.mean(train_F1, -1, keepdims=False)[i]}')
        print(f'test_F1: {np.mean(test_F1, -1, keepdims=False)[i]}')
        print(f'dataset accs: {np.mean(separated_accs, -1, keepdims=False)[...,i]}')
        print(f'dataset F1: {np.mean(separated_F1, -1, keepdims=False)[...,i]}')
    train_accs = np.mean(train_accs, -1, keepdims=False)
    test_accs = np.mean(test_accs, -1, keepdims=False)
    train_F1 = np.mean(train_F1, -1, keepdims=False)
    test_F1 = np.mean(test_F1, -1, keepdims=False)   
    separated_accs = np.mean(separated_accs, -1, keepdims=False)  
    separated_F1 = np.mean(separated_F1, -1, keepdims=False)  
    thrhs = np.mean(thrhs, -1, keepdims=False)
    
    print(f'train_accs: {train_accs}')
    print(f'test_accs: {test_accs}')
    print(f'train_F1: {train_F1}')
    print(f'test_F1: {test_F1}')
    print(f'thresholds: {thrhs}')
    print(f'dataset accs: {separated_accs}')
    print(f'dataset F1: {separated_F1}')    
    
    return train_accs, test_accs, separated_accs, train_F1, test_F1, separated_F1

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='Auditory attention event classifier training.')
    parser.add_argument("--jobname", type=str, required=True, help="Name of training entity.")
    parser.add_argument("--configs", type=str, required=True, nargs='+', help="Config file path.")
    args = parser.parse_args()
    jobname = args.jobname
    configs = []
    for p in args.configs:
        with open(os.path.abspath(p)) as file:
            config = yaml.safe_load(file)
            file.close()
            configs.append(config)
    output_path = os.path.abspath(configs[0]['setup']['output_path'])
    
    all_SI_train_accs = [0]*len(configs)
    all_SI_test_accs = [0]*len(configs)
    all_SI_ds_accs = [0]*len(configs)
    all_SI_train_F1 = [0]*len(configs)
    all_SI_test_F1 = [0]*len(configs)
    all_SI_ds_F1 = [0]*len(configs)
    
    
    all_CS_train_accs = [0]*len(configs)
    all_CS_test_accs = [0]*len(configs)
    all_CS_ds_accs = [0]*len(configs)
    all_CS_train_F1 = [0]*len(configs)
    all_CS_test_F1 = [0]*len(configs)    
    all_CS_ds_F1 = [0]*len(configs)
    
    all_SS_train_accs = [0]*len(configs)
    all_SS_test_accs = [0]*len(configs)
    all_SS_ds_accs = [0]*len(configs)
    all_SS_train_F1 = [0]*len(configs)
    all_SS_test_F1 = [0]*len(configs)
    all_SS_ds_F1 = [0]*len(configs)
    model_names = []
        
    for i in range(len(configs)):
        model_names.append(configs[i]['model']['tag'])
        
        all_SI_train_accs[i], all_SI_test_accs[i], all_SI_ds_accs[i], all_SI_train_F1[i], all_SI_test_F1[i], all_SI_ds_F1[i] = trainSubjecIndependent(configs[i], jobname)
        
        # all_CS_train_accs[i], all_CS_test_accs[i], all_CS_ds_accs[i], all_CS_train_F1[i], all_CS_test_F1[i], all_CS_ds_F1[i] = trainAndCrossValidate(configs[i], jobname)
        
        # all_SS_train_accs[i], all_SS_test_accs[i], all_SS_ds_accs[i], all_SS_train_F1[i], all_SS_test_F1[i], all_SS_ds_F1[i] = trainSubjecSpecific(configs[i], jobname)
        
    all_SI_train_accs = np.array(all_SI_train_accs).round(3)
    all_SI_test_accs = np.array(all_SI_test_accs).round(3)
    all_SI_train_F1 = np.array(all_SI_train_F1).round(3)
    all_SI_test_F1 = np.array(all_SI_test_F1).round(3)
    all_SI_ds_accs = np.array(all_SI_ds_accs).round(3)
    all_SI_ds_F1 = np.array(all_SI_ds_F1).round(3)

    
    all_CS_train_accs = np.array(all_CS_train_accs).round(3)
    all_CS_test_accs = np.array(all_CS_test_accs).round(3)
    all_CS_train_F1 = np.array(all_CS_train_F1).round(3)
    all_CS_test_F1 = np.array(all_CS_test_F1).round(3)    
    all_CS_ds_accs = np.array(all_CS_ds_accs).round(3)
    all_CS_ds_F1 = np.array(all_CS_ds_F1).round(3)
    
    all_SS_train_accs = np.array(all_SS_train_accs).round(3)
    all_SS_test_accs = np.array(all_SS_test_accs).round(3)
    all_SS_train_F1 = np.array(all_SS_train_F1).round(3)
    all_SS_test_F1 = np.array(all_SS_test_F1).round(3)
    all_SS_ds_accs = np.array(all_SS_ds_accs).round(3)
    all_SS_ds_F1 = np.array(all_SS_ds_F1).round(3)    

    all_SI_accs = np.concatenate([np.expand_dims(all_SI_test_accs, axis=1), all_SI_ds_accs[:,0,0:3,:]], axis=1)
    # all_CS_accs = np.concatenate([np.expand_dims(all_CS_test_accs, axis=1), all_CS_ds_accs[:,0,0:3,:]], axis=1)
    # all_SS_accs = np.concatenate([np.expand_dims(all_SS_test_accs, axis=1), all_SS_ds_accs[:,0,0:3,:]], axis=1)
    
    y_label = 'Accuracy'
    xtick_labels = ['Mixed','Prdm. 1','Prdm. 2', 'Prdm. 3']

    title = 'Subject-pooled classification performance'
    save_path = os.path.join(output_path, "fig3_SI_performance.png")
    plot_compare_bar_withSTDbar(all_SI_accs, bar_labels=model_names, xtick_labels=xtick_labels, y_label=y_label, title=title, save_path=save_path)

    # title = 'Leave-two-subjects-out classification performance'
    # save_path = os.path.join(output_path, "fig4_CS_performance.png")
    # plot_compare_bar_withSTDbar(all_CS_accs, bar_labels=model_names, xtick_labels=xtick_labels, y_label=y_label, title=title, save_path=save_path)
    
    # title = 'Individual classification performance'
    # save_path = os.path.join(output_path, "fig5_SS_performance.png")
    # plot_compare_bar_withSTDbar(all_SS_accs, bar_labels=model_names, xtick_labels=xtick_labels, y_label=y_label, title=title, save_path=save_path)
    