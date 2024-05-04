import os
from datetime import datetime, timedelta
import math
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
torch.multiprocessing.set_sharing_strategy('file_system')

from eventaad.AEC import *
from eventaad.utils import *
from eventaad.loss import *
from running import *

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("gloo", rank=rank, world_size=world_size, timeout=timedelta(days=1))
    
def cleanup():
    dist.destroy_process_group()
    
def prepare(rank, world_size, dataset, batch_size=32, pin_memory=False, num_workers=0):
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)    
    dataloader = DataLoader(dataset, batch_size=batch_size, pin_memory=pin_memory, num_workers=num_workers, drop_last=False, shuffle=False, sampler=sampler)
    
    return dataloader    

def data_job(rank, world_size, model, criterion, optimizer, scheduler, trainLoader, validLoader, epochs, threshold, devices, model_path, jobname):
    device = devices[rank]
    if device != 'cpu':
        torch.cuda.set_device(device)
    print(f'Running on device: {device}')
    # setup the process groups
    setup(rank, world_size)
    # prepare the dataloader    
    trainLoader = prepare(rank, world_size, trainLoader.dataset, trainLoader.batch_size)
    
    # model
    model = model.to(rank)
    model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=True)
    train_losses, train_accs = [], []
    valid_losses, valid_accs = [], []    
    best_loss_train = -1.0
    best_loss_valid = -1.0
    best_accr_train = 0.0
    best_accr_valid = 0.0
    print(f'Started process: {rank}, world_size: {world_size}, train data: {len(trainLoader)*trainLoader.batch_size}, valid data: {len(validLoader.dataset)}')
    epoch_loss, epoch_acc,_,_ = evaluate(model, validLoader, None, device, criterion, None, threshold, None, jobname, print_output=False)
    print(f'{datetime.now().time().replace(microsecond=0)} --- device {device}:'
          f'Epoch: -1\t'
          f'Valid loss: {epoch_loss:.8f}\t'
          f'Valid accuracy: {100 * epoch_acc:.2f}')     
    for epoch in range(epochs):
        trainLoader.sampler.set_epoch(epoch)
        #train
        model.train()
        cur_loss = 0
        cur_acc = 0
        all_y_hat = []
        all_y_true = []
        data_count = 0
        for (X, erp, y_true, _) in trainLoader:
            optimizer.zero_grad(set_to_none=True)
            batch_size = y_true.shape[0]
            data_count += batch_size
            X = X.to(device)
            erp = erp.to(device)
            y_true = y_true.to(device)
            if isinstance(criterion, DualTask_AE_Loss):
                X_pred, erp_hat, y_hat = model(X, erp, y_true)
                loss = criterion(X_pred, y_hat, X, y_true)
            else:
                erp_hat, y_hat = model(X, erp, y_true)
                loss = criterion(y_hat, y_true)
                
            cur_loss += loss.item()*batch_size
            cur_acc += binary_accuracy(y_hat.data,y_true.data,threshold).item()*batch_size
            all_y_hat.append(y_hat.data)
            all_y_true.append(y_true.data)            
            loss.backward()
            optimizer.step()
            del X, erp, y_true, loss, erp_hat, y_hat
        scheduler.step()
        epoch_loss = cur_loss/data_count
        train_losses.append(epoch_loss)
        epoch_acc = cur_acc/data_count
        train_accs.append(epoch_acc)
        (TP,FP,TN,FN,_,_) = metrics(torch.cat(all_y_hat), torch.cat(all_y_true), thresh=threshold)
        del all_y_hat, all_y_true    
        # evaluate on validation set
        epoch_loss, epoch_acc,_,_ = evaluate(model, validLoader, None, device, criterion, None, threshold, None, jobname, print_output=False)
        valid_losses.append(epoch_loss)
        valid_accs.append(epoch_acc)
        torch.cuda.empty_cache()
        if (valid_losses[-1] <= best_loss_valid) or best_loss_train<0:
            print(f'device {device}: Checkpoint saved at epoch {epoch}.')
            best_loss_train = train_losses[-1]
            best_loss_valid = valid_losses[-1]
            best_accr_train = train_accs[-1]
            best_accr_valid = valid_accs[-1]
            torch.save(model.module.state_dict(), model_path)
        if epoch % 1 == 0:
            print(f'{datetime.now().time().replace(microsecond=0)} --- device {device} '
                  f'Epoch: {epoch}\t'
                  f'Train loss: {train_losses[epoch]:.8f}\t'
                  f'Valid loss: {valid_losses[epoch]:.8f}\t'
                  f'Train accuracy: {100 * train_accs[epoch]:.2f}\t'
                  f'Valid accuracy: {100 * valid_accs[epoch]:.2f}') 
            print(f'\t\t\t TP: {TP} \t FP: {FP} \t TN: {TN} \t FN: {FN} --- F1: {2*TP/(2*TP+FP+FN):.5f}')     
    plt.clf()
    plt.plot(train_accs,'b-', label="train accuracy")
    plt.plot(valid_accs,'r.', label="validation accuracy")
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    filepath = os.path.join(os.path.dirname(model_path), f"{jobname}_accuracy_curve_device_{rank}.png") 
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    # plot loss    
    plt.clf()
    plt.plot(train_losses,'b-', label="train loss")
    plt.plot(valid_losses,'b.', label="validation loss")    
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    filepath = os.path.join(os.path.dirname(model_path), f"{jobname}_loss_curve_device_{rank}.png") 
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    #
    cleanup()
    
def fold_job(rank, world_size, devices, folds, loaded_data, scaler, splits, config, jobname, conn):
    fold = folds[rank]
    device = devices[rank]
    print(f'Running on device: {device}')
    # setup the process groups
    setup(rank, world_size)
    if device != 'cpu':
        torch.cuda.set_device(device)
    # config
    setup_params = config['setup']
    name = setup_params['name']
    trainModel = setup_params['trainModel']
    output_path = os.path.abspath(setup_params['output_path'])
    output_path = os.path.join(output_path, name)
    
    dataset_params = config['dataset']
    data_folder = os.path.expandvars(dataset_params['folder'])
    data_files = dataset_params['pre_processed']
    upsampling = dataset_params['upsampling']
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
    
    batch_size = running_params['batch_size']
    num_workers = running_params['num_workers']
    epochs = running_params['epochs']
    
    print_every = running_params['print_every']
    lr = optimizer_params['lr']
    lr_decay_step = optimizer_params['lr_decay_step']
    lr_decay_gamma = optimizer_params['lr_decay_gamma']  
               
    #
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
    # model
    if model_params['model_name'] not in ['CSP', 'TM', 'LDA']:
        lossClass = loss_params['name']
        criterion = eval(lossClass)()
        erp_criterion = eval(loss_params['erp_loss'])()    
        #
        model = eval(model_params['model_name'])(model_params, sr, start, end, channels, channels_erp, model_params['erp_forcing'], model_params['hybrid_training'])    
    if model_params['pretrained'] is not None:
        model.pretrained = os.path.join(os.path.abspath(model_params['pretrained']), os.path.basename(model_path))
    else:
        model.pretrained = None
    model.initialize()
    if trainModel:
        optimizer = Adam(model.parameters(), lr=lr)
        scheduler = StepLR(optimizer, step_size=lr_decay_step, gamma=lr_decay_gamma)     
        fit(model, criterion, erp_criterion, optimizer, scheduler, trainLoader, validLoader, epochs, threshold, device, model_path=model_path, jobname=f'{jobname}_SI_fold_{fold}', print_every=1)
    else:
        model_path = None
    # evaluate
    train_loss, train_accs, train_F1, thrhs = evaluate(model, validLoader, validset.scaler, device, criterion, sr, threshold=None, model_path=model_path, jobname=f'{jobname}_SI_fold_{fold}_valid', print_output=False)
    test_loss, test_accs, test_F1, threshold = evaluate(model, testLoader, testset.scaler, device, criterion, sr, threshold=thrhs, model_path=model_path, jobname=f'{jobname}_SI_fold_{fold}_test', print_output=False)
    # test on original dataset
    separated_accs = np.zeros((2, 12))
    separated_F1 = np.zeros((2, 12))
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
            _, separated_accs[0, idx], separated_F1[0, idx],_ = evaluate(model, loader1, ds1.scaler, device, criterion, sr, threshold=thrhs, model_path=model_path, jobname=f'{jobname}_SI_ds_{idx}_fold_{fold}', print_output=False)
            _, separated_accs[1, idx], separated_F1[1, idx],_ = evaluate(model, loader2, ds2.scaler, device, criterion, sr, threshold=thrhs, model_path=model_path, jobname=f'{jobname}_SI_ds_{idx}_fold_{fold}', print_output=False, weighted=True)          
            del ds1, ds2
    del trainset, validset, testset, mixed_trainset, mixed_validset, mixed_testset
    data = {
        'rank': rank,
        'device': device,
        'fold': fold,
        'train_accs': train_accs,
        'test_accs': test_accs,
        'train_F1': train_F1,
        'test_F1': test_F1,
        'thrhs': thrhs,
        'separated_accs': separated_accs,
        'separated_F1': separated_F1
    }
    print(f'sending {data}')
    conn.send(data)
    cleanup()
    
def fit_data_parallel(model, criterion, optimizer, scheduler, trainLoader, validLoader, epochs, threshold, devices, model_path, jobname):
    n_devices = len(devices)
    world_size = n_devices

    mp.spawn(
        data_job,
        args=(world_size, model, criterion, optimizer, scheduler, trainLoader, validLoader, epochs, threshold, devices, model_path, jobname),
        nprocs=world_size
    )    
    print('All child process finished!', flush=True)
    
def fold_parallel(devices, folds, loaded_data, scaler, splits, config, jobname):
    n_devices = len(devices)
    nFold = len(folds)
    nLoop = math.ceil(nFold/n_devices)
    train_accs = np.zeros((nFold))
    test_accs = np.zeros((nFold))
    train_F1 = np.zeros((nFold))
    test_F1 = np.zeros((nFold))  
    thrhs = np.zeros((nFold))
    separated_accs = np.zeros((2, 12, nFold))
    separated_F1 = np.zeros((2, 12, nFold))
    parent_conn, child_conn = mp.Pipe()
    for i in range(nLoop):
        world_size = min(n_devices, len(folds[i*n_devices:(i+1)*n_devices]))
        mp.spawn(
            fold_job,
            args=(world_size, devices, folds[i*n_devices:(i+1)*n_devices], loaded_data, scaler, splits, config, jobname, child_conn),
            nprocs=world_size
        )
        print('All child process finished!', flush=True)
        while parent_conn.poll():
            result = parent_conn.recv()
            fold = result['fold']
            fold_idx = np.where(folds==fold)[0][0]
            train_accs[fold_idx] = result['train_accs']
            test_accs[fold_idx] = result['test_accs']
            train_F1[fold_idx] = result['train_F1']
            test_F1[fold_idx] = result['test_F1']
            thrhs[fold_idx] = result['thrhs']
            separated_accs[...,fold_idx] = result['separated_accs']
            separated_F1[...,fold_idx] = result['separated_F1']   
            del result
    return train_accs, test_accs, train_F1, test_F1, thrhs, separated_accs, separated_F1
    
