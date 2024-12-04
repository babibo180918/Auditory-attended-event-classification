import os
import sys
import copy
from datetime import datetime
from sklearn.model_selection import KFold, train_test_split
import numpy as np
import torch
import torch.nn as nn
from torch.nn import BCELoss, MSELoss, CrossEntropyLoss, BCEWithLogitsLoss
from torch.optim import *
from torch.utils.data.dataset import random_split
from torch.utils.data import WeightedRandomSampler
from torch.optim.lr_scheduler import StepLR
from torch.profiler import profile, record_function, ProfilerActivity
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')


from eventaad.BaseNet import *
from eventaad.EEGModels import *
from eventaad.dataset import *
from eventaad.loss import *
from .utils import metrics, binary_accuracy


def make_splits(loaded_data, nFold):
    if type(loaded_data) is list:
        all_splits = []
        for j in range(len(loaded_data)):
            X = loaded_data[j]['X']
            # splitting data
            splits = []
            for i in range(len(X)):
                kf = KFold(n_splits=nFold, random_state=i, shuffle=True)
                split = [(train, test) for train, test in kf.split(X[i])]
                splits.append(split)
            all_splits.append(splits)
        return all_splits
    else:
        X = loaded_data['X']
        y = loaded_data['y']
        # splitting data
        splits = []
        for i in range(len(X)):
            kf = KFold(n_splits=nFold, random_state=i, shuffle=True)
            split = [(train, test) for train, test in kf.split(X[i])]
            splits.append(split)
        return splits
        
def get_splited_datasets(fold, loaded_data, splits, dataset_params, sbj_idxs=None):
    train_X = []
    train_y = []
    valid_X = []
    valid_y = []
    test_X = []
    test_y = []     
    X = loaded_data['X']
    y = loaded_data['y']
        
    if sbj_idxs is None:
        sbj_idxs = range(len(X))
    for i in sbj_idxs:
        test_X.append(X[i][splits[i][fold][1]])
        test_y.append(y[i][splits[i][fold][1]])
        trainX = X[i][splits[i][fold][0]]
        trainy = y[i][splits[i][fold][0]]
        trainX, validX, trainy, validy = train_test_split(trainX, trainy, random_state=i, test_size=0.2)# spliting train into train and validation set
        train_X.append(trainX)
        train_y.append(trainy) 
        valid_X.append(validX)
        valid_y.append(validy)                 
    loaded_data['X'] = train_X
    loaded_data['y'] = train_y
    trainset = eval(dataset_params['name'])(config=dataset_params, loaded_data=loaded_data)
    loaded_data['X'] = valid_X
    loaded_data['y'] = valid_y           
    validset = eval(dataset_params['name'])(config=dataset_params, loaded_data=loaded_data)            
    loaded_data['X'] = test_X
    loaded_data['y'] = test_y           
    testset = eval(dataset_params['name'])(config=dataset_params, loaded_data=loaded_data)
    
    loaded_data['X'] = X
    loaded_data['y'] = y    
    
    del X, y, train_X, valid_X, test_X, train_y, valid_y, test_y
    return trainset, validset, testset
    
def get_mixed_splited_datasets(fold, loaded_data, splits, dataset_params, sbj_idxs=None):
    trainset = []
    validset = []
    testset = []
    ds_types = dataset_params['name']
    SNRs = dataset_params['SNR']
    for i in range(len(loaded_data)):
        for ds_type in ds_types:
            dataset_params['name'] = ds_type
            if ds_type == 'SimulatedERPDataset':
                dataset_params['simulated'] = True
                for db in SNRs:
                    print(f'generating mixed dataset {ds_type}, {db} dB')
                    dataset_params['SNR'] = db
                    train, valid, test = get_splited_datasets(fold, loaded_data[i], splits[i], dataset_params, sbj_idxs)
                    trainset.append(train)
                    validset.append(valid)
                    testset.append(test)
                    del train, valid, test
            else:
                print(f'generating mixed dataset {ds_type}')
                dataset_params['simulated'] = False
                train, valid, test = get_splited_datasets(fold, loaded_data[i], splits[i], dataset_params, sbj_idxs)
                trainset.append(train)
                validset.append(valid)
                testset.append(test)            
                del train, valid, test
    return trainset, validset, testset

def evaluate(model, data_loader, scaler, device, criterion, sr=None, threshold=None, model_path=None, jobname=None, print_output=False, weighted=False):
    '''
    Calculating accuracy of model
    '''
    if model_path is not None:
        model.load_state_dict(torch.load(model_path))
    if type(device) is list:
        device = device[0]
    model.device = device
    model.to(device)
    model.eval()
    total_loss = 0.0
    total_erp_loss = 0.0
    all_y_hat = []
    all_y_true = []
    for (X, erp, y_true, _) in data_loader:
        batch_size = X.shape[0]
        X = X.to(device, dtype=torch.float)
        erp = erp.to(device, dtype=torch.float)
        y_true = y_true.to(device, dtype=torch.float)
        with torch.no_grad():
            if isinstance(criterion, DualTask_AE_Loss):
                X_pred, erp_hat, y_hat = model(X, erp, y_true)
                loss = criterion(X_pred, y_hat, X, y_true)
            else:
                erp_hat, y_hat = model(X, erp, y_true)
                loss = criterion(y_hat, y_true) 
            all_y_hat.append(y_hat.data)
            all_y_true.append(y_true.data)
        total_loss += loss.item()*batch_size
        del X, erp, y_true, loss, erp_hat, y_hat
        torch.cuda.empty_cache()
    (TP,FP,TN,FN,acc,threshold) = metrics(torch.cat(all_y_hat), torch.cat(all_y_true), thresh=threshold, weighted=weighted)
    avg_loss = total_loss/len(data_loader.dataset)
    avr_erp_loss = total_erp_loss/len(data_loader.dataset)
    avg_accr = acc.item()
    F1 = (2*TP/(2*TP+FP+FN)).item()
    
    if print_output:
        filename = os.path.join(os.path.dirname(model_path), f"{jobname}") 
        plot_confusion_matrix(torch.cat(all_y_true).cpu(), (torch.cat(all_y_hat)>threshold).cpu(), ['Unattended', 'Attended'], filename=filename+'_confusion_matrix.png')    
        X, erp, y_true, _ = next(iter(data_loader))
        X = X.to(device, dtype=torch.float)
        erp = erp.to(device, dtype=torch.float)
        y_true = y_true.to(device, dtype=torch.float)
        with torch.no_grad():
            if isinstance(criterion, DualTask_AE_Loss):
                X_pred, erp_hat, y_hat = model(X, erp, y_true)
            else:
                erp_hat, y_hat = model(X, erp, y_true)
        for i in range(10):
            filepath = os.path.join(os.path.dirname(model_path), f"{jobname}_output_{i}.png")
            model.plot_inout(X[i].cpu().detach().numpy(), y_true[i].cpu().detach().numpy(), y_hat[i].cpu().detach().numpy(), erp[i].cpu().detach().numpy(), scaler, filepath, sr, feature=erp_hat[i].cpu().detach().numpy())
        '''
        # T-SNE visualization
        X, erp, y, _ = data_loader.dataset[:]
        X = X.to(device)
        erp = erp.to(device)
        y = y.to(device)
        model(X,y)
        X = model.features
        if X.requires_grad:
            X = X.cpu().detach().data
        else:
            X = X.cpu().data
        y = y.cpu().data
        tsne_visualization(X, y, filename+'_tsne.png')
        '''
    return (avg_loss, avg_accr, F1, threshold)

def fit(model:BaseNet, criterion, erp_criterion, optimizer, lr_scheduler, train_loader, valid_loader, epochs, threshold, devices, model_path, jobname = None, print_every=10):
    train_losses, train_accs, train_erp_losses = [], [], []
    valid_losses, valid_accs, valid_erp_losses = [], [], []
    device = devices[0] if (type(devices) is list) else devices
    model.to(device)
    best_loss_train = -1.0
    best_accr_train = 0.0
    best_erp_loss_train = -1.0
    best_erp_loss_valid = -1.0
 
    best_loss_valid, best_accr_valid,_,_ = evaluate(model, valid_loader, None, device, criterion, None, threshold, None, jobname=jobname, print_output=False)
    print(f'device {device} {datetime.now().time().replace(microsecond=0)} --- '
          f'Epoch: -1\t'
          f'Valid loss: {best_loss_valid:.8f}\t'
          f'Valid accuracy: {100 * best_accr_valid:.2f}')
          
    for epoch in range(epochs):
        model.train()
        cur_loss = 0
        cur_erp_loss = 0
        cur_acc = 0
        all_y_hat = []
        all_y_true = []
        for (X, erp, y_true, _) in train_loader:
            # print_memory_info(device)
            if model.hybrid_training and model.feature_freeze:
                if (epoch%2):
                    model.freeze_feature_extractor(True)
                else:
                    model.freeze_feature_extractor(False)
            optimizer.zero_grad() # reset gradients
            batch_size = y_true.shape[0]
            X = X.to(device)
            erp = erp.to(device)
            y_true = y_true.to(device)
            if isinstance(criterion, DualTask_AE_Loss):
                X_pred, erp_hat, y_hat = model(X, erp, y_true)
                loss = criterion(X_pred, y_hat, X, y_true)
            else:
                erp_hat, y_hat = model(X, erp, y_true)
                loss = criterion(y_hat, y_true)
                
            erp_loss = erp_criterion(erp_hat, erp)
            cur_loss += loss.item()*batch_size
            cur_erp_loss += erp_loss.item()*batch_size
            cur_acc += binary_accuracy(y_hat.data,y_true.data,threshold).item()*batch_size
            all_y_hat.append(y_hat.data)
            all_y_true.append(y_true.data)
            
            if (model.hybrid_training) and (not(epoch%2)):
                erp_loss.backward()
            else:
                loss.backward()
            optimizer.step()
            del X, erp, y_true, loss, erp_loss, erp_hat, y_hat
            torch.cuda.empty_cache()
        
        lr_scheduler.step()
        epoch_loss = cur_loss/len(train_loader.dataset)
        train_losses.append(epoch_loss)
        epoch_erp_loss = cur_erp_loss/len(train_loader.dataset)
        train_erp_losses.append(epoch_erp_loss)
        epoch_acc = cur_acc/len(train_loader.dataset)
        train_accs.append(epoch_acc)
        (TP,FP,TN,FN,_,_) = metrics(torch.cat(all_y_hat), torch.cat(all_y_true), thresh=threshold)
        del all_y_hat, all_y_true
        
        model.eval()
        cur_loss = 0
        cur_erp_loss = 0
        cur_acc = 0
        for (X, erp, y_true, _) in valid_loader:
            batch_size = y_true.shape[0]
            X = X.to(device)
            erp = erp.to(device)
            y_true = y_true.to(device)    
            #y_true = y_true.float()
            if isinstance(criterion, DualTask_AE_Loss):
                X_pred, erp_hat, y_hat = model(X, erp, y_true)
                loss = criterion(X_pred, y_hat, X, y_true)
            else:
                erp_hat, y_hat = model(X, erp, y_true)
                loss = criterion(y_hat, y_true)
            erp_loss = erp_criterion(erp_hat, erp)
            cur_loss += loss.item()*batch_size
            cur_erp_loss += erp_loss.item()*batch_size
            cur_acc += binary_accuracy(y_hat.data,y_true.data,threshold).item()*batch_size
            del X, erp, y_true, loss, erp_loss, erp_hat, y_hat
            
        epoch_loss = cur_loss/len(valid_loader.dataset)
        valid_losses.append(epoch_loss)
        epoch_erp_loss = cur_erp_loss/len(valid_loader.dataset)
        valid_erp_losses.append(epoch_erp_loss)        
        epoch_acc = cur_acc/len(valid_loader.dataset)
        valid_accs.append(epoch_acc)

        if (valid_losses[-1] <= best_loss_valid):
            print(f'device {device} Checkpoint saved at epoch {epoch}.')
            best_loss_train = train_losses[-1]
            best_loss_valid = valid_losses[-1]
            best_accr_train = train_accs[-1]
            best_accr_valid = valid_accs[-1]
            torch.save(model.state_dict(), model_path)
            
        if (valid_erp_losses[-1] <= best_erp_loss_valid) or best_erp_loss_train<0:
            print(f'device {device} Feature extractor saved at epoch {epoch}.')
            best_erp_loss_train = train_erp_losses[-1]
            best_erp_loss_valid = valid_erp_losses[-1]
            head_tail = os.path.split(model_path)
            name, ext = os.path.splitext(head_tail[1])
            path = os.path.join(head_tail[0], f'{name}_erp{ext}')
            torch.save(model.feature_extractor.state_dict(), path)            
    
        if epoch % print_every == 0:
            print(f'device {device} {datetime.now().time().replace(microsecond=0)} --- '
                  f'Epoch: {epoch}\t'
                  f'Train loss: {train_losses[epoch]:.8f}\t'
                  f'Valid loss: {valid_losses[epoch]:.8f}\t'
                  f'Train erp loss: {train_erp_losses[epoch]:.8f}\t'
                  f'Valid erp loss: {valid_erp_losses[epoch]:.8f}\t'                  
                  f'Train accuracy: {100 * train_accs[epoch]:.2f}\t'
                  f'Valid accuracy: {100 * valid_accs[epoch]:.2f}') 
            print(f'device {device}\t\t\t TP: {TP} \t FP: {FP} \t TN: {TN} \t FN: {FN} --- F1: {2*TP/(2*TP+FP+FN):.5f}')              
        torch.cuda.empty_cache()

    plt.clf()
    plt.plot(train_accs,'b-', label="train accuracy")
    plt.plot(valid_accs,'r.', label="validation accuracy")
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    filepath = os.path.join(os.path.dirname(model_path), f"{jobname}_accuracy_curve.png") 
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    # plot loss    
    plt.clf()
    fig, ax1 = plt.subplots()
    l1 = ax1.plot(train_losses,'b-', label="train loss")
    l2 = ax1.plot(valid_losses,'b.', label="validation loss")    
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax2 = ax1.twinx()
    l3 = ax2.plot(np.log10(train_erp_losses),'r-', label="train erp loss")
    l4 = ax2.plot(np.log10(valid_erp_losses),'r.', label="validation erp loss")
    ax2.set_ylabel('Log(Loss)')
    fig.legend([l1, l2, l3, l4], labels=["train loss", "validation loss", "train erp loss", "validation erp loss"],loc="upper right")
    
    filepath = os.path.join(os.path.dirname(model_path), f"{jobname}_loss_curve.png") 
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
