import os
import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from scipy.signal import resample
import mne
import joblib
import torch
import math
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import torchaudio.transforms as T
from .ERPDataset import ERPDataset

MIN_LATENCY = 0.2
MAX_LATENCY = 0.7
MIN_WIDTH = 0.3
MAX_WIDTH = 0.6
PRE_STIMULUS = 0.2

def makeSinERP(length, ERP_ltc, erp_w, sr, amps, snr=0):
    chns = len(amps.shape)
    nepochs = len(ERP_ltc)
    erp = np.zeros((nepochs, chns, 5*length))
    #
    for i in range(nepochs):   
        t = np.arange(erp_w[i])
        sin = np.sin(np.pi/erp_w[i]*t)
        erp[i,:,2*length:2*length+erp_w[i]] = erp[i,:,2*length:2*length+erp_w[i]] + sin
        erp[i] = np.roll(erp[i], shift=(ERP_ltc[i]-erp_w[i]/2).astype(int), axis=-1)
    scaler = np.sqrt(10**(snr/10))
    erp = scaler*np.expand_dims(amps, axis=-1)*erp[...,2*length:3*length]   
    return erp

def findERPWidth(erp, min_w, max_w):
    cross = []
    starting = False
    for i in range(len(erp)-1):
        if not starting and erp[i] > 0:
            continue
        starting = True
        if (erp[i]*erp[i+1] < 0):
            cross.append(i)
    w = 128
    for i in range(1,len(cross),2):
        dw = cross[i]-cross[i-1]
        if dw >= min_w and dw <= max_w:
            w = dw
            break
    return w
    
def makeRealERP(length, ERP_ltc, erp_w, sr, grand_erps, grand_ltc, grand_w, snr=0):
    chns = len(grand_erps)
    nepochs = len(ERP_ltc)
    erp = np.zeros((nepochs, chns, 5*length))
    # find grand ERP width
    w_scale = erp_w/grand_w
    new_len = (length*w_scale).astype(int)
    new_grand_ltc = (grand_ltc*w_scale).astype(int)
    for i in range(nepochs):
        new_erp = resample(grand_erps, new_len[i], axis=1)
        erp[i,:,2*length+ERP_ltc[i]-new_grand_ltc[i]:2*length+ERP_ltc[i]-new_grand_ltc[i]+new_len[i]] = new_erp
    scaler = np.sqrt(10**(snr/10))
    erp = scaler*erp[...,2*length:3*length] 
    return erp    

def sampling(X, y, n, min_seed, max_seed):
    new_X = []
    new_y = []
    for i in range(n):
        k = random.randint(min_seed, max_seed) # number of candidates used to generate new data point.
        idx = np.random.choice(len(X), k)
        new_X.append(X[idx,...].mean(axis=0, keepdims=True))
        new_y.append(y[idx,...].mean(axis=0, keepdims=True))
    return (np.concatenate(new_X, axis=0), np.concatenate(new_y, axis=0))

def makeERPdata(ds_path):
    print(f'loading raw dataset {ds_path}')
    loaded_data = dict(np.load(ds_path, allow_pickle=True))    
    X_raw = list(loaded_data['X'])
    y_raw = list(loaded_data['y'])
    nsbjs = len(X_raw)
    ERP = []
    for i in range(nsbjs):
        labels, weights = np.unique(y_raw[i], return_counts=True)
        X_avg = []
        for j in range(len(labels)):
            X_avg.append(X_raw[i][y_raw[i] == labels[j]].mean(axis=0, keepdims=False))
        ERP.append(np.array(X_avg))
        del X_avg
    #
    loaded_data['ERP'] = ERP
    loaded_data['X'] = list(loaded_data['X'])
    loaded_data['y'] = list(loaded_data['y'])
    return loaded_data

class ExperimentalERPDataset(ERPDataset):
    def __init__(self, config, loaded_data):
        channels = list(loaded_data['channels'])
        channels_in = config['channels']
        channels_erp = channels_in
        if 'channels_erp' in config.keys():
            channels_erp = config['channels_erp']
        chn_idx_in = [channels.index(ch) for ch in channels_in]       
        chn_idx_out = [channels.index(ch) for ch in channels_erp]
        sr = config['sr']
        self.start = config['start']
        self.start = int(self.start*sr/1000) # samples
        self.end = config['end']
        self.end = int(self.end*sr/1000) # samples
        self.L = self.end - self.start # samples  
        #
        self.X = loaded_data['X']
        self.y = loaded_data['y']
        self.n_sbjs = len(self.X)
        if config['upsampling']:
            self.__up_sampling__(config['factor'], config['min_seed'], config['max_seed'])
        raw_ERP = loaded_data['ERP'] if 'ERP' in loaded_data.keys() else None
        self.ERP = self.__make_ERP_data__(raw_ERP, chn_idx_out)
        self.epoch_nums = []
        for i in range(self.n_sbjs):
            self.epoch_nums.append(len(self.y[i]))
        X = 1e6*np.concatenate(self.X, axis=0)[...,self.start:self.end].astype(np.float32)
        ERP = 1e6*np.concatenate(self.ERP, axis=0)[...,self.start:self.end].astype(np.float32)
        y = np.concatenate(self.y, axis=0).astype(np.int16)
        scaler = None
        if config['scaler']['type'] is not None:
            path = os.path.expandvars(config['scaler']['path'])
            data_shape = X.shape
            X = X.reshape(-1, 1)
            if os.path.exists(path):
                scaler = joblib.load(path)
                X = scaler.transform(X)
            else:
                if config['scaler']['type'] == 'MinMaxScaler':
                    feature_range = tuple(config['scaler']['feature_range'])
                    scaler = eval(config['scaler']['type'])(feature_range=feature_range)
                elif config['scaler']['type'] == 'RobustScaler':
                    scaler = eval(config['scaler']['type'])(quantile_range=(5.0, 95.0))   
                X = scaler.fit_transform(X)
                joblib.dump(scaler, path)
            X = X.reshape(data_shape)
            ERP_shape = ERP.shape
            ERP = scaler.transform(ERP.reshape(-1, 1)).reshape(ERP_shape)        
        super().__init__(self.n_sbjs, X, y, ERP, channels_in, channels_erp, config['ERP_types'], sr, scaler)
        print(f'self.epoch_nums: {self.epoch_nums}')     
        
    def __get_ERP__(self, idx):
        return self.ERP[idx]
        
    def __get_sbj_idx__(self, idx):
        # epoch_idx = self.epoch_indexes[idx]
        sum_ipochs = 0
        for i in range(self.n_sbjs):
            sum_ipochs += self.epoch_nums[i]
            if sum_ipochs>idx:
                sbj_idx = i
                break
        return sbj_idx
        
    def __up_sampling__(self, factor, min_seed, max_seed):
        for i in range(self.n_sbjs):
            labels, weights = np.unique(self.y[i], return_counts=True)
            X = [None]*len(weights)
            y = [None]*len(weights)
            original_L = len(self.X[i])
            for j in range(len(labels)):
                X[j] = self.X[i][self.y[i] == labels[j]]
                y[j] = self.y[i][self.y[i] == labels[j]]
                # up-sampling for each class
                new_X, new_y = sampling(X[j], y[j], factor*np.amax(weights), min_seed, max_seed)
                self.X[i] = np.concatenate((self.X[i], new_X))
                self.y[i] = np.concatenate((self.y[i], new_y))
                # add original data points for each class equally
                idx = random.sample(range(weights[j]), np.amin(weights))
                self.X[i] = np.concatenate((self.X[i], X[j][idx]))
                self.y[i] = np.concatenate((self.y[i], y[j][idx]))
                del new_X, new_y
            # discard the original data to have noise balanced dataset
            self.X[i] = self.X[i][original_L:]
            self.y[i] = self.y[i][original_L:]
            del X, y
    
    def __make_ERP_data__(self, raw_ERP, chn_idx_out):
        ERP = []    
        for i in range(self.n_sbjs):
            (nepochs, n_chns, L) = self.X[i].shape
            labels, weights = np.unique(self.y[i], return_counts=True)
            erp = np.zeros((nepochs, len(chn_idx_out), L))
            for j in range(len(labels)):
                if (raw_ERP is not None):
                    X_avg = raw_ERP[i][j][chn_idx_out]
                else:
                    X_avg = self.X[i][self.y[i] == labels[j]].mean(axis=0, keepdims=False)[chn_idx_out]
                erp[self.y[i] == labels[j]] = X_avg
            ERP.append(erp)
        return ERP
        
class SimulatedERPDataset(ERPDataset):
    def __init__(self, config, loaded_data):
        channels = list(loaded_data['channels'])
        channels_in = config['channels']
        channels_erp = channels_in
        if 'channels_erp' in config.keys():
            channels_erp = config['channels_erp']
        chn_idx_in = [channels.index(ch) for ch in channels_in]       
        chn_idx_out = [channels.index(ch) for ch in channels_erp]
        sr = config['sr']
        self.start = config['start']
        self.start = int(self.start*sr/1000) # samples
        self.end = config['end']
        self.end = int(self.end*sr/1000) # samples
        self.L = self.end - self.start # samples
        #
        self.X = loaded_data['X']
        self.y = loaded_data['y']
        self.n_sbjs = len(self.X)
        if config['upsampling']:
            self.__up_sampling__(config['factor'], config['min_seed'], config['max_seed'])
        raw_ERP = loaded_data['ERP'] if 'ERP' in loaded_data.keys() else None
        self.ERP = self.__make_ERP_data__(config['latency_std'], config['min_w'], config['max_w'], sr, config['SNR'], raw_ERP, chn_idx_out, config['synthesized_type'])
        self.epoch_nums = []
        for i in range(self.n_sbjs):
            self.epoch_nums.append(len(self.y[i]))
        X = 1e6*np.concatenate(self.X, axis=0)[...,self.start:self.end].astype(np.float32)
        ERP = 1e6*np.concatenate(self.ERP, axis=0)[...,self.start:self.end].astype(np.float32)
        y = np.concatenate(self.y, axis=0).astype(np.int16)
        scaler = None
        if config['scaler']['type'] is not None:
            path = os.path.expandvars(config['scaler']['path'])
            data_shape = X.shape
            X = X.reshape(-1, 1)
            if os.path.exists(path):
                scaler = joblib.load(path)
                X = scaler.transform(X)
            else:
                if config['scaler']['type'] == 'MinMaxScaler':
                    feature_range = tuple(config['scaler']['feature_range'])
                    print(f'feature_range: {feature_range}')
                    scaler = eval(config['scaler']['type'])(feature_range=feature_range)
                elif config['scaler']['type'] == 'RobustScaler':
                    scaler = eval(config['scaler']['type'])(quantile_range=(5.0, 95.0))   
                X = scaler.fit_transform(X)
                joblib.dump(scaler, path)
            X = X.reshape(data_shape)
            ERP_shape = ERP.shape
            ERP = scaler.transform(ERP.reshape(-1, 1)).reshape(ERP_shape)        
        super().__init__(self.n_sbjs, X, y, ERP, channels_in, channels_erp, config['ERP_types'], sr, scaler)
        print(f'self.epoch_nums: {self.epoch_nums}')        
        
    def __get_ERP__(self, idx):
        return self.ERP[idx]
        
    def __get_sbj_idx__(self, idx):
        sum_ipochs = 0
        for i in range(self.n_sbjs):
            sum_ipochs += self.epoch_nums[i]
            if sum_ipochs>idx:
                sbj_idx = i
                break
        return sbj_idx  
        
    def __up_sampling__(self, factor, min_seed, max_seed):
        for i in range(self.n_sbjs):
            labels, weights = np.unique(self.y[i], return_counts=True)
            # up-sampling for non-effect (class 0) only
            X = self.X[i][self.y[i] == 0]
            y = self.y[i][self.y[i] == 0]
            new_X, new_y = sampling(X, y, factor*len(labels)*len(y), min_seed, max_seed)
            self.X[i] = np.concatenate((self.X[i][self.y[i] == 1], new_X))
            self.y[i] = np.concatenate((self.y[i][self.y[i] == 1], new_y))            
            del X, y, new_X, new_y  

    def __make_ERP_data__(self, latency_std, min_w, max_w, sr, SNR, raw_ERP, chn_idx_out, erp_type):
        min_w = int(min_w*sr)
        max_w = int(max_w*sr)
        min_ltc = int((MIN_LATENCY+PRE_STIMULUS)*sr)
        max_ltc = int((MAX_LATENCY+PRE_STIMULUS)*sr)
        nsbjs = len(self.X)
        latencies = [None]*nsbjs
        
        erps = []
        non_erps = []
        for i in range(nsbjs):
            if raw_ERP is None:
                non_erp = self.X[i][self.y[i] == 0].mean(axis=0, keepdims=False)
                # labels, weights = np.unique(self.y[i], return_counts=True)
                erp = self.X[i][self.y[i] == 1].mean(axis=0, keepdims=False)
            else:
                non_erp = raw_ERP[i][0]
                erp = raw_ERP[i][1]
            non_erps.append(non_erp)
            erps.append(erp)
            (latency, amp) = mne.preprocessing.peak_finder(erp[chn_idx_out[0]])
            for ltc in latency:
                if ltc >= min_ltc and ltc <= max_ltc:
                   latencies[i] = ltc
                   break

        erps =  np.array(erps)
        grand_erps = erps.mean(axis=0, keepdims=False) # (nchs, L)
        (grand_ltc, grand_amp) = mne.preprocessing.peak_finder(grand_erps[chn_idx_out[0]])
        for ltc in grand_ltc:
            if ltc >= min_ltc and ltc <= max_ltc:
               grand_ltc = ltc
               break
        grand_w = findERPWidth(grand_erps[chn_idx_out[0]], min_w, max_w)
        print(f'grand_ltc: {grand_ltc}')
        print(f'grand_w: {grand_w}')
        
        # Sampling the mean and std for latency distribution of each subject
        ltc_dist = grand_erps[chn_idx_out[0]][min_ltc:max_ltc]
        ltc_dist = ((ltc_dist/np.sum(ltc_dist))*5000).astype(int)
        ltc_array = []
        ERP = []
        for i in range(min_ltc, max_ltc):
            ltc_array += [i for j in range(ltc_dist[i-min_ltc])]
        for i in range(nsbjs):
            non_target = self.X[i][self.y[i] == 0].mean(axis=0, keepdims=False)
            (nepochs, n_chns, L) = self.X[i].shape
            if latencies[i] is None:
                print(f'No ERP latency detected for subject {i}')
                labels, weights = np.unique(self.y[i], return_counts=True)
                erp = np.zeros((nepochs, len(chn_idx_out), L))
                for j in range(len(labels)):
                    X_avg = self.X[i][self.y[i] == labels[j]].mean(axis=0, keepdims=False)[chn_idx_out]
                    erp[self.y[i] == labels[j]] = X_avg
                ERP.append(erp)                
            else:
                # Add ERPs with latency in normal range as the target ERP (label 1s)
                X_bg = self.X[i][self.y[i] == 0]
                idxs = np.arange(len(X_bg))
                np.random.shuffle(idxs)
                part1 = idxs[:int(len(idxs)/4)]
                part2 = idxs[int(len(idxs)/4): int(len(idxs)/2)]
                part3 = idxs[int(len(idxs)/2):]
                X_new = X_bg[part1]
                y_new = np.zeros(len(X_new))
                ERP_new = np.zeros((X_new.shape[0], len(chn_idx_out), X_new.shape[2]))
                # X_bg = X_bg[:60]
                ltc_high = latencies[i] + int(np.sqrt(3)*latency_std*sr)
                ltc_low = latencies[i] - int(np.sqrt(3)*latency_std*sr)
                ltcs = np.random.randint(ltc_low, ltc_high, len(part3))
                widths = np.random.randint(min_w, max_w, len(part3))
                if erp_type == 'sinusoid':
                    erp = makeSinERP(L, ltcs, widths, sr, grand_erps[:,grand_ltc], snr=SNR)
                else:
                    erp = makeRealERP(L, ltcs, widths, sr, grand_erps, grand_ltc, grand_w, snr=SNR)
                X_new = np.concatenate((X_new, X_bg[part3]+erp))
                y_new = np.concatenate((y_new, np.ones(len(erp))))
                ERP_new = np.concatenate((ERP_new, erp[:,chn_idx_out,:]))           
                # Add ERPs with latency out normal range as the non-target ERP (label 0)
                ltcs = np.random.randint(-max_w, min_ltc, int(len(part2)/2))
                ltcs = np.concatenate((ltcs, np.random.randint(max_ltc, L+max_w, len(part2)-int(len(part2)/2))))
                widths = np.random.randint(min_w, max_w, len(ltcs))
                if erp_type == 'sinusoid':
                    erp = makeSinERP(L, ltcs, widths, sr, grand_erps[:,grand_ltc])
                else:
                    erp = makeRealERP(L, ltcs, widths, sr, grand_erps, grand_ltc, grand_w)
                X_new = np.concatenate((X_new, X_bg[part2]+erp))
                y_new = np.concatenate((y_new, np.zeros(len(erp))))
                ERP_new = np.concatenate((ERP_new, erp[:,chn_idx_out,:]))
                self.X[i] = X_new
                self.y[i] = y_new
                ERP.append(ERP_new)
                del X_new, y_new, ERP_new
        ERP = np.array(ERP, dtype=object)       
        return ERP            
        
class MixedERPDataset(ConcatDataset):
    def __init__(self, datasets, scaler):
        self.scaler = scaler       
        super().__init__(datasets)