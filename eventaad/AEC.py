import os
import numpy as np
import math
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import interpolate
from torch.autograd import Variable
import torchaudio.transforms as T
import matplotlib.pyplot as plt
import pickle
import joblib

from .BaseNet import *
from .EEGModels import *
from .convLSTM import ConvLSTM
from utils.utils import *

class LinearClassifier(nn.Module):
    def __init__(self, config:dict):
        super().__init__()
        self.linear = nn.Linear(in_features=config['in_features'], out_features=config['out_features'])
        self.actv = nn.Sigmoid()
        
    def forward(self, X, y=None):
        return self.actv(self.linear(X)).squeeze(-1)
        
class SpatialFilter(nn.Module):
    def __init__(self, weights):
        super().__init__()
        self.weights = torch.tensor(weights)
    def forward(self, X, y=None):
        return torch.matmul(self.weights.to(X.device), X)

class TimeFeatureLayer(nn.Module):
    def __init__(self, L, input_shape, in_channels, out_channels, conv_kernel, pool_kernel, pool_stride):
        super().__init__()
        self.cnn = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=conv_kernel)
        self.batchnorm = nn.BatchNorm1d(num_features=out_channels)
        self.actv = nn.Tanh()        
        self.pool = nn.AvgPool1d(kernel_size=pool_kernel, stride=pool_stride)
        out_shape = (L-conv_kernel+1)
        self.out_shape = int((out_shape-pool_kernel)/pool_stride + 1)
    def forward(self, X, y=None):
        X = self.cnn(X)
        X = self.batchnorm(X)        
        X = self.actv(X)
        X = self.pool(X)
        return X

class TimeSpatialFeatureLayer(nn.Module):        
    def __init__(self, L, input_shape, in_channels, out_channels, conv_kernel, pool_kernel, pool_stride):
        super().__init__()
        self.cnn = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=conv_kernel)
        self.batchnorm = nn.BatchNorm1d(num_features=out_channels)
        self.actv = nn.Tanh()
        self.pool = nn.AvgPool1d(kernel_size=pool_kernel, stride=pool_stride)
        out_shape = (L-conv_kernel+1)
        out_shape = int((out_shape-pool_kernel)/pool_stride + 1)
    def forward(self, X, y=None):
        X = self.cnn(X)
        X = self.batchnorm1(X)
        X = self.actv(X)
        X = self.pool(X)
        return None

class AECNet(BaseNet):
    def __init__(self):
        super().__init__()
        self.pretrained = None
        self.feature_extractor = None
        self.classifier = None
        self.pretrained_classifier = None
        self.pretrained_feature_extractor = None
        self.feature_freeze = True
        self.classifier_freeze = False        
        
    def __visualize__(self, X, y_true, y_hat, sr, scaler, filename, title):
        if scaler!=None:
            print('transforming scaler')
            X = scaler.inverse_transform(X)
            y_true = scaler.inverse_transform(y_true)
            y_hat = scaler.inverse_transform(y_hat)
            
        (nchns, L) = y_true.shape
        t = np.linspace(0, L/sr, L)
        colors = plt.cm.jet(np.linspace(0,1,nchns))
        plt.clf()
        for i in range(nchns):
            plt.plot(t, X[i], ':', color=colors[i], linewidth=0.5, label= f'input_chn{i}')
            plt.plot(t, y_true[i], '--', color=colors[i], linewidth=1, label= f'true_chn{i}')
            plt.plot(t, y_hat[i], '-', color=colors[i], linewidth=1, label= f'pred_chn{i}')
        # plt.legend()
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude ($\mu$V)')
        plt.title(title, loc='center')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()             

    def freeze_feature_extractor(self, freeze):
        for param in self.feature_extractor.parameters():
            param.requires_grad = (not freeze)

    def freeze_classifier(self, freeze):
        for param in self.classifier.parameters():
            param.requires_grad = (not freeze)       

    def initialize(self):
        for name, param in self.named_parameters():
            if param.requires_grad and 'weight' in name:
                if 'batchnorm' in name:
                    continue
                else:
                    #nn.init.xavier_normal_(param) # bell-shaped
                    nn.init.xavier_uniform_(param)
                    #nn.init.normal_(param)
        if self.pretrained is not None:
            print(f'Loading pretrained model: {self.pretrained}')
            states = torch.load(self.pretrained)
            self.load_state_dict(states)
        if self.pretrained_feature_extractor is not None:
            print(f'Loading pretrained feature extractor: {self.pretrained_feature_extractor}')
            states = torch.load(self.pretrained_feature_extractor)
            self.feature_extractor.load_state_dict(states)
        if self.feature_freeze:
            print('freezing feature')
            for param in self.feature_extractor.parameters():
                param.requires_grad = False
        if self.pretrained_classifier is not None:
            print(f'Loading pretrained classifier: {self.pretrained_classifier}')
            states = torch.load(self.pretrained_classifier)
            self.classifier.load_state_dict(states)
        if self.classifier_freeze:
            print('freezing classifier')
            for param in self.classifier.parameters():
                param.requires_grad = False
            
    def fit(self):
        pass

    def evaluate(self, data_loader, scaler, device, criterion, model_path, jobname=None, print_output=False):
        pass
        
    def plot_inout(self, X, y_true, y_hat, erp, scaler, filename, sr, feature=None):
        (nchns, ntimes) = X.shape
        colors = plt.cm.jet(np.linspace(0,1, nchns))
        if scaler!=None:
            X = scaler.inverse_transform(X)
            erp = scaler.inverse_transform(erp)
            if feature is not None:
                feature = scaler.inverse_transform(feature)
        T = np.arange(ntimes)/sr
        plt.clf()
        fig, ax1 = plt.subplots()
        for i in range(nchns):
            ax1.plot(T, X[i], '--', color=colors[i], linewidth=0.5)
        for i in range(len(erp)):
            l1, = ax1.plot(T, erp[i], '-', color=colors[i], label='ERP', linewidth=1.0)
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Amplitude ($\mu$V)')
        ax1.legend()
        ax1.set_ylim(-20,20)
        ax1.set_title(f"1s epoched data: y_hat={y_hat:.4f}, y_true={y_true}", loc='center')

        if feature is not None:
            ax2 = ax1.twinx()
            for i in range(len(feature)):
                l2, = ax2.plot(T, feature[i], '-', color='r', label='estimated', linewidth=1.0)
            ax2.legend()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()

class AECNet1(AECNet):
    def __init__(self, config:dict, sr, start, end, channels, channels_erp, erp_forcing, hybrid_training):
        super().__init__()
        self.device = None
        self.sr = sr
        self.start = start
        self.end = end
        self.L = self.end - self.start # samples
        self.channels = channels
        self.channels_erp = channels_erp
        self.erp_forcing = erp_forcing
        self.hybrid_training = hybrid_training
        #
        fex = config['fex']
        if fex['type']=='SpatialFilter':
            feature_path = None
            if fex['pretrained'] is not None:
                feature_path = os.path.expandvars(fex['pretrained'])     
            weights = None
            if  feature_path is not None and os.path.exists(feature_path):
                with open(feature_path, 'rb') as f:
                    variables = pickle.load(f)
                    (weights,) = tuple(variables)
            if weights is not None:
                X_train = np.matmul(weights, X_train)
                X_test = np.matmul(weights, X_test)
        elif fex['type']=='EEGNet':
            self.feature_extractor = eval(fex['type'])(fex, self.L, self.sr, self.channels, self.channels_erp)
            if fex['pretrained'] is not None:
                self.pretrained_feature_extractor = os.path.expandvars(fex['pretrained'])
            self.feature_freeze = fex['freeze']
        #
        classifier = config['classifier']        
        self.classifier = eval(classifier['type'])(classifier)
        self.classifier_freeze = classifier['freeze']
        if classifier['pretrained'] is not None:
            self.pretrained_classifier = os.path.abspath(classifier['pretrained'])
        

    def forward(self, X, erp=None, y=None):
        input_shape = X.shape
        device = X.device
        X = self.feature_extractor(X)
        if hasattr(self.feature_extractor, 'features'):
            X = self.feature_extractor.features
        self.features = X
        if (self.erp_forcing and erp is not None):
            out = self.classifier(erp)
        else:
            out = self.classifier(X)
        del X
        return (self.features, out.squeeze(-1)) if out.ndim>1 else (self.features, out)  

class AECNet2(AECNet):
    def __init__(self, config:dict, sr, start, end, channels, channels_erp, erp_forcing, hybrid_training):
        super().__init__()
        self.device = None
        self.sr = sr
        self.start = start
        self.end = end
        self.L = self.end - self.start # samples
        self.channels = channels
        self.channels_erp = channels_erp
        self.erp_forcing = erp_forcing
        self.hybrid_training = hybrid_training
        if config['pretrained'] is not None:
            self.pretrained = os.path.abspath(config['pretrained'])
        #
        self.downsampled_len = config['downsampled']
        fex = config['fex']
        self.hidden_size = fex['hidden_size']
        self.spatial_shape = tuple(fex['spatial_shape'])
        self.feature_extractor = ConvLSTM(fex['input_size'], fex['hidden_size'], tuple(fex['kernel_size']), fex['num_layers'], fex['batch_first'], fex['dropoutRate'], fex['bias'])
        self.spatial_CNN = nn.Conv2d(in_channels=fex['hidden_size'], out_channels=fex['hidden_size'], kernel_size=self.spatial_shape, groups=fex['hidden_size'], padding='valid')
        # self.spatial_FC = []
        # self.spatial_FC = nn.ModuleList(self.spatial_FC)
        if fex['pretrained'] is not None:
            self.pretrained_feature_extractor = os.path.expandvars(fex['pretrained'])
        self.feature_freeze = fex['freeze']
        #
        classifier = config['classifier']        
        self.classifier = eval(classifier['type'])(classifier)
        self.classifier_freeze = classifier['freeze']
        if classifier['pretrained'] is not None:
            self.pretrained_classifier = os.path.abspath(classifier['pretrained'])
        

    def forward(self, X, erp=None, y=None):
        device = X.device
        X = interpolate(X, size=self.downsampled_len)
        (N, chs, L) = X.shape
        X = X.transpose(-1, -2)
        X = self.__arrange_channels__(X, self.channels, system='1020', shape=tuple(self.spatial_shape))
        X = X.view(N, L, 1, self.spatial_shape[0], self.spatial_shape[1])
        X,_ = self.feature_extractor(X)
        if hasattr(self.feature_extractor, 'features'):
            X = self.feature_extractor.features
        X = X.view(-1, self.hidden_size, self.spatial_shape[0], self.spatial_shape[1])
        X = self.spatial_CNN(X)
        X = X.view(-1, L, self.hidden_size)
        self.features = interpolate(X[...,0:1].transpose(-1,-2), size=self.L)
        X = torch.flatten(X, 1, -1)
        if (self.erp_forcing and erp is not None):
            out = self.classifier(erp)
        else:
            out = self.classifier(X)
        del X
        return (self.features, out.squeeze(-1)) if out.ndim>1 else (self.features, out)  

    def __arrange_channels__(self, X, channels, system='1020', shape=(7,5)):
        """ Arrange electrodes into 2D plane for CNN. The electrodes POz, M1, M2 are discarded due to less relevance and easy arrangement. Adding 6 more electrodes by averaging interpolation:
        Fp3/AF7 = (Fp1+F7)/2 and Fp4/AF8 = (Fp2+F8)/2 for the 1st row of Fpx
        FCz = (Fz+Cz+FC1+FC2)/4 for the 3rd row of FCx
        CPz = (Cz+Pz+CP1+CP2)/4 for the 5th row of CPx
        O3/PO7 = (O1+P7)/2 and O4/PO8 = (O2+P8)/2 for the 7th row of Ox
        """
        if system=='1020':
            if shape==(7,5) and len(channels)==32:
                selected_chns = ['Fp1', 'Fpz', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FC5', 'FC1', 'FC2', 'FC6', 'T7-T3', 'C3', 'Cz', 'C4', 'T8-T4', 'CP5', 'CP1', 'CP2', 'CP6', 'P7-T5', 'P3', 'Pz', 'P4', 'P8-T6', 'O1', 'Oz', 'O2']
                idxs = [channels.index(ch) for ch in selected_chns]
                Y = X[...,idxs]                
                Fp3 = X[...,[channels.index(chn) for chn in ['Fp1', 'F7']]].mean(axis=-1, keepdims=True)
                Fp4 = X[...,[channels.index(chn) for chn in ['Fp2', 'F8']]].mean(axis=-1, keepdims=True)
                FCz = X[...,[channels.index(chn) for chn in ['Fz', 'FC1', 'FC2', 'Cz']]].mean(axis=-1, keepdims=True)
                CPz = X[...,[channels.index(chn) for chn in ['Cz', 'CP1', 'CP2', 'Pz']]].mean(axis=-1, keepdims=True)
                O3PO7 = X[...,[channels.index(chn) for chn in ['O1', 'P7-T5']]].mean(axis=-1, keepdims=True)
                O4PO8 = X[...,[channels.index(chn) for chn in ['O2', 'P8-T6']]].mean(axis=-1, keepdims=True)
                new_Y = [Fp3, Y[...,:3], Fp4, Y[...,3:10], FCz, Y[...,10:19], CPz, Y[...,19:26], O3PO7, Y[...,26:29], O4PO8]
                new_Y = torch.cat(new_Y, -1)
                del Y, 
                return new_Y
        return X