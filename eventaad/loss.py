import os
import numpy as np
import torch
import torch.nn as nn
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T


class PLCompLoss(nn.Module):
    def __init__(self, c:float, n_fft:int, hop_length:int, time_dim:int=-1, reduction='mean', spectro_aware:bool=False, phase_aware:bool=False):
        super().__init__()
        self.__c = c
        self.__spectro_aware = spectro_aware
        self.__phase_aware = phase_aware
        self.__n_fft = n_fft
        self.__hop_length = hop_length
        self.__time_dim = time_dim
        self.__reduction = reduction
        self.spectrogramer = T.Spectrogram(n_fft=self.__n_fft, hop_length=self.__hop_length, power=None, pad_mode='replicate', normalized=False)
    def forward(self, y_hat:torch.Tensor, y_true:torch.Tensor) -> torch.Tensor:
        assert y_hat.shape == y_true.shape, "Two input tensors are not same size."
        device = torch.device('cpu')
        y_hat = y_hat.to(device)
        y_true = y_true.to(device)
        if self.__spectro_aware:
            y_hat = y_hat.transpose(-1, self.__time_dim)
            y_true = y_true.transpose(-1,self.__time_dim)
            spec_hat = self.spectrogramer(y_hat)
            spec_true = self.spectrogramer(y_true)
            A_hat_comp = torch.pow(spec_hat.abs(), self.__c)
            A_true_comp = torch.pow(spec_true.abs(), self.__c)
            if self.__phase_aware:
                phase_hat = spec_hat.angle()
                phase_true = spec_true.angle()
                sComp_hat = torch.complex(A_hat_comp*torch.cos(phase_hat), A_hat_comp*torch.sin(phase_hat))
                sComp_true = torch.complex(A_true_comp*torch.cos(phase_true), A_true_comp*torch.sin(phase_true))
                powComp = torch.sum(torch.pow((sComp_hat-sComp_true).abs(), 2))/sComp_hat.numel()
            else:
                powComp = nn.MSELoss(reduction=self.__reduction)(A_hat_comp, A_true_comp)
        else:
            y_hat_comp = torch.pow(y_hat.abs(), self.__c) 
            y_true_comp = torch.pow(y_true.abs(), self.__c) 
            powComp = nn.MSELoss(reduction=self.__reduction)(y_hat_comp, y_true_comp)
        return powComp


class DualTask_AE_Loss(nn.Module):
    def __init__(self, alpha1:float=0.3, alpha2:int=0.7, loss1=nn.MSELoss(), loss2=nn.BCELoss()):
        super().__init__()
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.loss1 = loss1
        self.loss2 = loss2
    def forward(self, X_pred:torch.Tensor, y_hat:torch.Tensor, X:torch.Tensor, y_true:torch.Tensor) -> torch.Tensor:
        return self.alpha1*self.loss1(X_pred, X) + self.alpha2*self.loss2(y_hat, y_true)
        
class WeightedFocalBCELoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=0.0):
        super().__init__()
        self.alpha = torch.tensor([alpha, 1-alpha])
        self.gamma = gamma
    def forward(self, inputs:torch.Tensor, targets:torch.Tensor) -> torch.Tensor:
        BCE_loss = nn.BCEWithLogitsLoss(reduction='none')(inputs, targets)
        targets = targets.type(torch.long)
        self.alpha = self.alpha.to(inputs.device)
        at = self.alpha.gather(0, targets.data.view(-1)).to(inputs.device)
        pt = torch.exp(-BCE_loss).to(inputs.device)
        F_loss = at*(1-pt)**self.gamma * BCE_loss
        return F_loss.mean()      