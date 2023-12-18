from abc import ABC, abstractmethod
import numpy as np
import torch
from torch.utils.data import Dataset

class ERPDataset(ABC, Dataset):
    """
    Abstract class for general ERP classification.
    """    
    @abstractmethod
    def __init__(self, n_sbjs, X, y, ERP, channels_in, channels_erp, n_types, sr, scaler=None):
        Dataset.__init__(self)
        self.n_sbjs = n_sbjs
        self.X = X
        self.y = y
        self.ERP = ERP
        self.channels_in = channels_in
        self.channels_erp = channels_erp
        self.n_types = n_types
        self.sr = sr
        self.data_shape = self.X.shape
        self.max_value = np.amax(self.X)
        self.min_value = np.amin(self.X)
        self.scaler = scaler
        pass
        
    def __len__(self):
        return len(self.y)        
        
    def __getitem__(self, idx):
        erp = self.__get_ERP__(idx)
        sbj_idx = self.__get_sbj_idx__(idx)
        return (torch.tensor(self.X[idx]),torch.tensor(erp), torch.tensor(self.y[idx], dtype=torch.float), sbj_idx)
        
    def __visualize__(self, X, ERP, y, idx):
        """
        Visualizing data point.
        """
        sbj_idx = self.__get_sbj_idx__(idx)
        L = X.shape[-1]
        t = np.linspace(self.start, self.end, L)
        colors = plt.cm.jet(np.linspace(0,1,len(self.channels_in)))
        plt.clf()
        for i in range(len(self.channels_in)):
            plt.plot(t, X[i], ':', color=colors[i], linewidth=0.5)
        for i in range(len(self.channels_erp)):
            plt.plot(t, ERP[i], '-', color=colors[i], linewidth=1, label= f'ERP_{self.channels_erp[i]}')
        plt.legend()
        plt.title(f'input of sbj {sbj_idx} label_{y}')
        plt.xlabel('samples')
        plt.ylabel('Amplitude')
        #plt.gcf().savefig(f"estimator/input_sbj_{sbj_idx}_label_{y}", dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()        
    
    @abstractmethod
    def __get_ERP__(self, idx):
        pass
        
    @abstractmethod
    def __get_sbj_idx__(self, idx):
        pass        
        
    @abstractmethod
    def __up_sampling__(self):
        pass
        
    @abstractmethod
    def __make_ERP_data__(self):
        pass                