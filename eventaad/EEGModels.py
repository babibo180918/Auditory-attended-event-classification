import os
import numpy as np
import math
import torch
import torch.nn as nn

class DepthwiseConv2D(nn.Module):
    def __init__(self,
                 in_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 depth_multiplier=1,
                 dilation=1,
                 groups=1,
                 bias=True,
                 padding_mode='zeros',
                 device=None,
                 dtype=None) -> None:
        super().__init__()

        self.depthwise_conv = nn.Conv2d(in_channels=in_channels,
                                        out_channels=in_channels*depth_multiplier,
                                        kernel_size=kernel_size,
                                        stride=stride,
                                        padding=padding,
                                        dilation=dilation,
                                        groups=groups,
                                        bias=bias,
                                        padding_mode=padding_mode,
                                        device=device,
                                        dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise_conv(x)
        return x
        
class PointwiseConv2D(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 bias=True,
                 device=None,
                 dtype=None) -> None:
        super().__init__()

        self.pointwise_conv = nn.Conv2d(in_channels=in_channels,
                                        out_channels=out_channels,
                                        kernel_size=(1, 1),
                                        stride=1,
                                        padding=0,
                                        dilation=1,
                                        groups=1,
                                        bias=bias,
                                        padding_mode='zeros',
                                        device=device,
                                        dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pointwise_conv(x)
        return x
        
class SeparableConv2D(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 depth_multiplier=1,
                 bias=True,
                 padding_mode='zeros',
                 device=None,
                 dtype=None) -> None:
        super().__init__()

        self.depthwise_conv = DepthwiseConv2D(in_channels=in_channels,
                                              kernel_size=kernel_size,
                                              stride=stride,
                                              padding=padding,
                                              depth_multiplier=depth_multiplier,
                                              dilation=dilation,
                                              groups=in_channels,
                                              bias=bias,
                                              padding_mode=padding_mode,
                                              device=device,
                                              dtype=dtype)

        self.pointwise_conv = PointwiseConv2D(in_channels=in_channels,
                                              out_channels=out_channels,
                                              bias=bias,
                                              device=device,
                                              dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        return x        

def max_norm(model, max_val=3, eps=1e-8):
    for name, param in model.named_parameters():
        if 'bias' not in name:
            norm = param.norm(2, dim=0, keepdim=True)
            desired = torch.clamp(norm, 0, max_val)
            param = param * (desired / (eps + norm))

class EEGNet(nn.Module):
    """ Pytorch Implementation of EEGNet
    http://iopscience.iop.org/article/10.1088/1741-2552/aace8c/meta
    Inputs:

    nb_classes      : int, number of classes to classify
    Chans, Samples  : number of channels and time points in the EEG data
    dropoutRate     : dropout fraction
    kernLength      : length of temporal convolution in first layer. We found
                    that setting this to be half the sampling rate worked
                    well in practice. For the SMR dataset in particular
                    since the data was high-passed at 4Hz we used a kernel
                    length of 32.     
    F1, F2          : number of temporal filters (F1) and number of pointwise
                    filters (F2) to learn. Default: F1 = 4, F2 = F1 * D. 
    D               : number of spatial filters to learn within each temporal
                    convolution. Default: D = 2
    dropoutType     : Either SpatialDropout2D or Dropout, passed as a string.
    """
    def __init__(self,  config:dict, L, sr, channels, channels_erp, nb_classes=1):
        super().__init__()
        self.device = None
        self.sr = sr
        self.L = L
        self.channels = channels
        self.channels_erp = channels_erp
        self.nb_classes = nb_classes
        #
        self.filter_path = None
        self.filter_ = None
        self.features = None
        self.in_channels = len(self.channels)
        self.spatial_filter = config['spatial_filter']['type']
        if config['spatial_filter']['filter_path'] is not None:
            self.filter_path = os.path.expandvars(config['spatial_filter']['filter_path'])
        self.n_components = config['spatial_filter']['n_components']        
        #
        if self.spatial_filter is not None and self.filter_path is not None:
            if os.path.exists(self.filter_path):
                with open(self.filter_path, 'rb') as f:
                    variables = pickle.load(f)
                    (self.weights,) = tuple(variables)
                    self.filter_ = SpatialFilter(np.float32(self.weights))
                    self.in_channels = self.n_components
        #        
        self.F1 = config['F1']
        self.D = config['D']
        self.F2 = config['F2']
        self.dropoutRate = config['dropoutRate']
        self.norm_rate = config['norm_rate']
        self.kernLength = config['kernLength']
        
        #
        self.block1_conv2d = nn.Conv2d(in_channels=1, out_channels=self.F1, kernel_size=(1, self.kernLength), padding='same', bias=False)
        self.block1_batchnorm1 = nn.BatchNorm2d(num_features=self.F1)
        self.block1_dwConv2d = DepthwiseConv2D(in_channels=self.F1, kernel_size=(self.in_channels,1), depth_multiplier=self.D, groups=self.F1, bias=False)
        self.block1_batchnorm2 = nn.BatchNorm2d(num_features=self.D*self.F1)
        self.block1_actv = nn.ELU()
        self.block1_avgpool = nn.AvgPool2d(kernel_size=(1,4), stride=(1,4))
        self.block1_dropout = nn.Dropout(p=self.dropoutRate)
        #
        self.block2_separableConv2d = SeparableConv2D(in_channels=self.D*self.F1, out_channels=self.F2, kernel_size=(1, 16), bias=False, padding='same')
        self.block2_batchnorm = nn.BatchNorm2d(num_features=self.F2)
        self.block2_actv = nn.ELU()
        self.block2_avgpool = nn.AvgPool2d(kernel_size=(1,8), stride=(1,8))
        self.block2_dropout = nn.Dropout(p=self.dropoutRate)
        #
        self.out_FC = nn.Linear(in_features=self.F2*(self.L//32), out_features=self.nb_classes)
        self.out_actv = nn.Sigmoid()
        
    def forward(self, X, y=None):
        if self.filter_ is not None:
            X = self.filter_(X)    
        batch_size = X.shape[0]
        #X = F.normalize(X, dim=-1)
        X = X.unsqueeze(1)
        X = self.block1_conv2d(X)
        X = self.block1_batchnorm1(X)
        X = self.block1_dwConv2d(X)
        X = self.block1_batchnorm2(X)
        X = self.block1_actv(X)
        X = self.block1_avgpool(X)
        block1_out = self.block1_dropout(X)
        #
        X = self.block2_separableConv2d(block1_out)
        X = self.block2_batchnorm(X)
        X = self.block2_actv(X)
        X = self.block2_avgpool(X)
        X = self.block2_dropout(X)
        X = torch.flatten(X, 1, -1)
        self.features = X.unsqueeze(1)
        X = self.out_FC(X)
        #max_norm(self.out_FC, self.norm_rate)
        out = self.out_actv(X)
        del X
        return out.squeeze(-1)
        
    def initialize(self):
        for name, param in self.named_parameters():
            if param.requires_grad and 'weight' in name:
                if 'batchnorm' in name:
                    continue
                else:
                    nn.init.xavier_uniform_(param)
    def plot_inout(self, X, y_true, y_hat, scaler, filename, sr, feature=None):
        (nchns, ntimes) = X.shape
        colors = plt.cm.jet(np.linspace(0,1, nchns))
        plt.clf()
        if scaler!=None:
            X = scaler.inverse_transform(X)
            if feature is not None:
                feature = scaler.inverse_transform(feature)
        T = np.arange(ntimes)/sr
        for i in range(nchns):
            plt.plot(T, i*5 + X[i], '--', color=colors[i], label="input", linewidth=0.5)
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude ($\mu$V)')
        plt.title(f"1s epoched data: y_hat={y_hat:.4f}, y_true={y_true}", loc='center')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()                      
        

class ERPEncLayer(nn.Module):
    """
	Encoder layer of ERPENet
	""" 
    def __init__(self, in_channels, conv_size, conv_sublayers, conv_kernel_size, dropoutRate):
        super().__init__()
        self.numOfSublayers = conv_sublayers
        self.convs = []
        self.batchnorms = []
        self.dropouts = []
        for i in range(self.numOfSublayers):
            if i==0:
                self.convs.append(nn.Conv2d(in_channels=in_channels, out_channels=2*conv_size, kernel_size=conv_kernel_size, padding='same'))
                self.batchnorms.append(nn.BatchNorm2d(num_features=2*conv_size))
            elif i==1:
                self.convs.append(nn.Conv2d(in_channels=2*conv_size, out_channels=conv_size, kernel_size=conv_kernel_size, padding='same'))
                self.batchnorms.append(nn.BatchNorm2d(num_features=conv_size))
            else:
                self.convs.append(nn.Conv2d(in_channels=conv_size, out_channels=conv_size, kernel_size=conv_kernel_size, padding='same'))
                self.batchnorms.append(nn.BatchNorm2d(num_features=conv_size))
            self.dropouts.append(nn.Dropout(p=dropoutRate))

        self.convs = nn.ModuleList(self.convs)
        self.batchnorms = nn.ModuleList(self.batchnorms)
        self.dropouts = nn.ModuleList(self.dropouts)
        self.out_channels = conv_size

    def forward(self, X, y=None):
        for i in range(self.numOfSublayers):
            X = self.convs[i](X)
            X = self.batchnorms[i](X)
            X = nn.LeakyReLU(negative_slope=0.1)(X)
            X = self.dropouts[i](X)
        return X            
    
class ERPEncoder(nn.Module):
    """
    Encoder portion of ERPENet
    """
    def __init__(self, cnn_input_shape, depth, init_conv_size, conv_sublayers, conv_kernel_size, dense_size, dropoutRate):
        super().__init__()
        self.cnn_input_shape = cnn_input_shape
        self.depth = depth
        self.dense_size = dense_size
        self.L = cnn_input_shape[0]
        cnn_chns = init_conv_size
        layer_in_channels = cnn_input_shape[-3] # number of input channels of CNN
        self.enc_cnn_layers = []
        for i in range(depth):
            self.enc_cnn_layers.append(ERPEncLayer(in_channels=layer_in_channels, conv_size=cnn_chns, conv_sublayers=conv_sublayers, conv_kernel_size=conv_kernel_size, dropoutRate=dropoutRate))
            cnn_chns = 2*cnn_chns
            layer_in_channels = self.enc_cnn_layers[i].out_channels
        self.enc_cnn_layers = nn.ModuleList(self.enc_cnn_layers)
        test_tensor = torch.rand(cnn_input_shape)
        for i in range(depth):
            test_tensor = self.enc_cnn_layers[i](test_tensor)   
        self.conv_out_shape = test_tensor.shape[-3:]
        test_tensor = torch.flatten(test_tensor, -3, -1)
        self.rnn_layer = nn.LSTM(input_size=test_tensor.shape[-1], hidden_size=dense_size, batch_first=True, num_layers=1, dropout=dropoutRate)
    def forward(self, X, y=None):
        #print(f'encoder_in: {X.shape}')
        batch_size = X.shape[0]
        X = X.transpose(-2, -1).reshape(-1, self.cnn_input_shape[-3], self.cnn_input_shape[-2], self.cnn_input_shape[-1])
        for i in range(self.depth):
            X = self.enc_cnn_layers[i](X)
        #X = torch.flatten(X, -3, -1)
        X = X.reshape(batch_size, self.L, -1)
        (h0, c0) = self.init_rnn_hidden(batch_size, self.dense_size, 1, 1)
        h0 = h0.to(X.device)
        c0 = c0.to(X.device)        
        X, (hn,cn) = self.rnn_layer(X,(h0,c0))
        #print(f'encoder_out: {X[:,-1,:].shape}')
        return X[:,-1,:]
        
    def init_rnn_hidden(self, batch_size, hidden_size, layer_nums, D):
        return (torch.zeros(layer_nums*D, batch_size, hidden_size),
                torch.zeros(layer_nums*D, batch_size, hidden_size))  
        
class ERPDecLayer(nn.Module):
    """
    Decoder layer of ERPENet
    """
    def __init__(self, in_channels, conv_size, conv_sublayers, conv_kernel_size, dense_size, dropoutRate):
        super().__init__()
        self.numOfSublayers = conv_sublayers
        self.convs = []
        self.batchnorms = []
        self.dropouts = []
        for i in range(self.numOfSublayers):
            if i==0:
                self.convs.append(nn.Conv2d(in_channels=in_channels, out_channels=conv_size, kernel_size=(3,3), padding='same'))
                self.upsampling = nn.Upsample(scale_factor=2)
                self.zeropadding = nn.ZeroPad2d((1,1))
                self.batchnorms.append(nn.BatchNorm2d(num_features=conv_size))
            elif i==1:
                self.convs.append(nn.Conv2d(in_channels=conv_size, out_channels=int(conv_size/2), kernel_size=(3,3), padding='same'))
                self.batchnorms.append(nn.BatchNorm2d(num_features=int(conv_size/2)))
            else:
                self.convs.append(nn.Conv2d(in_channels=int(conv_size/2), out_channels=int(conv_size/2), kernel_size=(3,3), padding='same'))
                self.batchnorms.append(nn.BatchNorm2d(num_features=int(conv_size/2)))
            self.dropouts.append(nn.Dropout(p=dropoutRate))
        
        self.convs = nn.ModuleList(self.convs)
        self.batchnorms = nn.ModuleList(self.batchnorms)
        self.dropouts = nn.ModuleList(self.dropouts)
        self.out_channels = int(conv_size/2)

    def forward(self, X, y=None):
        for i in range(self.numOfSublayers):
            '''
            if i==0:
                X = self.upsampling(X)
                X = self.zeropadding(X)
            '''
            X = self.convs[i](X)
            X = self.batchnorms[i](X)
            X = nn.LeakyReLU(negative_slope=0.1)(X)
            X = self.dropouts[i](X)
        return X
                    
        
class ERPDecoder(nn.Module):
    """
    Decoder portion of ERPENet
    """    
    def __init__(self, L, conv_in_shape, depth, conv_sublayers, conv_kernel_size, dense_size, dropoutRate):
        super().__init__()
        self.L = L
        self.dec_cnn_in_shape = conv_in_shape
        self.depth = depth
        self.rnn_hidden_size = self.dec_cnn_in_shape[-3]*self.dec_cnn_in_shape[-2]*self.dec_cnn_in_shape[-1]
        self.rnn_layer = nn.LSTM(input_size=dense_size, hidden_size=self.rnn_hidden_size, batch_first=True, num_layers=1, dropout=dropoutRate)
        self.dec_cnn_layers = []
        cnn_chns = self.dec_cnn_in_shape[-3]*2
        layer_in_channels = self.dec_cnn_in_shape[-3]
        for i in range(depth):
            self.dec_cnn_layers.append(ERPDecLayer(in_channels=layer_in_channels, conv_size=cnn_chns, conv_sublayers=conv_sublayers, conv_kernel_size=conv_kernel_size, dense_size=dense_size, dropoutRate=dropoutRate))
            cnn_chns = int(cnn_chns/2)
            layer_in_channels = self.dec_cnn_layers[i].out_channels
        self.dec_cnn_layers = nn.ModuleList(self.dec_cnn_layers)
        cnn_out_chns = self.dec_cnn_layers[depth-1].out_channels
        self.reconstruct_layer = nn.Conv2d(in_channels=cnn_out_chns, out_channels=1, kernel_size=(1,1), padding='same')
        
    def forward(self, X, y=None):
        #print(f'decoder_in: {X.shape}')
        batch_size = X.shape[0]
        X = X.unsqueeze(1).repeat(1,self.L, 1)
        (h0, c0) = self.init_rnn_hidden(batch_size, self.rnn_hidden_size, 1, 1)
        h0 = h0.to(X.device)
        c0 = c0.to(X.device)
        X, (hn,cn) = self.rnn_layer(X,(h0,c0))           
        X = X.reshape(-1, self.dec_cnn_in_shape[-3], self.dec_cnn_in_shape[-2], self.dec_cnn_in_shape[-1])
        for i in range(self.depth):
            X = self.dec_cnn_layers[i](X)
        X = self.reconstruct_layer(X).squeeze()
        X = X.reshape(batch_size, self.L, -1).transpose(-2,-1)
        #print(f'decoder_out: {X.shape}')
        return X
    def init_rnn_hidden(self, batch_size, hidden_size, layer_nums, D):
        return (torch.zeros(layer_nums*D, batch_size, hidden_size),
                torch.zeros(layer_nums*D, batch_size, hidden_size))         
            

class ERPENet(nn.Module):
    """
    Pytorch implementation of ERPENet from
    https://ieeexplore.ieee.org/abstract/document/8723080
    Args:
        depth (int): number of CNN blocks, each has 3 CNN layers with BN and a dropout
        conv_size (int): initial CNN filter size, doubled in each depth level
        dense_size (int): size of latent vector and a number of filters of ConvLSTM2D
        input_dim (tuple): input dimention, should be in (y_spatial,x_spatial,temporal)
        dropoutRate (float): dropout rate used in all nodes
    """    
    def __init__(self, config:dict):
        super().__init__()
        #
        self.sr = config['dataset']['sr']
        self.start = config['dataset']['start']
        self.start = int(self.start*self.sr/1000) # samples
        self.end = config['dataset']['end']
        self.end = int(self.end*self.sr/1000) # samples
        self.L = self.end - self.start # samples
        self.channels = config['dataset']['channels']   
        self.nb_classes = config['dataset']['nb_classes']
        #
        self.channel_shape = tuple([self.L] + config['model']['channel_shape'])
        self.depth = config['model']['depth']
        self.conv_sublayers = config['model']['conv_sublayers']
        self.conv_kernel_size = tuple(config['model']['conv_kernel_size'])
        self.init_conv_size = config['model']['init_conv_size']
        self.dense_size = config['model']['dense_size']
        self.dropoutRate = config['model']['dropoutRate']
      
        #
        self.encoder = ERPEncoder(cnn_input_shape=self.channel_shape, depth=self.depth, conv_sublayers=self.conv_sublayers, init_conv_size=self.init_conv_size, conv_kernel_size=self.conv_kernel_size, dense_size=self.dense_size, dropoutRate=self.dropoutRate)
        self.latent_vector = None
        self.decoder = ERPDecoder(L=self.L, conv_in_shape=self.encoder.conv_out_shape, depth=self.depth, conv_sublayers=self.conv_sublayers, conv_kernel_size=self.conv_kernel_size, dense_size=self.dense_size, dropoutRate=self.dropoutRate)
        self.task_FC = nn.Linear(in_features=self.dense_size, out_features=self.nb_classes)
        if (self.nb_classes==1):
            self.task_actv = nn.Sigmoid()
        else:
            self.task_actv = nn.Softmax()
        
    def forward(self, X, y=None):
        self.input_shape = X.shape        
        X = self.encoder(X)
        self.latent_vector = X
        self.features = X
        decoded = self.decoder(X)
        X = self.task_FC(X)
        X = self.task_actv(X)
        return (decoded, X.squeeze(dim=1))
        
    def initialize(self):
        for name, param in self.named_parameters():
            if param.requires_grad and 'weight' in name:
                if 'batchnorm' in name:
                    continue
                else:
                    nn.init.xavier_uniform_(param)
                    
    def plot_inout(self, X, y_true, y_hat, scaler, filename, sr, feature=None):
        (nchns, ntimes) = X.shape
        colors = plt.cm.jet(np.linspace(0,1, nchns))
        plt.clf()
        if scaler!=None:
            X = scaler.inverse_transform(X)
            if feature is not None:
                feature = scaler.inverse_transform(feature)
        T = np.arange(ntimes)/sr
        for i in range(nchns):
            plt.plot(T, i*5 + X[i], '--', color=colors[i], label="input", linewidth=0.5)
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude ($\mu$V)')
        plt.title(f"1s epoched data: y_hat={y_hat:.4f}, y_true={y_true}", loc='center')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()                     