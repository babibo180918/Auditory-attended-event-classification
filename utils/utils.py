import os
from os.path import isfile
import glob
import matplotlib.pyplot as plt
import torchaudio.transforms as T
import torch
from torchsummary import summary
import librosa
import random

from sklearn.manifold import TSNE
from string import ascii_uppercase
from pandas import DataFrame
import numpy as np
import seaborn as sn
from sklearn.metrics import confusion_matrix

plt.rcParams["font.family"] = "Times New Roman"
colors = ['tab:red', 'tab:blue', 'tab:green', 'tab:gray', 'tab:orange', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:cyan']
markers = ['o', '^', '*', 's', 'x', 'p']
titlefontsizebig = 22
labelfontsizebig = 22
tickfontsizebig = 20
legendfontsizebig = 16
textfontsizebig = 20
#
titlefontsize = 16
labelfontsize = 16
tickfontsize = 14
legendfontsize = 12
textfontsize = 12

def one_hot_encode(numOfClasses, labels):
    encoded_labels = np.zeros((numOfClasses, len(labels)), dtype=int)
    for i in range(0, len(labels)):
        if labels[i] > 0:
            encoded_labels[labels[i]-1, i] = 1
            
    return encoded_labels

def getFileList(in_path):
    filepaths = []
    if os.path.isfile(in_path):
        filepaths.append(in_path)
    elif os.path.isdir(in_path):
        for filename in glob.glob(in_path + '/**/*.*', recursive=True):
            filepaths.append(filename)
    else:
        print("Path is invalid: " + in_path)
        return None

    return filepaths

def addToHDF5(filepath, *args):
    numOfArgs = len(args)
    if numOfArgs%2 != 0:
        print("Number of arguments are incorrect.")
        return
    
def plot_spectrogram(x, n_fft:int, hop_length:int, Fs:float, save_file=None):
    spectrogram = T.Spectrogram(
        n_fft=n_fft,
        hop_length=hop_length,
        center=True,
        pad_mode='replicate',
        power=2.0      
    )
    specs = spectrogram(torch.tensor(x))
    
    fig, axs = plt.subplots(1, 1)
    axs.set_title("Spectrogram (db)")
    axs.set_ylabel("frequency (bin)")
    axs.set_xlabel("frame")    
    im = axs.imshow(librosa.power_to_db(specs), origin="lower", aspect='auto')
    fig.colorbar(im, ax=axs)
    if save_file != None:
        plt.savefig(save_file, dpi=300, bbox_inches='tight')
    plt.close()

def tsne_visualization(embedding, labels, filename):
    tsne = TSNE(2, verbose=1)
    tsne_proj = tsne.fit_transform(embedding)    
    label_unique = torch.unique(labels, sorted=True)
    colors = ['#{:02x}{:02x}{:02x}'.format(random.randint(0,0xff), random.randint(0,0xff), random.randint(0,0xff)) for i in label_unique]
    plt.clf()
    fig, ax = plt.subplots(figsize=(8,8))
    for i in range(len(label_unique)):
        indices = (labels==label_unique[i])
        ax.scatter(tsne_proj[indices,0],tsne_proj[indices,1], c=colors[i], label=label_unique[i].item() ,alpha=0.5)
    ax.legend(fontsize='large', markerscale=2)
    plt.savefig(filename, dpi=500, bbox_inches='tight')
    plt.close()
    
def plot_confusion_matrix(y_true, y_predict, labels, filename, normalize='all'):
    columns = labels
    confm = confusion_matrix(y_true, y_predict, normalize=normalize)
    if confm.shape == (1,1):
        print(confm)
        return
    df_cm = DataFrame(confm, index=columns, columns=columns)
    plt.clf()
    ax = sn.heatmap(df_cm, cmap='Oranges', annot=True)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    plt.savefig(filename, dpi=500, bbox_inches='tight')
    plt.close()

def print_memory_info(device):
    t = torch.cuda.get_device_properties(device).total_memory
    r = torch.cuda.memory_reserved(device)
    a = torch.cuda.memory_allocated(device)
    f = r-a  # free inside reserved
    print(f'total memory: {t/1024/1024/1024}')
    print(f'reserved memory: {r/1024/1024/1024}')
    print(f'allocated memory: {a/1024/1024/1024}')
    print(f'free memory: {f/1024/1024/1024}')

def binary_accuracy(y_hat, y_true, thresh=0.5):
    y_pred = y_hat>thresh
    accuracy = (y_pred == y_true).float().mean()   
    return accuracy
    
def metrics(y_hat,y_true, thresh=None, weighted=False):
    if (thresh is None):
        max_acc = 0
        for i in range(100):
            thr = 0.5 + ((-1)**i)*0.01*int((i+1)/2)
            y_pred = y_hat>thr
            true_pos = torch.count_nonzero((y_true==1)&(y_pred==1))
            false_pos = torch.count_nonzero((y_true==0)&(y_pred==1))
            true_neg = torch.count_nonzero((y_true==0)&(y_pred==0))
            false_neg = torch.count_nonzero((y_true==1)&(y_pred==0))
            acc = (true_pos+true_neg)/len(y_hat)
            if acc>max_acc:
                max_acc = acc
                thresh = thr
    y_pred = y_hat>thresh
    true_pos = torch.count_nonzero((y_true==1)&(y_pred==1))
    false_pos = torch.count_nonzero((y_true==0)&(y_pred==1))
    true_neg = torch.count_nonzero((y_true==0)&(y_pred==0))
    false_neg = torch.count_nonzero((y_true==1)&(y_pred==0))
    acc = torch.count_nonzero(y_true==y_pred)/torch.numel(y_pred)
    if weighted:    
        classes, weights = torch.unique(y_true, return_counts=True)
        n_classes = len(classes)
        weights = weights.float()
        for i in range(n_classes):
            weights[i] = torch.count_nonzero((y_true==y_pred)&(y_pred==classes[i])).float()/weights[i]
        acc = weights.mean()
    return (true_pos, false_pos, true_neg, false_neg, acc, thresh)
    
def multiclass_accuracy(scores, yb):
    score2prob = nn.Softmax(dim=1)
    preds = torch.argmax(score2prob(scores), dim=1)
    return (preds == yb).float().mean()   

def getHardLabels(X, y, n_classes):
    idxs = (y==0)
    for i in range(1,n_classes):
        idxs = idxs|(y==i)
    return X[idxs], y[idxs]
           
def plot_compare_bar(compare_data, bar_labels, xtick_labels, y_label=None, title=None, save_path=None):
    nBars = len(compare_data)
    width = 0.2
    plt.clf()
    for i in range(nBars):
        (nXticks,) = compare_data[i].shape
        loc = np.arange(nXticks) - (nBars-1)*width/2 + i*width
        rects = plt.bar(loc, compare_data[i], width, label=bar_labels[i], color=colors[i], zorder=3)
    #
    ticks = np.arange(len(xtick_labels))  # the label locations
    plt.xticks(ticks=ticks, labels=xtick_labels, fontsize=tickfontsize)
    # plt.yticks(ticks=[0.5, 0.6, 0.7, 0.8, 0.9, 1], fontsize=tickfontsize)
    plt.yticks(ticks=[0.5, 0.6, 0.7, 0.8, 0.9, 1], labels=['${}^*$0.5', 0.6, 0.7, 0.8, 0.9, 1], fontsize=tickfontsize)
    plt.ylim(0.5, 1)    
    if y_label is not None:
        plt.ylabel(y_label, fontsize=labelfontsize)
    if title is not None:
        plt.title(title, fontsize=titlefontsize)
    plt.legend(fontsize=legendfontsize)
    plt.grid(axis = 'y', linestyle='--', linewidth=1.0, zorder=0)
    fig = plt.gcf()
    fig.set_size_inches(8, 2.5)
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
def plot_compare_bar_withSTDbar(compare_data, bar_labels, xtick_labels, y_label=None, title=None, save_path=None):
    nBars = len(compare_data)
    width = 0.2
    plt.figure()
    plt.clf()
    ticks = np.arange(len(xtick_labels))  # the label locations
    plt.xticks(ticks=ticks, labels=xtick_labels, fontsize=tickfontsize)
    # plt.yticks(ticks=[0.5, 0.6, 0.7, 0.8, 0.9, 1], fontsize=tickfontsize)
    plt.yticks(ticks=[0.5, 0.6, 0.7, 0.8, 0.9, 1], labels=['${}^*$0.5', 0.6, 0.7, 0.8, 0.9, 1], fontsize=tickfontsize)
    plt.ylim(0.5, 1)    
    plt.grid(axis = 'y', linestyle='--', linewidth=0.5, zorder=0)
    for i in range(nBars):
        (nXticks, obvs) = compare_data[i].shape
        mean = np.mean(compare_data[i], 1, keepdims=False)
        std = np.std(compare_data[i], 1, keepdims=False)
        loc = np.arange(nXticks) - (nBars-1)*width/2 + i*width
        rects = plt.bar(loc, mean, width, label=bar_labels[i], color=colors[i], zorder=3)
        plt.errorbar(loc, mean, std, capsize=3, linestyle='none', color='k', zorder=3)
        print(f'mean: {mean}')
        print(f'std: {std}')
    #    
    if y_label is not None:
        plt.ylabel(y_label, fontsize=labelfontsize)
    if title is not None:
        plt.title(title, fontsize=titlefontsize)
    # plt.legend(fontsize=legendfontsize)
    plt.legend(fontsize=10, ncol=5, columnspacing=1.0)
    fig = plt.gcf()
    fig.set_size_inches(8, 2.5)  
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig

def plot_compare_line(compare_data, bar_labels=None, xtick_labels=None, y_label=None, title=None, save_path=None):
    (nBars, nTicks) = compare_data.shape
    #
    x = np.arange(nTicks)  # the label locations
    width = 0.2
    plt.clf()
    fig, ax = plt.subplots()
    for i in range(nBars):
        rects = ax.bar(x - (nBars-1)*width/2+i*width, compare_data[i], width, label=bar_labels[i])
        ax.bar_label(rects, padding=2, fontsize=6)
    if y_label is not None:
        ax.set_ylabel(y_label)
    if title is not None:
        ax.set_title(title)
    ax.set_xticks(x)
    if xtick_labels is not None:
        ax.set_xticklabels(xtick_labels, fontsize=8)
    ax.legend(loc='lower right')
    #autolabel(ax, rects1)
    #autolabel(ax, rects2)
    fig.tight_layout()  
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')          

def plot_compare_model_AVG_withSTDbar(train, test, names=None, y_label=None, title=None, save_path=None):
    nModels = len(train)
    #
    x = np.arange(nModels)  # the label locations
    width = 0.2
    plt.clf()
    plt.errorbar(x - width/2, np.mean(train, 1, keepdims=False), np.std(train, 1, keepdims=False), capsize=3, linestyle='none', marker='o', label='train')
    plt.errorbar(x + width/2, np.mean(test, 1, keepdims=False), np.std(test, 1, keepdims=False), capsize=3, linestyle='none', marker='x', label='test')
    if y_label is not None:
        plt.ylabel(y_label, fontsize=labelfontsize)
    if title is not None:
        plt.title(title, fontsize=titlefontsize)
    plt.xticks(x)
    if names is not None:
        plt.xticks(ticks=x, labels=names, fontsize=tickfontsize)
    plt.legend(fontsize=legendfontsize)
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
def plot_compare_line_withSTDbar(compare_data, line_labels, xtick_labels, y_label=None, title=None, save_path=None):
    nLines = len(compare_data)
    width = 0.06
    plt.clf()
    for i in range(nLines):
        (nXticks, obvs) = compare_data[i].shape
        line_mean = np.mean(compare_data[i], 1, keepdims=False)
        line_std = np.std(compare_data[i], 1, keepdims=False)
        loc = np.arange(nXticks) - (nLines-1)*width/2 + i*width
        plt.plot(loc, line_mean, '-', marker=markers[i], color=colors[i], label=line_labels[i])
        for j in range(nXticks):
            plt.errorbar(j - (nLines-1)*width/2 + i*width, line_mean[j], line_std[j], capsize=3, linestyle='none', color=colors[i])
    #
    ticks = np.arange(len(xtick_labels))  # the label locations
    plt.xticks(ticks=ticks, labels=xtick_labels, fontsize=tickfontsize)
    plt.yticks(ticks=[0.5, 0.6, 0.7, 0.8, 0.9, 1], fontsize=tickfontsize)
    plt.ylim(0.5, 1)    
    if y_label is not None:
        plt.ylabel(y_label, fontsize=labelfontsize)
    if title is not None:
        plt.title(title, fontsize=titlefontsize)
    plt.legend(fontsize=legendfontsize)
    plt.grid(axis = 'y', linestyle='--', linewidth=1.0, zorder=3)
    fig = plt.gcf()
    fig.set_size_inches(8, 2.5)
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')        

def plot_average_and_fill_deviation(y, x, title, save_path, legends):
    if y.ndim > 3:
        raise ValueError("Inputs have more than 3 dimensions!")
    elif y.ndim==2:
        y = np.expand_dims(y, axis=0)
    elif y.ndim==1:
        y = np.expand_dims(y, axis=(0, 1))
    num_of_lines, num_of_objects, points = y.shape
    plt.clf()
    colors = plt.cm.jet(np.linspace(0,1, num_of_lines))
    for i in range(num_of_lines):
        mu = y[i].mean(axis=0, keepdims=False)
        std = y[i].std(axis=0, keepdims=False)
        plt.plot(x, mu, "-o", color=colors[i], linewidth=2.0, markersize=4, label=legends[i])
        plt.fill_between(x, mu+std, mu-std, facecolor=colors[i], alpha=0.3)
    plt.title(title)
    plt.xticks(x)
    plt.legend()        
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def autolabel(ax, rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

def plot_accuracy(valid_accs, test_accs, filename):
    valid_accs = np.array(valid_accs)
    test_accs = np.array(test_accs)
    avg_valid = valid_accs.mean()
    avg_test = test_accs.mean()
    valid_accs = np.append(valid_accs, [avg_valid], axis=0)
    test_accs = np.append(test_accs, [avg_test], axis=0)
    labels = []
    for i in range(len(valid_accs)):
        labels.append(f'{i}')
    labels[-1] = 'avg'
    #
    x = np.arange(len(valid_accs))  # the label locations
    width = 0.35
    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/2, valid_accs, width, label=f'valid: {valid_accs[-1]:.3f}')
    rects2 = ax.bar(x + width/2, test_accs, width, label=f'test: {test_accs[-1]:.3f}')
    ax.set_ylabel('Accuracy')
    ax.set_title('Leave-one-out accuracy')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    #autolabel(ax, rects1)
    #autolabel(ax, rects2)
    fig.tight_layout()    
    plt.savefig(filename, dpi=300, bbox_inches='tight')

def model_summary(model):
    print(model)
    summary(model, input_size=())
    
def permutation_test(x1, x2, n_iters:int=1000, tail=0, plot_hist=False) -> np.ndarray:
    """ Perform permutation test of 2 random variables
    Arguments:
        x1: Variable 1 of type Numpy array corresponding to condition 1. The last dimension is applied for permutation test.
        x2: Variable 2 of type Numpy array corresponding to condition 2. The last dimension is applied for permutation test.
    Returns:
        The numpy array of p values 
    """
    N1 = x1.shape[-1]
    N2 = x2.shape[-1]
    x1 = np.swapaxes(x1, 0, -1)
    x2 = np.swapaxes(x2, 0, -1)
    T_obs = np.mean(x1, axis=0, keepdims=True) - np.mean(x2, axis=0, keepdims=True)
    shuffled = np.concatenate((x1, x2), axis=0)    
    H0 = np.zeros((n_iters,) + T_obs.shape)
    for i in np.arange(n_iters):
        np.random.shuffle(shuffled)
        H0[i] = np.mean(shuffled[:N1,...], axis=0) - np.mean(shuffled[N1:,...], axis=0)
    if (tail==0):
        p_values = np.count_nonzero(np.abs(H0) >= np.abs(T_obs), axis=0)/n_iters
    elif (tail==-1):
        p_values = np.count_nonzero(H0 <= T_obs, axis=0)/n_iters
    elif (tail==1):
        p_values = np.count_nonzero(H0 >= T_obs, axis=0)/n_iters        
    
    p_values = np.swapaxes(p_values, 0, -1)
    if plot_hist:
        hist_data = H0.reshape(n_iters, -1)
        for i in range(hist_data.shape[1]):
            plt.figure(i)
            plt.hist(hist_data[:,i])
            plt.title(f'Sampling distribution of difference of means variables {i}')
            plt.show()
    return np.squeeze(p_values, axis=-1)    