import os
import argparse
import matplotlib.pyplot as plt
import scipy.io as sio
import numpy as np

parser = argparse.ArgumentParser(description='')
parser.add_argument('--filename', default='stats/bn_train/SNIP/mnist_ffn.mat', type=str)
parser.add_argument('--stat_dir', default='stats', type=str)
parser.add_argument('--save_dir', default='upload/exp', type=str)
args = parser.parse_args()
    

def main():
    stats = sio.loadmat(args.filename)
    upload = 'upload' in args.save_dir
    if upload:
        save_dir = args.save_dir
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        filename = args.filename.replace(args.stat_dir+'/','')
        filename = filename.replace('/', '_')
        filename = filename.replace('.mat', '')
        filename = '{}/{}_'.format(save_dir, filename)
    else:
        save_dir = args.filename.replace(args.stat_dir, args.save_dir)
        save_dir = save_dir.replace('.mat', '')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        filename = '{}/'.format(save_dir)
    
    if upload: image_extension = 'eps'
    else: image_extension = 'jpg'
    
    width, height = 6.4, 4.8
    adjust_font_size(args.save_dir)
    
    # num
    plt.figure(constrained_layout=True, figsize=(width, height))
    plt.plot(stats['num_impt'][0], ':o', label='PM')
    plt.plot(stats['num_impt_ratio'][0], ':o', label='UNI')
    plt.xlabel('Trial index')
    plt.ylabel('# impt. weights')
    # plt.legend(loc='upper right')
    plt.legend()
    plt.savefig('{}num.{}'.format(filename, image_extension))
    plt.close()
    
    '''
    # mean
    ms = stats['mean_score'][0]
    ms_ratio = stats['mean_score_ratio'][0]
    s_th = stats['threshold_val'][0]

    
    plt.figure(constrained_layout=True, figsize=(width, height))
    plt.plot(ms, ':o', label='from pruning method')
    plt.plot(ms_ratio, ':o', label='uniform')
    plt.xlabel('trial index')
    plt.ylabel('mean score')
    plt.legend(loc='upper right')
    plt.savefig('{}mean.{}'.format(filename, image_extension))
    plt.close()
    
    
    plt.figure(figsize=(width, height))
    fig, ax1 = plt.subplots(constrained_layout=True)
    ax2 = ax1.twinx()
    ax1.plot(ms, ':o', label='from pruning method')
    ax1.plot(ms_ratio, ':o', label='uniform')
    # ax2.plot(s_th, color='r', linestyle='-', alpha=0.3, label='threshold value')
    ax2.plot(s_th, color='lightcoral', linewidth=0.6, label='threshold value')
    ax1.set_xlabel('trial index')
    ax1.set_ylabel('average score')
    ax2.set_ylabel('threshold score ($S_{th}$)')
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    plt.savefig('{}mean_th.{}'.format(filename, image_extension))
    plt.close()
    
    
    plt.figure(constrained_layout=True, figsize=(width, height))
    plt.plot(ms/ms_ratio, ':o', label='from pruning method')
    plt.xlabel('trial index')
    plt.ylabel('ratio')
    plt.savefig('{}mean_ratio.{}'.format(filename, image_extension))
    plt.close()

    plt.figure(constrained_layout=True, figsize=(width, height))
    plt.plot(ms-ms_ratio, ':o', label='from pruning method')
    plt.xlabel('trial index')
    plt.ylabel('difference')
    plt.savefig('{}mean_diff.{}'.format(filename, image_extension))
    plt.close() 
    '''
    

def adjust_font_size(save_dir):
    SMALL_SIZE = 23
    BIG_SIZE = 28
    if 'yolk' in save_dir: SMALL_SIZE = 28
    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=BIG_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=BIG_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=SMALL_SIZE)  # fontsize of the figure title
    # plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')    
    
    
def extract_mat(m):
    rst_m = []
    for i in range(len(m)):
        j=0
        while j<i:
            rst_m.append(m[i][j])
            j += 1
    return rst_m


def centered_bins(x, nbins=50):
    bins = np.linspace(min(x), max(x), nbins-1)
    bins_abs = np.absolute(bins)
    idx = np.argmin(bins_abs)
    offset = bins[idx]
    bins = bins - offset
    if offset<0:
        new_bins = np.array([bins[0]-(bins[1]-bins[0])])
        new_bins = np.concatenate((new_bins, bins))
    elif offset>0:
        new_bins = np.append(bins, bins[-1]+bins[-1]-bins[-2])
    else: 
        new_bins = bins
    return new_bins
    
    
if __name__ == '__main__':
    main()