import os
import argparse
import matplotlib.pyplot as plt
import scipy.io as sio
import numpy as np
from matplotlib.axis import Axis
from matplotlib.ticker import FormatStrFormatter

parser = argparse.ArgumentParser(description='')
parser.add_argument('--filename', default='stats/fixed/SNIP/mnist_ffn.mat', type=str)
parser.add_argument('--stat_dir', default='stats/fixed/', type=str)
parser.add_argument('--save_dir', default='upload/fixed/', type=str)
args = parser.parse_args()
    

def main():
    stats = sio.loadmat(args.filename)
    upload = 'upload' in args.save_dir
    if upload:
        save_dir = args.save_dir
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        filename = args.filename.replace(args.stat_dir,'')
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
    
    
    num_impt = stats['num_impt'][0]
    num_impt_ratio = stats['num_impt_ratio'][0]
    # bins_num = create_bins(num_impt, num_impt_ratio)
    max_len = max(max(num_impt)-min(num_impt), max(num_impt_ratio)-min(num_impt_ratio))
    eps = max_len*0.1
    bins_num1 = np.linspace(min(num_impt)-2*eps,min(num_impt)+max_len+eps, 25)
    bins_num2 = np.linspace(min(num_impt_ratio)-eps,min(num_impt_ratio)+max_len+2*eps, 25)
    
   
    counts_num, _ = np.histogram(num_impt, bins_num1)
    counts_num_ratio, _ = np.histogram(num_impt_ratio, bins_num2)

    
    expected_num = stats['expected_num'][0]
    expected_num_ratio = stats['expected_num_ratio'][0]
    expected_mean = stats['expected_mean'][0]
    expected_mean_ratio = stats['expected_mean_ratio'][0]


    plt.figure(constrained_layout=True, figsize=(width, height))
    f,(ax,ax2) = plt.subplots(1,2,sharey=True, facecolor='w')
    
    ax.stairs(counts_num_ratio, bins_num2, fill=True, color='tab:orange', alpha=0.8, label='UNI')
    ax.axvline(x=expected_num_ratio, color='r', linestyle='--')

    ax2.stairs(counts_num, bins_num1, fill=True, alpha=0.8, label='PM')
    ax2.axvline(x=expected_num, color='b', linestyle='--')

    ax.set_xlim(bins_num2[0], bins_num2[-1])
    ax2.set_xlim(bins_num1[0], bins_num1[-1])
    
    ax.spines['right'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    ax.yaxis.tick_left()
    ax2.tick_params(labelleft='off')
    ax2.yaxis.tick_right()    

    d = .015 
    kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
    ax.plot((1-d,1+d), (-d,+d), **kwargs)
    ax.plot((1-d,1+d),(1-d,1+d), **kwargs)

    kwargs.update(transform=ax2.transAxes)  
    ax2.plot((-d,+d), (1-d,1+d), **kwargs)
    ax2.plot((-d,+d), (-d,+d), **kwargs)    
    
    
    # ax.set_rasterized(True)
    # ax2.set_rasterized(True)
    
    fig = plt.gcf()
    fig.text(0.5, -0.12, '# impt. weights', ha='center', fontsize=28)
    ax.set_ylabel('Count')
    ax.legend(loc='upper right')
    ax2.legend(loc='upper right')
    # ax.ticklabel_format(style='sci',scilimits=(0,1))
    # ax2.ticklabel_format(style='sci',scilimits=(0,1))
    ax.set_xticks([bins_num2[0], bins_num2[15]])
    ax2.set_xticks([bins_num1[10], bins_num1[24]])
    dec_point = np.log10(bins_num2[0])//1
    ax.xaxis.set_major_formatter(FormatStrFormatter('%{}f'.format(dec_point)))
    ax2.xaxis.set_major_formatter(FormatStrFormatter('%{}f'.format(dec_point)))
    plt.savefig('{}num.{}'.format(filename, image_extension), bbox_inches='tight')
    plt.close()    




def adjust_font_size(save_dir):
    SMALL_SIZE = 23
    BIG_SIZE = 28
    if 'yolk' in save_dir: SMALL_SIZE = 28
    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=SMALL_SIZE)  # fontsize of the figure title
    plt.rc('axes', titlesize=BIG_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=BIG_SIZE)    # fontsize of the x and y labels    
    # plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right') 


def create_bins(x1, x2, nbins=75):
    min_x = min(min(x1), min(x2))
    max_x = max(max(x1), max(x2))
    eps = min(abs(min_x),abs(max_x)) * 1e-5
    bins = np.linspace(min_x-eps, max_x+eps, nbins)
    return bins
    
    
if __name__ == '__main__':
    main()