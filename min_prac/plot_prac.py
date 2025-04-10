import os
import argparse
import matplotlib.pyplot as plt
import scipy.io as sio
import numpy as np
from matplotlib.ticker import ScalarFormatter


parser = argparse.ArgumentParser(description='')
parser.add_argument('--filename', default='stats/avg/SNIP/mnist_ffn.mat', type=str)
parser.add_argument('--stat_dir', default='stats/avg', type=str)
parser.add_argument('--save_dir', default='images/avg/', type=str)
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
        filename = '{}/{}'.format(save_dir, filename)
    else:    
        filename = args.filename.split('/')[-1]
        save_dir = args.filename.replace(args.stat_dir, args.save_dir)
        save_dir = save_dir.replace(filename,'')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        filename = filename.replace('.mat','')
        filename = '{}/{}'.format(save_dir, filename)
    
    if upload: image_extension = 'eps'
    else: image_extension = 'jpg'

    nums = [stats['rand_uni_num'][0][0], stats['rand_max_num'][0][0], stats['max_num'][0][0]]
    scores = [stats['rand_uni_score'][0][0], stats['rand_max_score'][0][0], stats['max_score'][0][0]]
    opts =  [stats['rand_uni_opt'][0][0], stats['rand_max_opt'][0][0], stats['max_opt'][0][0]]
        
    width, height = 6.4, 4.8
    adjust_font_size(args.save_dir)    
    
    # num
    plt.figure(figsize=(width, height))

    f, (ax, ax2) = plt.subplots(1, 2, sharey=True, constrained_layout=True)
    ax.plot(nums, opts, marker='o', linestyle='--')
    ax2.plot(nums, opts, marker='o', linestyle='--')
    
    ax2.set_xlim(nums[2]-nums[0], nums[2]+nums[0])  
    ax.set_xlim(0, nums[0]+nums[1])      
    
    ax.spines['right'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    ax.yaxis.tick_left()
    ax.tick_params(labeltop=False)  
    ax2.yaxis.tick_right()
        
            
    d = .015 
    kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
    ax.plot((1-d,1+d), (-d,+d), **kwargs)
    ax.plot((1-d,1+d),(1-d,1+d), **kwargs)

    kwargs.update(transform=ax2.transAxes)  
    ax2.plot((-d,+d), (1-d,1+d), **kwargs)
    ax2.plot((-d,+d), (-d,+d), **kwargs)

    x_scale1 = (nums[1]-nums[0])*0.1
    x_scale2 = nums[0]*0.1
    y_scale = (opts[2]-opts[0])*0.1

   
    ax.annotate(r'(rand, $\bf{n}_u$)',(nums[0], opts[0]),fontsize=25,xytext=(nums[0], opts[0]-2*y_scale))
    ax.annotate(r'(rand, $\bf{n}^*$)',(nums[1], opts[1]),fontsize=25,xytext=(nums[1], opts[1]+y_scale))
    ax2.annotate(r'(max, $\bf{n}^*$)',(nums[2], opts[2]),fontsize=25,xytext=(nums[2]-7*x_scale2, opts[2]+y_scale)) 
    if 'mnist' in args.filename:    
        ax.annotate(r'${:.2f}$'.format(opts[0]),(nums[0], opts[0]),fontsize=25, color='grey', xytext=(nums[0], opts[0]-4.5*y_scale))
        ax.annotate(r'${:.2f}$'.format(opts[1]),(nums[1], opts[1]),fontsize=25, color='grey', xytext=(nums[1], opts[1]+3.5*y_scale))
        ax2.annotate(r'${:.2f}$'.format(opts[2]),(nums[2], opts[2]),fontsize=25, color='grey', xytext=(nums[2]-7*x_scale2, opts[2]+3.5*y_scale))   
    ax.margins(0.7)
    ax2.margins(0.7)
    
    ax.set_xlabel('Number of impt. weights', loc='left')
    if 'SNIP' in args.filename: ax.set_ylabel(r'$\Delta$ Loss') #r"$x_{\Delta T}$"
    elif 'GraSP_abs' in args.filename: ax.set_ylabel(r'$\Delta$ Grad. norm')
    else: ax.set_ylabel('Grad. norm')
    
    plt.setp(ax.get_xticklabels(), rotation=90, horizontalalignment='right')
    plt.setp(ax2.get_xticklabels(), rotation=90, horizontalalignment='right')
    ax.xaxis.labelpad = 20

    plt.savefig('{}_prac_num.{}'.format(filename, image_extension))
    plt.close()
  

def adjust_font_size(save_dir):
    SMALL_SIZE = 23
    BIG_SIZE = 28
    if 'yolk' in save_dir: SMALL_SIZE = 25
    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=BIG_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=BIG_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=BIG_SIZE)    # legend fontsize
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
    
    
class ScalarFormatterForceFormat(ScalarFormatter):
    def _set_format(self):  # Override function that finds format to use.
        self.format = "%1.1f" 


if __name__ == '__main__':
    main()