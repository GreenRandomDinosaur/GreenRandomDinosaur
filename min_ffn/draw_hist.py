import numpy as np
from matplotlib import pyplot as plt
from matplotlib import scale


def draw_histogram(args, fname, empi, semi, n_bins = 50):
    scores1, scores2 = empi, semi

    bins_min = min(min(scores1), min(scores2))
    bins_max = max(max(scores1), max(scores2))
    bins = np.linspace(bins_min, bins_max, n_bins)
    
    counts1, _ = np.histogram(scores1, bins)
    counts2, _ = np.histogram(scores2, bins)
    
    counts1 = [c/len(scores1) for c in counts1]
    counts2 = [c/len(scores2) for c in counts2]


    width, height = 6.4, 4.8
    adjust_font_size(args.save_dir)
    
    plt.figure(constrained_layout=True, figsize=(width, height))
    plt.stairs(counts1, bins, fill=True, alpha=0.5, label='EMPI')
    plt.stairs(counts2, bins, fill=True, alpha=0.5, color='r', label='SEMI')       
    # plt.yscale('log')
    plt.yscale(scale.LogScale(plt.gca(),base=2))
    
    ax = plt.gca()
    ax.set_rasterized(True)       
    
    # plt.ticklabel_format(scilimits=(0,2))
    # plt.ylabel('Proportion')
    plt.ylabel('Counts')
    plt.xlabel('Score')
    plt.legend(loc='upper right')
    if 'upload' in args.save_dir: 
        plt.savefig('{}/{}.eps'.format(args.save_dir, fname))
    else: 
        plt.savefig('{}/{}.jpg'.format(args.save_dir, fname))
    plt.close() 


def adjust_font_size(save_dir):
    SMALL_SIZE = 28
    # SMALL_SIZE = 23
    if 'yolk' in save_dir: SMALL_SIZE = 28
    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=SMALL_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=SMALL_SIZE)  # fontsize of the figure title
    # plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')
