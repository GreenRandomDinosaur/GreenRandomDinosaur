import matplotlib.pyplot as plt
import scipy.io as sio
import numpy as np



def draw_histogram(x, filename, nbins=50, title=None, xlabel=None, ylabel=None):
    counts1, bins1 = np.histogram(x, nbins)
    plt.stairs(counts1, bins1, fill=True)
    if xlabel is not None: plt.xlabel('{}'.format(xlabel))
    if ylabel is not None: plt.ylabel('{}'.format(ylabel))
    plt.savefig('{}'.format(filename))
    plt.close()