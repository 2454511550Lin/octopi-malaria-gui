import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc
from tqdm import tqdm
from sklearn.metrics import ConfusionMatrixDisplay

cm = 1/2.54 

# Global variable to control figure size
MANUSCRIPT_MODE = True

# Helper function to set figure properties based on mode
def set_figure_properties(fig, ax, fontsize, linewidth):
    if MANUSCRIPT_MODE:
        ax.tick_params(width=linewidth, length=2*linewidth)
        for spine in ax.spines.values():
            spine.set_linewidth(linewidth)
        plt.rcParams['font.size'] = fontsize
        plt.rcParams['axes.linewidth'] = linewidth
    else:
        # Default settings for better visibility
        plt.rcParams['font.size'] = 12
        plt.rcParams['axes.linewidth'] = 1.5
        ax.tick_params(width=1.5, length=6)
        # tight layout


def plot_confusion_matrix(TP, FP, TN, FN, out_dir):
    """
    Plots the confusion matrix with raw numbers and saves it to a file.
    """
    if MANUSCRIPT_MODE:
        figsize = (5.8, 5.5)  # in cm
        fontsize = 6
        ticksize = 5
        linewidth = 1 # the plot line width
        axislinewidth = 0.5 # the axis line width
    else:
        figsize = (10, 8)  # in inches
        fontsize = 12
        linewidth = 1.5
    
    fig, ax = plt.subplots(figsize=(figsize[0]/2.54, figsize[1]/2.54))
    
    cm = np.array([[TP, FN], [FP, TN]])
    
    # Normalize the confusion matrix for color mapping
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    print(cm_norm)
    
    im = ax.imshow(cm_norm, interpolation='nearest', cmap='Blues', aspect='auto')
    cbar = fig.colorbar(im, ax=ax)
    
    # colorbar ticks line width
    cbar.outline.set_linewidth(axislinewidth)
    cbar.ax.tick_params(labelsize=ticksize, width=axislinewidth)
        
    ax.set(xticks=[0, 1],
           yticks=[0, 1],
           xticklabels=['P', 'N'], 
           yticklabels=['P', 'N'])
    
    ax.set_xlabel('Predicted Label', fontsize=fontsize)
    ax.set_ylabel('True Label', fontsize=fontsize)
    
    # Display raw numbers in the cells
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f'{cm[i, j]}',
                    ha="center", va="center",
                    color="white" if cm_norm[i, j] > 0.5 else "black",
                    #color = 'black',
                    fontsize=fontsize)
    
    # Set figure properties (assuming this function exists)
    if 'set_figure_properties' in globals():
        set_figure_properties(fig, ax, fontsize, linewidth)
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'confusion_matrix.png'), dpi=300)
    plt.close()


TP = 43  # True Positives
FP = 0  # False Positives
TN = 19   # True Negatives
FN = 0   # False Negatives
out_dir = './'

confusion_matrix = plot_confusion_matrix(TP, FP, TN, FN, out_dir)