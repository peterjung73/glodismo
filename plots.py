import matplotlib.cbook as cbook
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import glob
import os


def plot(ax, df, label, key, color):
    means =  np.mean([x[key].values for x in df], axis=0)
    stds =  np.std([x[key].values for x in df], axis=0)
    mins =  np.min([x[key].values for x in df], axis=0)
    maxs =  np.max([x[key].values for x in df], axis=0)
    
    # ax.plot(means, c=color, label=label)
    # ax.fill_between(range(len(means)), means - stds, means + stds, color=color, alpha=0.3, linewidth=0.0)
    ax.plot(means, label=label)
    ax.fill_between(range(len(means)), means - stds, means + stds, alpha=0.3, linewidth=0.0)

def plot_average(algs,key,experiment,resultsdir='results',traintest='train',suffix='Synthetic*_*'):
    fig_dpi=96
    fig_ext='png'
    fig_dir=resultsdir
    fig = plt.figure(figsize=(10.0, 3.0))
    axes1 = fig.add_subplot(1, 1, 1)
    for alg in algs:
        df=[]
        csvfiles=glob.glob(os.path.join(resultsdir+'/'+ 
            experiment+'_'+alg+'_'+suffix+'_'+traintest+'.csv'))
        for csvfile in csvfiles:    
            df.append(pd.read_csv(csvfile))
        plot(axes1,df,alg,key,"Red")
    axes1.set_ylabel(key)
    axes1.legend()
    fig.tight_layout();
    # fig.show();
    plt.savefig(fig_dir+'/average_'+experiment+'_'+key+'.'+fig_ext,dpi=fig_dpi,bbox_inches='tight')


def plot_history(algs,key,experiment,resultsdir='results',traintest='train',suffix='Synthetic*_*'):
    fig_dpi=96
    fig_ext='png'
    fig_dir=resultsdir
    fig = plt.figure(figsize=(10.0, 3.0))
    axes1 = fig.add_subplot(1, 1, 1)
    for alg in algs:
        msft = pd.concat(map(pd.read_csv, glob.glob(os.path.join(resultsdir+'/'+ 
            experiment+'_'+alg+'_'+suffix+'_'+traintest+'.csv'))))
        axes1.plot(msft[key].values,label=alg)
    axes1.set_ylabel(key)
    axes1.legend()
    fig.tight_layout();
    # fig.show();
    plt.savefig(fig_dir+'/history_'+experiment+'_'+key+'.'+fig_ext,dpi=fig_dpi,bbox_inches='tight')