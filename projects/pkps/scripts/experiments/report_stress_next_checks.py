"""Compact scientific figure for completed EG2 follow-up checks."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    root=Path('projects/pkps/artifacts/stress-test')
    out=root/'next-checks'
    additive=pd.read_csv(out/'additive_summary.csv')
    old=pd.read_csv(root/'query_summary.csv')
    sparse=pd.read_csv(out/'sparse_summary.csv')
    selective=pd.read_csv(out/'selective-16seed/summary.csv')
    style=pd.read_csv(out/'style_paired_changes.csv')
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    colors=['#2862ba','#489d9c','#bd6b2d','#817399']
    methods=['model-expanded','constant','additive-raw-fixed','additive-nested']
    labels=['PKPS selector','Constant kernel','Raw additive','Tuned additive']
    joined=pd.concat([old,additive],ignore_index=True)
    for k,(method,label) in enumerate(zip(methods,labels)):
        values=[float(joined[(joined.regime==regime)&(joined.m==1)&(joined.method==method)].mae.iloc[0])
                for regime in ['full-reference','cache-only']]
        axes[0,0].bar(np.arange(2)+(k-1.5)*.18,values,width=.18,label=label,color=colors[k])
    axes[0,0].set(xticks=[0,1],xticklabels=['Full reference','Cache only'],ylabel='Task macro MAE',
                  title='A. Query efficiency: m=1 per task, 16 seeds')
    axes[0,0].legend(fontsize=8)
    methods=['model-expanded','constant','additive','lrmc']
    labels=['PKPS selector','Constant kernel','Additive','Tuned LRMC']
    for k,(method,label) in enumerate(zip(methods,labels)):
        values=[float(sparse[(sparse.design==design)&(sparse.method==method)].mae.iloc[0])
                for design in ['natural','temporal-0.6','temporal-0.8']]
        axes[0,1].bar(np.arange(3)+(k-1.5)*.18,values,width=.18,label=label,color=colors[k])
    axes[0,1].set(xticks=[0,1,2],
                  xticklabels=['Natural graph\n189 cells/seed','March 13 cutoff\n51 future cells','April 18 cutoff\n95 future cells'],
                  ylim=(0,.105),ylabel='Task macro MAE',title='B. Sparse/temporal completion: m=16, 16 seeds')
    axes[0,1].legend(fontsize=8)
    for k,(method,label) in enumerate(zip(methods,labels)):
        values=[float(selective[(selective['mode']==mode)&(selective.method==method)].mae.iloc[0])
                for mode in ['uniform','hide-poor']]
        axes[1,0].bar(np.arange(2)+(k-1.5)*.18,values,width=.18,label=label,color=colors[k])
    axes[1,0].set(xticks=[0,1],xticklabels=['Uniform mask','Oracle hides poor scores'],
                  ylabel='Task macro MAE',title='C. Same hidden cells within each of 16 seeds')
    axes[1,0].legend(fontsize=8)
    for k,(method,label) in enumerate(zip(methods[:2],labels[:2])):
        values=[float(style[(style.variant==v)&(style.regime=='full-reference')&(style.method==method)].mean_absolute_prediction_change.iloc[0])
                for v in ['json','blockquote']]
        axes[1,1].bar(np.arange(2)+(k-.5)*.3,values,width=.3,label=label,color=colors[k])
    axes[1,1].set(xticks=[0,1],xticklabels=['JSON wrapper','Blockquote wrapper'],
                  ylabel='Mean absolute prediction change',
                  title='D. Target formatting: full reference, m=1\n44 models, 3 seeds; exact answer content preserved')
    axes[1,1].legend(fontsize=8)
    for ax in axes.flat:
        ax.set_axisbelow(True); ax.grid(axis='y',alpha=.15)
    fig.suptitle('EG2 follow-ups: conditional on the pinned EEE snapshot',fontsize=13)
    fig.savefig(out/'overview.png',dpi=180)
    fig.savefig(out/'overview.pdf')
    plt.close(fig)


if __name__=='__main__': main()
