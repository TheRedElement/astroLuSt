
#%%imports
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import pandas as pd
from typing import Union, Tuple, Callable



#%%definitions



def wandb_hypsearchplot(
    grid:list=None,
    save:Tuple[str,bool]=False,
    fig_kwargs:dict=None
    ):

    if fig_kwargs is None: fig_kwargs = {}

    

    idcol = 'param_name'
    df = pl.DataFrame(grid)
    names = df.select(pl.col(idcol))
    df = df.drop(idcol)

    df = df.select(pl.col(r'^param_.*$'))

    hyperparams = df.columns
    n_hyperparams = len(hyperparams)-1
    print(len(hyperparams))
    print(hyperparams)

    # for c in df.columns:
        
    #     #convert categorical columns to integers
    #     exp = (pl.col(c).rank('dense')/pl.col(c).rank('dense').max())
    #     df = df.with_columns(exp.alias(f'{c}_cat'))
    
    


    print(df)
    
    fig = plt.figure(**fig_kwargs)
    ax1 = fig.add_subplot(111)
    ax1.set_xmargin(0)
    ax1.spines[['top', 'bottom']].set_visible(False)
    ax1.spines[['top', 'bottom', 'left', 'right']].set_visible(False)

    # for config in grid:
    #     xvals = np.linspace(0,1,len(hyperparams))
    #     yvals = np.array(list(config.values()))
    #     yvals -= yvals.min()
    #     yvals /= (yvals.max()-yvals.min())
    #     ax1.plot(xvals, yvals)
    # for config in hyperparams:
    #     print(config)
    #     xvals = np.linspace(0,1,len(hyperparams))
    #     yvals = np.array(config)
    #     ax1.plot(xvals, yvals)
    # for row in df.select(pl.col(r'^.*_cat$')).iter_rows():
    
    lines = []
    for row, name in zip(df.iter_rows(), names.iter_rows()):
        yvals = row
        xvals = np.linspace(0,1,len(yvals))
        line, = ax1.plot(xvals, yvals, label=name[0])
        lines.append(line)
        # print(name)
    # print(ax1.get_yticks())
    # ax1.set_yticks(df[:,df.shape[0]//2+1].value_counts()[:,0])
    # ax1.set_yticklabels(df[:,0].value_counts()[:,0])

    labels = [item.get_text() for item in ax1.get_yticklabels()]
    for idx, row in enumerate(hyperparams):

        # print(df.select(pl.col(hyperparam)))
        # print(ax1.get_yticklabels())
        # cur_labs = np.array(ax1.get_yticklabels())[np.isin(labels, df.select(pl.col(hyperparam)))]
        # print(hyperparam)
        # print(cur_labs)
        axp = ax1.twiny()
        axp.xaxis.set_visible(False)
        # axp.spines[['right','top','bottom']].set_visible(False)
        axp.spines['right'].set_position(('axes', idx/n_hyperparams))
        axp.spines['right'].set_color('k')
        axp.tick_params(axis='y', colors='b')
        axp.spines['right'].set_linewidth(2)
        axp.yaxis.set_ticks_position('right')
        axp.yaxis.set_label_position('right')

        # axp.set_zorder(idx)
        # axp.patch.set_visible(False)
        # axp.set_yticks([2])

        # print(df[:,df.shape[0]//2+idx+1].value_counts()[:,0])
        # axp.set_yticks(df[:,df.shape[0]//2+idx+1].value_counts()[:,0])
        # axp.set_yticklabels(df[:,0+idx].value_counts()[:,0])
        # axp.set_zorder(idx)

    plt.tight_layout()
    plt.show()

    axs = fig.axes
    
    return fig, axs