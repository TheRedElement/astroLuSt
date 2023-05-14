
#TODO: cbar scale
#TODO: subplot with distribution of score_column - optional
#TODO: subplot with model-labels (i.e. legend?) - optional

#%%imports
from joblib.parallel import Parallel, delayed
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.figure import Figure
import numpy as np
import polars as pl
import re
from scipy.interpolate import interp1d
import time
from typing import Union, Tuple, List, Callable



#%%definitions
class WB_HypsearchPlot:

    def __init__(self,
        interpkind:str='quadratic',
        res:int=1000,
        ticks2display:int=5, tickcolor:Union[str,tuple]='tab:grey', ticklabelrotation:float=45,
        nancolor:Union[str,tuple]='tab:grey',
        linealpha:float=1, linewidth:float=1,
        base_cmap:Union[str,mcolors.Colormap]='plasma',
        n_jobs:int=1, sleep:float=0.1,
        verbose:int=0,
        ) -> None:
        
        
        self.interpkind         = interpkind
        self.res                = res
        self.ticks2display      = ticks2display
        self.tickcolor          = tickcolor
        self.ticklabelrotation  = ticklabelrotation
        self.nancolor           = nancolor
        self.linealpha          = linealpha
        self.linewidth          = linewidth
        self.base_cmap          = base_cmap
        self.n_jobs             = n_jobs
        self.sleep              = sleep
        self.verbose            = verbose

        
        return

    def __repr__(self) -> str:
        return (
            f'WB_HypsearchPlot(\n'
            f'    interpkind={self.interpkind},\n'
            f'    res={self.res},\n'
            f'    ticks2display={self.ticks2display}, tickcolor={self.tickcolor}, ticklabelrotation={self.ticklabelrotation},\n'
            f'    nancolor={self.nancolor},\n'
            f'    linealpha={self.linealpha},\n'
            f'    base_cmap={self.base_cmap},\n'
            f'    n_jobs={self.n_jobs}, sleep={self.sleep},\n'
            f'    verbose={self.verbose},\n'
            f')'
        )

    def make_new_cmap(self,
        cmap:mcolors.Colormap,
        nancolor:Union[str,tuple]='tab:grey'
        ) -> mcolors.Colormap:
        
        newcolors = cmap(np.linspace(0, 1, 256))
        c_nan = mcolors.to_rgba(nancolor)
        newcolors[:4] = c_nan                       #tiny fraction of the colormap gets attributed to nan values
        cmap = mcolors.ListedColormap(newcolors)

        return cmap

    def categorical2indices(self,
        df:pl.DataFrame, colname:str,
        ) -> Tuple[pl.DataFrame,np.ndarray]:
        
        #for numeric columns squeeze them into range(0,1) to be comparable accross hyperparams
        if str(df.select(pl.col(colname)).dtypes[0]) != 'Utf8' and len(df.select(pl.col(colname)).unique()) > 1:
            exp = (pl.col(colname)-pl.col(colname).min())/(pl.col(colname).max()-pl.col(colname).min())
        #for categorical columns convert them to unique indices in the range(0,1)
        else:
            #convert categorical columns to unique indices
            exp = (pl.col(colname).rank('dense')/pl.col(colname).rank('dense').max())
        
        #apply expression and get labels to put on yaxis
        df = df.with_columns(exp.alias(f'{colname}_cat'))
        ylabs = df.select(pl.col(colname)).unique().to_numpy().flatten()
        
        return df, ylabs

    def plot_model(self,
        hyperparams:Union[tuple,list], hyperparams_cat:Union[tuple,list],
        fill_value:float, name:str,
        ax1:plt.Axes,
        cmap:mcolors.Colormap, nancolor:Tuple[str,tuple]='tab:grey',
        resolution:int=1000, interpkind:str='quadratic',
        linealpha:float=1, linewidth:float=1,
        sleep=0,
        ) -> None:

        #interpolate using a spline to make everything look nice and neat (also ensures that the lines are separable by eye)
        xvals  = np.linspace(0,1,len(hyperparams))
        x_plot = np.linspace(0,1,resolution)
        yvals_func = interp1d(xvals, hyperparams_cat, kind=interpkind)
        y_plot = yvals_func(x_plot)
        
        #cutoff at upper/lower boundary to have no values out of bounds due to the interpolation
        y_plot[(y_plot>1)] = 1
        y_plot[(y_plot<0)] = 0
        
        #actually plot the line for current model (colomap according to cmap)
        if hyperparams[-1] == fill_value:
            line, = ax1.plot(x_plot, y_plot, label=name[0], alpha=linealpha, lw=linewidth, color=nancolor)
        else:
            line, = ax1.plot(x_plot, y_plot, label=name[0], alpha=linealpha, lw=linewidth, color=cmap(hyperparams_cat[-1]))

        time.sleep(sleep)

        return

    def add_hypaxes(self,
        ax1:plt.Axes,
        df:pl.DataFrame,
        ylabs:list,
        hyperparameter:str,
        fill_value:float,
        idx:int, n_hyperparams:int,
        tickcolor:Union[str,tuple]='tab:grey', ticks2display:int=5, ticklabelrotation:float=45,
        sleep=0,
        ) -> plt.Axes:

        #initialize new axis
        axp = ax1.twinx()
        
        #hide all spines but one (the one that will show the chosen values)
        axp.spines[['right','top','bottom']].set_visible(False)
        #position the axis to be aligned with the respective hyperparameter
        axp.spines['left'].set_position(('axes', (idx/n_hyperparams)))
        axp.yaxis.set_ticks_position('left')
        axp.yaxis.set_label_position('left')
        
        #hide xaxis because not needed in the plot
        axp.xaxis.set_visible(False)
        axp.set_xticks([])
        
        #additional formatting
        axp.spines['left'].set_color(tickcolor)
        axp.tick_params(axis='y', colors=tickcolor)
        
        #format the ticks differently depending on the datatype of the hyperparameter (i.e. is it numeric or not)
        
        ##numeric hyperparameter
        ###make sure to label the original nan values with nan
        if str(df.select(pl.col(hyperparameter)).dtypes[0]) != 'Utf8':

            #if only one value got tried, show a single tick of that value
            if len(df.select(pl.col(hyperparameter+'_cat')).unique()) == 1:
                axp.set_yticks([1.00])
                axp.set_yticklabels(
                    [lab if lab != fill_value else 'nan' for lab in df.select(pl.col(hyperparameter)).unique().to_numpy().flatten()],
                    rotation=ticklabelrotation
                )
            #'ticks2display' equidistant ticks for the whole value range
            else:
                axp.set_yticks(np.linspace(0, 1, ticks2display, endpoint=True))
                labs = np.linspace(np.nanmin(df.select(pl.col(hyperparameter)).to_numpy()), np.nanmax(df.select(pl.col(hyperparameter)).to_numpy()), ticks2display, endpoint=True)
                axp.set_yticklabels(
                    [f'{lab:.2f}' if lab != fill_value else 'nan' for lab in labs],
                    rotation=ticklabelrotation
                )

        ##non-numeric hyperparameter
        else:
            #get ticks
            axp.set_yticks(df.select(pl.col(hyperparameter+'_cat')).unique().to_numpy().flatten())
            #set labels to categories
            axp.set_yticklabels(ylabs, rotation=ticklabelrotation)


        #add spine labels (ylabs) on top  of each additional axis
        ax1.text(x=(idx/n_hyperparams), y=1.01, s=hyperparameter, rotation=45, transform=ax1.transAxes, color=tickcolor)

        time.sleep(sleep)

        return axp, hyperparameter

    def plot(self,
        grid:Union[pl.DataFrame,List[dict]],
        idcol:str='param_name',
        score_col:str='mean_test_score',
        param_cols:Union[str,list]=r'^param_.*$',
        min_score:float=-np.inf, max_score:float=np.inf, remove_nanscore:bool=False,
        score_scaling:str='pl.col(score_col)',
        interpkind:str=None,
        res:int=None,
        ticks2display:int=None, tickcolor:Union[str,tuple]=None, ticklabelrotation:float=None,
        nancolor:Union[str,tuple]=None,
        linealpha:float=None, linewidth:float=None,
        base_cmap:Union[str,mcolors.Colormap]=None,
        n_jobs:int=None, sleep:float=None,
        save:Union[str,bool]=False,
        show:bool=True,
        verbose:int=None,
        fig_kwargs:dict=None,
        ) -> Tuple[Figure, plt.Axes]:

        if interpkind is None:          interpkind          = self.interpkind
        if res is None:                 res                 = self.res
        if ticks2display is None:       ticks2display       = self.ticks2display
        if tickcolor is None:           tickcolor           = self.tickcolor
        if ticklabelrotation is None:   ticklabelrotation   = self.tickcolor
        if nancolor is None:            nancolor            = self.nancolor
        if linealpha is None:           linealpha           = self.linealpha
        if linewidth is None:           linewidth           = self.linewidth
        if base_cmap is None:           base_cmap           = self.base_cmap
        if n_jobs is None:              n_jobs              = self.n_jobs
        if sleep is None:               sleep               = self.sleep
        if verbose is None:             verbose             = self.verbose


        #initialize
        if fig_kwargs is None: fig_kwargs = {}

        #convert grid to polars DataFrame
        if isinstance(grid, pl.DataFrame):
            df = grid
        else:
            df = pl.DataFrame(grid)
        df_input_shape = df.shape[0]    #shape if input dataframe (for verbosity)
        
        #filter which range of scores to display and remove scores evaluating to nan if desired
        df = df.filter(((pl.col(score_col).is_between(min_score, max_score))|(pl.col(score_col).is_nan())))
        df_minmaxscore_shape = df.shape[0]  #shape of dataframe after score_col boundaries got applied
        
        if remove_nanscore:
            df = df.filter(pl.col(score_col).is_not_nan())
        df_nonan_shape = df.shape[0]    #shape of dataframe after nans got removed
        
        if verbose > 0:
            print(
                f'INFO(WB_HypsearchPlot): Removed\n'
                f'    {df_input_shape-df_minmaxscore_shape} row(s) via ({min_score} < {score_col} < {max_score}),\n'
                f'    {df_minmaxscore_shape-df_nonan_shape} row(s) containig nans,\n'
                f'    {df_input_shape-df_nonan_shape} row(s) total.\n'
            )


        #apply user defined expression to scale the score-function and thus color-scale
        df = df.with_columns(eval(score_scaling).alias(score_col))

        #get colormap
        ##check if there are nan in the 'score_col'
        if isinstance(base_cmap, str): cmap = plt.get_cmap(base_cmap)
        else: cmap = base_cmap

        if df.select(pl.col(score_col).is_nan().any()).item():
            cmap = self.make_new_cmap(cmap=cmap, nancolor=nancolor)

        #extract model-names, parameter-columns, ...
        names = df.select(pl.col(idcol))
        df = df.drop(idcol)
        df = df.select(pl.col(param_cols),pl.col(score_col))

        #deal with missing values (necessary because otherwise raises issues in plotting)
        fill_value = df.min()
        cols = np.array(fill_value.columns)[(np.array(fill_value.dtypes, dtype=str)!='Utf8')]
        fill_value = np.nanmin(fill_value.select(pl.col(cols)).to_numpy())-0.5                  #fill with value slightly lower than minimum of whole dataframe -> Ensures that nan end up in respective part of colormap

        df = df.fill_nan(fill_value=fill_value)

        #all fitted hyperparameters
        hyperparams = df.columns
        n_hyperparams = len(hyperparams)-1

        #generate labels for the axis ticks of categorical columns
        #convert categorical columns to unique integers for plotting - ensures that each hyperparameter lyes within range(0,1)
        ylabels = []
        for c in df.columns:
            df, ylabs = self.categorical2indices(df=df, colname=c)
            ylabels.append(ylabs)

                
        #plot        
        fig = plt.figure(**fig_kwargs)
        
        ##axis used for plotting models
        ax1 = fig.add_subplot(111)
        
        #no space between figure edge and plot
        ax1.set_xmargin(0)
        ax1.set_ymargin(0)

        #hide all spines ticks ect. of ax1, because custom axis showing correct ticks will be generated
        ax1.spines[['top', 'bottom', 'left', 'right']].set_visible(False)
        ax1.set_yticks([])
        
        #iterate over all rows in the dataframe (i.e. all fitted models) and plot a line for every model
        cat_cols = df.select(pl.col(r'^*_cat$')).columns
        # for idx, (row_cat, row, name) in enumerate(zip(df.select(cat_cols).iter_rows(), df.drop(cat_cols).iter_rows(), names.iter_rows())):
        Parallel(n_jobs, verbose=verbose, prefer='threads')(
            delayed(self.plot_model)(
                hyperparams=row, hyperparams_cat=row_cat,
                fill_value=fill_value, name=name,
                ax1=ax1,
                cmap=cmap, nancolor=nancolor,
                resolution=res, interpkind=interpkind,
                linealpha=linealpha, linewidth=linewidth,
                sleep=sleep,
            ) for idx, (row_cat, row, name) in enumerate(zip(df.select(cat_cols).iter_rows(), df.drop(cat_cols).iter_rows(), names.iter_rows()))
        )

        #plot one additional y-axis for every single hyperparameter
        res = Parallel(n_jobs=n_jobs, verbose=verbose, prefer='threads')(
            delayed(self.add_hypaxes)(
                ax1=ax1,
                df=df,
                ylabs=ylabs,
                hyperparameter=hyperparameter,
                fill_value=fill_value,
                idx=idx, n_hyperparams=n_hyperparams,
                tickcolor=tickcolor, ticks2display=ticks2display, ticklabelrotation=ticklabelrotation,
                sleep=sleep,
            ) for idx, (hyperparameter, ylabs) in enumerate(zip(hyperparams, ylabels))
        )
        axps, hyps = np.array(res)[:,0], np.array(res)[:,1]
        
        
        #add colorbar
        norm = mcolors.Normalize(vmin=df.select(pl.col(score_col)).min(), vmax=df.select(pl.col(score_col)).max())
        cbar = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=norm), ax=axps[hyps==score_col], pad=0.0005, drawedges=False, anchor=(0,0))
        cbar.outline.set_visible(False) #hide colorbar outline

        ##generate colorbar-label from score_scaling expression
        cbar_lab = re.sub(r'pl\.col\(score\_col\)', score_col, score_scaling)
        cbar_lab = re.sub(r'(np|pl|pd)\.', '', cbar_lab)
        cbar.set_label(cbar_lab, color=tickcolor)

        cbar.ax.set_zorder(0)
        cbar.ax.set_yticks([])  #use ticks from last subplot

        
        if isinstance(save, str): plt.savefig(save, bbox_inches='tight')
        # plt.tight_layout()
        if show: plt.show()

        axs = fig.axes
        
        return fig, axs
    
    def add_score_distribution(self
        ):
    
        return
    
    def add_model_legend(self,
        ):

        return
# %%
