#TODO: don't pass grid but pass 'id', 'coordinates', 'scores' -> ALSO NEEDS TO BE CHANGED IN plot_mode()
#TODO: add_distribution: only pass score_col and score_col__map


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
class ParallelCoordinates:
    """
        - class to display a plot of a previously executed hyperparameter search
        - inspired by the Weights&Biases Parallel-Coordinates plot 
            - https://docs.wandb.ai/guides/app/features/panels/parallel-coordinates (last access: 15.05.2023)

        Attributes
        ----------
            - show_idcol
                - bool, optional
                - whether to show a column encoding the ID of a particular run/model
                - only recommended if the number of models is not too large
                - the default is False
            - interpkind
                - str, optional
                - function to use for the interpolation between the different hyperparameters
                - argument passed as 'kind' to scipy.interpolate.interp1d()
                - the default is 'quadratic'
            - res
                - int, optional
                - resolution of the interpolated line
                    - i.e. the number of points used to plot each run/model
                - the default is 1000
            - axpos_coord
                - tuple, int, optional
                - position of where to place the axis containing the coordinate-plot
                - specification following the matplotlib convention
                    - will be passed to fig.add_subplot()
                - the default is None
                    - will use 121
            - axpos_hist
                - tuple, int, optional
                - position of where to place the axis containing the histogram of scorings
                - specification following the matplotlib convention
                    - will be passed to fig.add_subplot()
                - the default is None
                    - will use 164
            - map_suffix
                - str, optional
                - suffix to add to the columns created by mapping the coordinates onto the intervals [0,1]
                - only use if your column-names contain the default ('__map')
                - the default is '__map'
            - ticks2display
                - int, optional
                - number of ticks to show for numeric hyperparameters
                - the default is 5
            - tickcolor
                - str, tuple, optional
                - color to draw ticks and ticklabels in
                - if a tuple is passed it has to be a RGBA-tuple
                - the default is 'tab:grey'
            - ticklabelrotation
                - float, optional
                - rotation of the ticklabels
                - the default is 45
            - tickformat
                - str, optional
                - formatstring for the (numeric) ticklabels
                - the default is '%g'                
            - nancolor
                - str, tuple, optional
                - color to draw failed runs (evaluate to nan) in
                - if a tuple is passed it has to be a RGBA-tuple
                - the default is 'tab:grey'
            - nanfrac
                - float, optional
                - the fraction of the colormap to use for nan-values (i.e. failed runs)
                    - fraction of 256 (resolution of the colormap)
                - will also influence the number of bins/binsize used in the performance-histogram
                - a value between 0 and 1
                - the default is 4/256
            - linealpha
                - float, optional
                - alpha value of the lines representing runs/models
                - the default is 1
            - linewidth
                - float, optional
                - linewidth of the lines representing runs/models
                - the default is 1
            - base_cmap
                - str, mcolors.Colormap
                - colormap to map the scores onto
                - some space will be allocated for nan, if nans shall be displayed as well
                - the default is 'plasma'
            - cbar_over_hist
                - bool, optional
                - whether to plot a colorbar instead of the default performance-histogram
                - the histogram also has a colorbar encoded
                - the default is False
            - n_jobs
                - int, optional
                - number of threads to use when plotting the runs/modes
                - use for large hyperparameter-searches
                - argmument passed to joblib.Parallel
                - the default is 1
            - n_jobs_addaxes
                - int, optional
                - number of threads to use when plotting the axes for the different hyperparameters
                - use if a lot hyperparameters have been searched
                - argmument passed to joblib.Parallel
                - it could be that that a RuntimeError occurs if too many threads try to add axes at the same time
                    - this error should be caught with a try-except statement, but in case it something still goes wrong try setting 'n_jobs_addaxes' to 1
                - the default is 1
            - sleep
                - float, optional
                - time to sleep after finishing each job in plotting runs/models and hyperparameter-axes
                - the default is 0.1
            - verbose
                - int, optional
                - verbosity level
                - the default is 0

        Methods
        -------
            - make_new_cmap()
            - categorical2indices()
            - plot_model()
            - add_hypaxes()
            - add_score_distribution()
            - plot()

        Dependencies
        ------------
            - joblib
            - matplotlib
            - numpy
            - polars
            - re
            - scipy
            - time
            - typing

        Comments
        --------

    """


    def __init__(self,
        show_idcol:bool=True,                 
        interpkind:str='quadratic',
        res:int=1000,
        axpos_coord:Union[tuple,int]=None, axpos_hist:Union[tuple,int]=None,
        map_suffix:str='__map',
        ticks2display:int=5, tickcolor:Union[str,tuple]='tab:grey', ticklabelrotation:float=45, tickformat:str='%g',
        nancolor:Union[str,tuple]='tab:grey', nanfrac:float=4/256,
        linealpha:float=1, linewidth:float=1,
        base_cmap:Union[str,mcolors.Colormap]='plasma', cbar_over_hist:bool=False,
        n_jobs:int=1, n_jobs_addaxes:int=1, sleep:float=0.1,
        verbose:int=0,
        ) -> None:
        
        
        self.show_idcol         = show_idcol
        self.interpkind         = interpkind
        self.res                = res
        self.map_suffix         = map_suffix
        self.ticks2display      = ticks2display
        self.tickcolor          = tickcolor
        self.ticklabelrotation  = ticklabelrotation
        self.tickformat         = tickformat
        self.nancolor           = nancolor
        self.nanfrac            = nanfrac
        self.linealpha          = linealpha
        self.linewidth          = linewidth
        self.base_cmap          = base_cmap
        self.cbar_over_hist     = cbar_over_hist
        self.n_jobs             = n_jobs
        self.n_jobs_addaxes     = n_jobs_addaxes
        self.sleep              = sleep
        self.verbose            = verbose
        
        if axpos_coord is None:     self.axpos_coord    = (1,2,1)
        else:                       self.axpos_coord    = axpos_coord
        if axpos_hist is None:      self.axpos_hist     = (1,6,4)
        else:                       self.axpos_hist     = axpos_hist


        
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
        nancolor:Union[str,tuple]='tab:grey',
        nanfrac:float=4/256,
        ) -> mcolors.Colormap:
        """
            - method to generate a colormap allocating some space for nan-values
            - nan-values are represented by some fill-value

            Parameters
            ----------
                - cmap
                    - mcolors.Colormap
                    - template-colormap to use for the creation
            - nancolor
                - str, tuple, optional
                - color to draw values representing nan in
                - if a tuple is passed it has to be a RGBA-tuple
                - the default is 'tab:grey'
            - nanfrac
                - float, optional
                - the fraction of the colormap to use for nan-values (i.e. failed runs)
                    - fraction of 256 (resolution of the colormap)
                - will also influence the number of bins/binsize used in the performance-histogram
                - a value between 0 and 1
                - the default is 4/256

            Raises
            ------

            Returns
            -------
                - cmap
                    - mcolors.Colormap
                    - modified input colormap

            Comments
            --------
        """
        
        newcolors = cmap(np.linspace(0, 1, 256))
        c_nan = mcolors.to_rgba(nancolor)
        newcolors[:int(256*nanfrac)] = c_nan                       #tiny fraction of the colormap gets attributed to nan values
        cmap = mcolors.ListedColormap(newcolors)

        return cmap

    def col2range01(self,
        df:pl.DataFrame, colname:str,
        map_suffix:str='__map',
        ) -> Tuple[pl.DataFrame,np.ndarray]:
        """
            - method to map any column onto the interval [0,1]
            - a unique value for every unique element in categorical columns will be assigned to them
            - continuous columns will just be scaled to lye within the range

            Parameters
            ----------
                - df
                    - pl.DataFrame
                    - input dataframe
                        - the mapped column (name = original name with the appendix __map) will be append to it
                        - the new colum is the input column ('colname') mapped onto the interval [0,1]
                    - colname
                        - str
                        - name of the column to be mapped
                - map_suffix
                    - str, optional
                    - suffix to add to the columns created by mapping the coordinates onto the intervals [0,1]
                    - only use if your column-names contain the default ('__map')
                    - the default is '__map'

            Raises
            ------

            Returns
            -------
                - df
                    - pl.DataFrame
                    - input dataframe with additional column ('colname__map') appended to it

            Comments
            --------
        """
        
        coordinate = df.select(pl.col(colname)).to_series()
        #for numeric columns squeeze them into range(0,1) to be comparable accross hyperparams
        if coordinate.is_numeric() and len(coordinate.unique()) > 1:
            exp = (pl.col(colname)-pl.col(colname).min())/(pl.col(colname).max()-pl.col(colname).min())
       
        #for categorical columns convert them to unique indices in the range(0,1)
        else:
            #convert categorical columns to unique indices
            uniques = coordinate.unique()
            mapping = {u:m for u,m in zip(uniques.sort(), np.linspace(0,1,len(uniques)))}
            exp = coordinate.map_dict(mapping)
        
        #apply expression
        df = df.with_columns(exp.alias(f'{colname}{map_suffix}'))
        

        """
        #for numeric columns squeeze them into range(0,1) to be comparable accross hyperparams
        if str(df.select(pl.col(colname)).dtypes[0]) != 'Utf8' and len(df.select(pl.col(colname)).unique()) > 1:
            exp = (pl.col(colname)-pl.col(colname).min())/(pl.col(colname).max()-pl.col(colname).min())
        #for categorical columns convert them to unique indices in the range(0,1)
        else:
            #convert categorical columns to unique indices
            exp = (pl.col(colname).rank('dense')/pl.col(colname).rank('dense').max())
        
        #apply expression
        df = df.with_columns(exp.alias(f'{colname}{map_suffix}'))
        """

        return df

    #TODO: add param for score
    def plot_model(self,
        coordinates:Union[tuple,list], coordinates_map:Union[tuple,list],
        fill_value:float,
        ax:plt.Axes,
        cmap:mcolors.Colormap, nancolor:Tuple[str,tuple]='tab:grey',
        interpkind:str='quadratic', res:int=1000,
        linealpha:float=1, linewidth:float=1,
        sleep=0,
        ) -> None:
        """
            - method to add aline into the plot representing an individual run/model

            Parameters
            ----------
                - coordinates
                    - tuple, list
                    - iterable of hyperparameters specifying this particular run/model
                - coordinates_map
                    - tuple, list
                    - iterable of hyperparameters specifying this particular run/model mapped to the interval [0,1]
                - fill_value
                    - float
                    - value to use for plotting instead of nan
                    - i.e. value representing failed runs
                - ax
                    - plt.Axes
                    - axis to plot the line onto
                - cmap
                    - mcolor.Colormap
                    - colormap to use for coloring the lines
                - nancolor
                    - str, tuple, optional
                    - color to draw failed runs (evaluate to nan) in
                    - if a tuple is passed it has to be a RGBA-tuple
                    - the default is 'tab:grey'
                - interpkind
                    - str, optional
                    - function to use for the interpolation between the different hyperparameters
                    - argument passed as 'kind' to scipy.interpolate.interp1d()
                    - the default is 'quadratic'                
                - res
                    - int, optional
                    - res of the interpolated line
                        - i.e. the number of points used to plot each run/model
                    - the default is 1000
                - linealpha
                    - float, optional
                    - alpha value of the lines representing runs/models
                    - the default is 1
                - linewidth
                    - float, optional
                    - linewidth of the lines representing runs/models
                    - the default is 1    
                - sleep
                    - float, optional
                    - time to sleep after finishing each job in plotting runs/models and hyperparameter-axes
                    - the default is 0.1

            Raises
            ------

            Returns
            -------

            Comments
            --------
        """

        #interpolate using a spline to make everything look nice and neat (also ensures that the lines are separable by eye)
        xvals  = np.linspace(0,1,len(coordinates))
        x_plot = np.linspace(0,1,res)
        yvals_func = interp1d(xvals, coordinates_map, kind=interpkind)
        y_plot = yvals_func(x_plot)
        
        #cutoff at upper/lower boundary to have no values out of bounds due to the interpolation
        y_plot[(y_plot>1)] = 1
        y_plot[(y_plot<0)] = 0
        
        #actually plot the line for current model (colomap according to cmap)
        if coordinates[-1] == fill_value:
            line, = ax.plot(x_plot, y_plot, alpha=linealpha, lw=linewidth, color=nancolor)
        else:
            line, = ax.plot(x_plot, y_plot, alpha=linealpha, lw=linewidth, color=cmap(coordinates_map[-1]))

        time.sleep(sleep)

        return

    def add_coordaxes(self,
        ax:plt.Axes,
        coordinate:pl.Series, coordinate_map:pl.Series,
        fill_value:float,
        idx:int, n_coords:int,
        tickcolor:Union[str,tuple]='tab:grey', ticks2display:int=5, ticklabelrotation:float=45, tickformat:str='%g',
        sleep=0,
        ) -> plt.Axes:
        """
            - method to add a new axis for each coordinate (hyperparameter)
            - will move the spine to be aligned with the cordinates x-position in 'ax'

            Parameters
            ----------
                - ax
                    - plt.Axes
                    - axis to add the new axis to
                    - should be the axis used for plotting the runs/models
                - coordinate
                    - pl.Series
                    - series containing the coordinate to plot
                - coordinate_map
                    - pl.Series
                    - series containing the mapped values of 'coordinate'
                - fill_value
                    - float
                    - value to use for plotting instead of nan
                    - i.e. value representing failed runs
                - idx
                    - int
                    - index specifying which axis the current one is (i.e. 0 meaning it is the first axis, 1 the second, ect.)
                    - used to determine where to place the axis based on 'n_coords'
                        - position = idx//n_coords
                        - 0 meaning directly at the location of 'ax'
                        - 1 meaning at the far right of 'ax'
                - n_coords
                    - int
                    - number of coordinates to plot in total
                - ticks2display
                    - int, optional
                    - number of ticks to show for numeric hyperparameters
                    - the default is 5
                - tickcolor
                    - str, tuple, optional
                    - color to draw ticks and ticklabels in
                    - if a tuple is passed it has to be a RGBA-tuple
                    - the default is 'tab:grey'
                - ticklabelrotation
                    - float, optional
                    - rotation of the ticklabels
                    - the default is 45
                - tickformat
                    - str, optional
                    - formatstring for the (numeric) ticklabels
                    - the default is '%g'                   
                - sleep
                    - float, optional
                    - time to sleep after finishing each job in plotting runs/models and hyperparameter-axes
                    - the default is 0.1                    

            Raises
            ------

            Returns
            -------
                - axp
                    - plt.Axes
                    - newly created axis

            Comments
            --------
        """

        #initialize new axis
        axp = ax.twinx()
        
        #hide all spines but one (the one that will show the chosen values)
        axp.spines[['right','top','bottom']].set_visible(False)
        #position the axis to be aligned with the respective hyperparameter
        axp.spines['left'].set_position(('axes', (idx/n_coords)))
        axp.yaxis.set_ticks_position('left')
        axp.yaxis.set_label_position('left')
        
        #hide xaxis because not needed in the plot
        axp.xaxis.set_visible(False)
        axp.set_xticks([])
        
        #additional formatting
        axp.spines['left'].set_color(tickcolor)
        axp.tick_params(axis='y', colors=tickcolor)
        
        #format the ticks differently depending on the datatype of the coordinate (i.e. is it numeric or not)
        
        ##numeric coordinate
        ###make sure to label the original nan values with nan
        if coordinate.is_numeric():

            #if only one value got tried, show a single tick of that value
            if len(coordinate_map.unique()) == 1:
                axp.set_yticks([fill_value])
                axp.set_yticklabels(
                    [lab if lab != fill_value else 'nan' for lab in coordinate.unique().to_numpy().flatten()],
                    rotation=ticklabelrotation
                )
            #'ticks2display' equidistant ticks for the whole value range
            else:
                axp.set_yticks(np.linspace(0, 1, ticks2display, endpoint=True))
                labs = np.linspace(np.nanmin(coordinate.to_numpy()), np.nanmax(coordinate.to_numpy()), ticks2display, endpoint=True)
                axp.set_yticklabels(
                    [tickformat%lab if lab != fill_value else 'nan' for lab in labs],
                    rotation=ticklabelrotation
                )

        ##non-numeric hyperparameter
        else:
            #get ticks
            axp.set_yticks(coordinate_map.unique().to_numpy().flatten())
            #set labels to categories
            # axp.set_yticklabels(ylabs, rotation=ticklabelrotation)
            axp.set_yticklabels(coordinate.unique().sort().to_numpy().flatten(), rotation=ticklabelrotation)
        
        # print(ylabs)
        # print(coordinate.unique().to_numpy().flatten())
        
        #add spine labels (ylabs) on top  of each additional axis
        ax.text(x=(idx/n_coords), y=1.01, s=coordinate.name, rotation=45, transform=ax.transAxes, color=tickcolor)

        time.sleep(sleep)

        return axp

    #TODO: remove df from params
    def add_score_distribution(self,
        df:pl.DataFrame, score_col:str,
        nanfrac:float=4/256,
        lab:str=None, ticklab_color:Union[str,tuple]='tab:grey',
        cmap:Union[mcolors.Colormap,str]='plasma',
        fig:Figure=None, axpos:Union[tuple,int]=None,
        map_suffix:str='__map',
        ) -> None:
        """
            - method to add a distribution of scores into the figure
            - the distribution will be colorcoded to matcht he colormapping of the different runs/models

            Parameters
            ----------
                - df
                    - pl.DataFrame
                    - input dataframe to draw the score from
                - score_col
                    - str
                    - name of the column containing the score
                - nanfrac
                    - float, optional
                    - the fraction of the colormap to use for nan-values (i.e. failed runs)
                        - fraction of 256 (resolution of the colormap)
                    - will also influence the number of bins/binsize used in the performance-histogram
                    - a value between 0 and 1
                    - the default is 4/256
                - lab
                    - str, optional
                    - label to use for the y-axis of the plot
                    - the default is None
                - ticklabcolor
                    - str, tuple, optional
                    - color to draw ticks, ticklabels and axis labels in
                    - if a tuple is passed it has to be a RGBA-tuple
                    - the default is 'tab:grey'                    
                - cmap
                    - mcolor.Colormap, str
                    - colormap to apply to the plot for encoding the score
                    - the default is 'plasma'
                - fig
                    - Figure, optional
                    - figure to plot into
                    - the default is None
                        - will create a new figure
                - axpos
                    - tuplple, int
                    - axis position in standard matplotlib convention
                    - will be passed to fig.add_subplot()
                    - the default is None
                        - will use 111
                - map_suffix
                    - str, optional
                    - suffix to add to the columns created by mapping the coordinates onto the intervals [0,1]
                    - only use if your column-names contain the default ('__map')
                    - the default is '__map'
                
            Raises
            ------

            Returns
            -------

            Comments
            --------
        """
        
        if fig is None: fig = plt.figure()
        if axpos is None: axpos = 111

        #initialize new axis
        if isinstance(axpos, (int, float)):
            ax = fig.add_subplot(int(axpos))
        else:
            ax = fig.add_subplot(*axpos)
        ax.set_zorder(0)
        ax.set_ymargin(0)

        #use scaled score_col for plotting
        to_plot = df.select(pl.col(score_col+map_suffix))

        #adjust bins to colorbar
        bins = np.linspace(to_plot.min().item(), to_plot.max().item(), int(1//nanfrac))

        #get colors for bins
        if isinstance(cmap, str): cmap = plt.get_cmpa(cmap)
        colors = cmap(bins)
        
        #get histogram
        hist, bin_edges = np.histogram(to_plot, bins)

        #plot and colormap hostogram
        ax.barh(bin_edges[:-1], hist, height=nanfrac, color=colors)
        ax.set_xscale('symlog')

        #labelling
        ax.set_ylabel(lab, rotation=270, labelpad=15, va='bottom', ha='center', color=ticklab_color)
        ax.yaxis.set_label_position("right")
        
        ax.tick_params(axis='x', colors=ticklab_color)
        ax.spines[['top', 'left', 'right']].set_visible(False)
        ax.spines['bottom'].set_color(ticklab_color)
        ax.set_yticks([])
        ax.set_xlabel('Counts', color=ticklab_color)
        

        return
    

    def plot(self,
        grid:Union[pl.DataFrame,List[dict]],
        idcol:str=None,
        score_col:str='mean_test_score',
        param_cols:Union[str,list]=r'^param_.*$',
        min_score:float=None, max_score:float=None, remove_nanscore:bool=False,
        score_scaling:str='pl.col(score_col)',
        show_idcol:bool=None,
        interpkind:str=None,
        res:int=None,
        axpos_coord:tuple=None, axpos_hist:tuple=None,
        ticks2display:int=None, tickcolor:Union[str,tuple]=None, ticklabelrotation:float=None, tickformat:str=None,
        nancolor:Union[str,tuple]=None, nanfrac:float=None,
        linealpha:float=None, linewidth:float=None,
        base_cmap:Union[str,mcolors.Colormap]=None, cbar_over_hist:bool=None,
        n_jobs:int=None, n_jobs_addaxes:int=None, sleep:float=None,
        map_suffix:str=None,
        save:Union[str,bool]=False,
        show:bool=True,
        max_nretries:int=4,
        verbose:int=None,
        fig_kwargs:dict=None, save_kwargs:dict=None
        ) -> Tuple[Figure, plt.Axes]:
        """
            - method to create a coordinate plot 

            Parameters
            ----------

            Raises
            ------

            Returns
            -------

            Comments
            --------
        """
        if show_idcol is None:          show_idcol          = self.show_idcol
        if interpkind is None:          interpkind          = self.interpkind
        if res is None:                 res                 = self.res
        if axpos_coord is None:         axpos_coord         = self.axpos_coord
        if axpos_hist is None:          axpos_hist          = self.axpos_hist
        if ticks2display is None:       ticks2display       = self.ticks2display
        if tickcolor is None:           tickcolor           = self.tickcolor
        if ticklabelrotation is None:   ticklabelrotation   = self.ticklabelrotation
        if tickformat is None:          tickformat          = self.tickformat
        if nancolor is None:            nancolor            = self.nancolor
        if nanfrac is None:             nanfrac             = self.nanfrac
        if linealpha is None:           linealpha           = self.linealpha
        if linewidth is None:           linewidth           = self.linewidth
        if base_cmap is None:           base_cmap           = self.base_cmap
        if cbar_over_hist is None:      cbar_over_hist      = self.cbar_over_hist
        if n_jobs is None:              n_jobs              = self.n_jobs
        if sleep is None:               sleep               = self.sleep
        if map_suffix is None:          map_suffix          = self.map_suffix
        if n_jobs_addaxes is None:      n_jobs_addaxes      = self.n_jobs_addaxes
        if verbose is None:             verbose             = self.verbose

        if min_score is None: min_score = -np.inf
        if max_score is None: max_score =  np.inf

        #initialize
        if fig_kwargs is None:  fig_kwargs  = {}
        if save_kwargs is None: save_kwargs = {}

        #convert grid to polars DataFrame
        if isinstance(grid, pl.DataFrame):
            df = grid
        else:
            df = pl.DataFrame(grid)
        df_input_shape = df.shape[0]    #shape if input dataframe (for verbosity)
        
        #initialize idcol correctly (generate id none has been passed)
        if idcol is None:
            idcol = 'id'
            df = df.insert_at_idx(0, pl.Series(idcol, np.arange(0, df.shape[0], 1, dtype=np.int64)))
        
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
            cmap = self.make_new_cmap(cmap=cmap, nancolor=nancolor, nanfrac=nanfrac)

        #extract model-names, parameter-columns, ...
        names = df.select(pl.col(idcol))
        df = df.drop(idcol)                 #in case idcol is part of the searched parameters
        df = df.select(pl.col(param_cols),pl.col(score_col))
        
        #if desired also show the individual model-ids
        if show_idcol: df.insert_at_idx(0, names.to_series())

        #deal with missing values (necessary because otherwise raises issues in plotting)
        fill_value = df.min()
        cols = np.array(fill_value.columns)[(np.array(fill_value.dtypes, dtype=str)!='Utf8')]
        fill_value = np.nanmin(fill_value.select(pl.col(cols)).to_numpy())-0.5                  #fill with value slightly lower than minimum of whole dataframe -> Ensures that nan end up in respective part of colormap

        df = df.fill_nan(fill_value=fill_value)

        #all fitted hyperparameters
        coords = df.columns
        coords_map = [c+map_suffix for c in df.columns]
        n_coords = len(coords)-1

        #generate labels for the axis ticks of categorical columns
        #convert categorical columns to unique integers for plotting - ensures that each hyperparameter lyes within range(0,1)
        ylabels = []
        for c in df.columns:
            df = self.col2range01(df=df, colname=c, map_suffix=map_suffix)
            
            #get labels to put on yaxis
            ylabs = df.select(pl.col(c)).unique().to_numpy().flatten()
            ylabels.append(ylabs)

        from IPython.display import display
        display(df.select([pl.col(r'^param_.*$'),pl.col('mean_test_score')]))

        #plot        
        fig = plt.figure(**fig_kwargs)
        
        ##axis used for plotting models
        if isinstance(axpos_coord, (int, float)):
            ax1 = fig.add_subplot(int(axpos_coord))
        else:
            ax1 = fig.add_subplot(*axpos_coord)
        
        #no space between figure edge and plot
        ax1.set_xmargin(0)
        ax1.set_ymargin(0)

        #hide all spines ticks ect. of ax1, because custom axis showing correct ticks will be generated
        ax1.spines[['top', 'bottom', 'left', 'right']].set_visible(False)
        ax1.set_yticks([])
        
        #iterate over all rows in the dataframe (i.e. all fitted models) and plot a line for every model
        ##try except to retry if a RuntimeError occured
        map_cols = df.select(pl.col(r'^*%s$'%map_suffix)).columns
        e = True
        nretries = 0
        while e and nretries < max_nretries:
            try:
                Parallel(n_jobs, verbose=verbose, prefer='threads')(
                    delayed(self.plot_model)(
                        coordinates=row, coordinates_map=row_map,
                        fill_value=fill_value,
                        ax=ax1,
                        cmap=cmap, nancolor=nancolor,
                        interpkind=interpkind, res=res,
                        linealpha=linealpha, linewidth=linewidth,
                        sleep=sleep,
                    ) for idx, (row_map, row) in enumerate(zip(df.select(map_cols).iter_rows(), df.drop(map_cols).iter_rows()))
                )
                e = False
            except RuntimeError as err:
                e = True
                nretries += 1
                if verbose > 0:
                   print(
                        f'INFO(WB_HypsearchPlot):\n'
                        f'    The following error occured while plotting the models: {err}.\n'
                        f'    Retrying to plot. Number of elapsed retries: {nretries}.'
                    )

        #plot one additional y-axis for every single hyperparameter
        e = True
        nretries = 0
        while e and nretries < max_nretries:
            try:
                axps = Parallel(n_jobs=n_jobs_addaxes, verbose=verbose, prefer='threads')(
                    delayed(self.add_coordaxes)(
                        ax=ax1,
                        ylabs=ylabs,
                        coordinate=coordinate, coordinate_map=coordinate_map,
                        fill_value=fill_value,
                        idx=idx, n_coords=n_coords,
                        tickcolor=tickcolor, ticks2display=ticks2display, ticklabelrotation=ticklabelrotation, tickformat=tickformat,
                        sleep=sleep,
                    # ) for idx, (coordinate, ylabs) in enumerate(zip(coords, ylabels))
                    ) for idx, (coordinate, coordinate_map, ylabs) in enumerate(zip(df.select(pl.col(coords)), df.select(pl.col(coords_map)), ylabels))
                )
                e = False
            except RuntimeError as err:
                e = True
                nretries += 1
                if verbose > 0:
                    print(
                        f'INFO(WB_HypsearchPlot):\n'
                        f'    The following error occured while plotting the models: {err}.\n'
                        f'    Retrying to plot. Number of elapsed retries: {nretries}.'
                    )
        # axps, hyps = np.array(res)[:,0], np.array(res)[:,1]
        
        
        #generate score label score_scaling expression
        score_lab = re.sub(r'pl\.col\(score\_col\)', score_col, score_scaling)
        score_lab = re.sub(r'(np|pl|pd)\.', '', score_lab)
        
        if cbar_over_hist:
            #add colorbar
            norm = mcolors.Normalize(vmin=df.select(pl.col(score_col)).min(), vmax=df.select(pl.col(score_col)).max())
            cbar = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=norm), ax=axps[-1], pad=0.0005, drawedges=False, anchor=(0,0))
            cbar.outline.set_visible(False) #hide colorbar outline
            cbar.set_label(score_lab, color=tickcolor)


            cbar.ax.set_zorder(0)
            cbar.ax.set_yticks([])  #use ticks from last subplot

        else:
            #generate a histogram including the colorbar
            self.add_score_distribution(
                df=df, score_col=score_col,
                nanfrac=nanfrac,
                lab=score_lab, ticklab_color=tickcolor,
                cmap=cmap,
                fig=fig, axpos=axpos_hist,
                map_suffix=map_suffix,
            )
            plt.subplots_adjust(wspace=0)
        


        if isinstance(save, str): plt.savefig(save, bbox_inches='tight', **save_kwargs)
        # plt.tight_layout()    #moves cbar around


        axs = fig.axes
        
        return fig, axs
    
