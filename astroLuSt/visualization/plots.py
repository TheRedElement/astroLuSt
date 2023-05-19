

#%%imports
from joblib.parallel import Parallel, delayed
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator
import numpy as np
import polars as pl
import re
from scipy.interpolate import interp1d
from scipy import stats
from sklearn.neighbors import KNeighborsClassifier
import time
from typing import Union, Tuple, List, Callable
import warnings



#%%classes
class ParallelCoordinates:
    """
        - class to create a Prallel-Coordinate plot
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
                - function to use for the interpolation between the different coordinates
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
                - number of ticks to show for numeric coordinates
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
                - number of threads to use when plotting the runs/models
                - use for large coordinate-plots
                - argmument passed to joblib.Parallel
                - the default is 1
            - n_jobs_addaxes
                - int, optional
                - number of threads to use when plotting the axes for the different coordinates
                - use if a lot coordinates shall be plotted
                - argmument passed to joblib.Parallel
                - it could be that that a RuntimeError occurs if too many threads try to add axes at the same time
                    - this error should be caught with a try-except statement, but in case it something still goes wrong try setting 'n_jobs_addaxes' to 1
                - the default is 1
            - sleep
                - float, optional
                - time to sleep after finishing each job in plotting runs/models and coordinate-axes
                - the default is 0.1
            - verbose
                - int, optional
                - verbosity level
                - the default is 0
            - text_kwargs
                - dict, optional
                - kwargs passed to ax.text()
                    - affect the labels of the created subplots for each coordinate
                - if the passed dict does not contain 'rotation', 'rotation':45 will be added
                - the default is None
                    - will be initialized with {'rotation':45}                

        Methods
        -------
            - make_new_cmap()
            - col2range01()
            - plot_model()
            - add_coordaxes()
            - add_score_distribution()
            - __init_scorecol()
            - __deal_with_nan()
            - __deal_with_inf()
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
        text_kwargs:dict=None,
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

        if text_kwargs is None: self.text_kwargs = {'rotation':45}
        else:
            self.text_kwargs = text_kwargs
            if 'rotation' not in self.text_kwargs.keys():
                self.text_kwargs['rotation'] = 45

        
        return

    def __repr__(self) -> str:
        return (
            f'WB_HypsearchPlot(\n'
            f'    interpkind={self.interpkind},\n'
            f'    res={self.res},\n'
            f'    axpos_coord={self.axpos_coord}, axpos_hist={self.axpos_hist},\n'
            f'    map_suffix={self.map_suffix},\n'
            f'    ticks2display={self.ticks2display}, tickcolor={self.tickcolor}, ticklabelrotation={self.ticklabelrotation}, tickformat={self.tickformat},\n'
            f'    nancolor={self.nancolor}, nanfrac={self.nanfrac},\n'
            f'    linealpha={self.linealpha}, linewidth={self.linewidth},\n'
            f'    base_cmap={self.base_cmap}, cmap_over_hist={self.cbar_over_hist},\n'
            f'    n_jobs={self.n_jobs}, n_jobs_addaxes={self.n_jobs_addaxes}, sleep={self.sleep},\n'
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
        ) -> pl.DataFrame:
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
                    - input dataframe with additional column ('colname'+'map_suffix') appended to it

            Comments
            --------
        """
                
        coordinate = df.select(pl.col(colname)).to_series()

        #for numeric columns squeeze them into range(0,1) to be comparable accross hyperparams
        if coordinate.is_numeric() and len(coordinate.unique()) > 1:
            exp = (pl.col(colname)-pl.col(colname).min())/(pl.col(colname).max()-pl.col(colname).min())
       
        #for categorical columns convert them to unique indices in the range(0,1)
        else:
            # coordinate = coordinate.fill_null('None')
            #convert categorical columns to unique indices
            uniques = coordinate.unique()
            mapping = {u:m for u,m in zip(uniques.sort(), np.linspace(0,1,len(uniques)))}
            exp = coordinate.map_dict(mapping)
        
        #apply expression
        df = df.with_columns(exp.alias(f'{colname}{map_suffix}'))
        
        return df

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
                    - iterable of coordinates specifying this particular run/model
                - coordinates_map
                    - tuple, list
                    - iterable of coordinates specifying this particular run/model mapped to the interval [0,1]
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
                    - function to use for the interpolation between the different coordinates
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
                    - time to sleep after finishing each job in plotting runs/models and coordinate-axes
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
        ticks2display:int=5, tickcolor:Union[str,tuple]='tab:grey', ticklabelrotation:float=45, tickformat:str='%g',
        sleep=0,
        text_kwargs:dict=None,
        ) -> plt.Axes:
        """
            - method to add a new axis for each coordinate
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
                    - number of ticks to show for numeric coordinates
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
                    - time to sleep after finishing each job in plotting runs/models and coordinate-axes
                    - the default is 0.1
                - text_kwargs
                    - dict, optional
                    - kwargs passed to ax.text()
                        - affect the labels of the created subplots for each coordinate
                    - the default is None
                        - will be initialized with an empty dict

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

        if text_kwargs is None: text_kwargs = {}

        #initialize new axis
        axp = ax.twinx()
        
        #hide all spines but one (the one that will show the chosen values)
        axp.spines[['right','top','bottom']].set_visible(False)
        #position the axis to be aligned with the respective coordinate
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

        ##non-numeric coordinate
        else:
            #get ticks
            axp.set_yticks(coordinate_map.unique().to_numpy().flatten())
            #set labels to categories
            # axp.set_yticklabels(ylabs, rotation=ticklabelrotation)
            axp.set_yticklabels(coordinate.unique().sort().to_numpy().flatten(), rotation=ticklabelrotation)
        
        # print(ylabs)
        # print(coordinate.unique().to_numpy().flatten())
        
        #add spine labels (ylabs) on top  of each additional axis
        ax.text(
            x=(idx/n_coords), y=1.01,
            s=coordinate.name,
            transform=ax.transAxes,
            color=tickcolor,
            **text_kwargs
        )

        time.sleep(sleep)

        return axp

    def add_score_distribution(self,
        score_col_map:pl.Series,
        nanfrac:float=4/256,
        lab:str=None, ticklabcolor:Union[str,tuple]='tab:grey',
        cmap:Union[mcolors.Colormap,str]='plasma',
        fig:Figure=None, axpos:Union[tuple,int]=None,
        ) -> None:
        """
            - method to add a distribution of scores into the figure
            - the distribution will be colorcoded to matcht he colormapping of the different runs/models

            Parameters
            ----------
                - score_col_map
                    - pl.Series
                    - series containing some score mapped onto the interval [0,1]
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

        #adjust bins to colorbar
        bins = np.linspace(score_col_map.min().item(), score_col_map.max().item(), int(1//nanfrac))

        #get colors for bins
        if isinstance(cmap, str): cmap = plt.get_cmpa(cmap)
        colors = cmap(bins)
        
        #get histogram
        hist, bin_edges = np.histogram(score_col_map, bins)

        #plot and colormap hostogram
        ax.barh(bin_edges[:-1], hist, height=nanfrac, color=colors)
        ax.set_xscale('symlog')

        #labelling
        ax.set_ylabel(lab, rotation=270, labelpad=15, va='bottom', ha='center', color=ticklabcolor)
        ax.yaxis.set_label_position("right")
        
        ax.tick_params(axis='x', colors=ticklabcolor)
        ax.spines[['top', 'left', 'right']].set_visible(False)
        ax.spines['bottom'].set_color(ticklabcolor)
        ax.set_yticks([])
        ax.set_xlabel('Counts', color=ticklabcolor)
        

        return

    def __init_scorecol(self,
        df:pl.DataFrame,
        score_col:str=None, score_scaling:str='pl.col(score_col)',
        min_score:float=None, max_score:float=None,
        remove_nanscore:bool=False,
        verbose:int=0
        ) -> Tuple[pl.DataFrame,str]:
        """
            - method to inizialize the score column
            - will be called in self.plot()

            Parameters
            ----------
                - df
                    - pl.DataFrame
                    - input dataframe containing the score column/coordinate columns
                    - this dataframe will be modified
                - score_col
                    - str, optional
                    - name of the column to use for scoring
                    - the default is None
                        - will generate a placeholder column
                - score_scaling
                    - str, optional
                    - a polars expression
                    - scaling to apply to the (generated) score column
                    - the default is 'pl.col(score_col)'
                - min_score
                    - float, optional
                    - minimum score to plot
                    - everything below will be dropped from 'df'
                    - the default is None
                        - will be set to -np.inf
                - max_score
                    - float, optional
                    - maximum score to plot
                    - everything above will be dropped from 'df'
                    - the default is None
                        - will be set to np.inf
                - remove_nanscore
                    - bool, optional
                    - whether to drop all rows that have a score evaluating to np.nan
                    - the default is False
                - verbose
                    - int
                    - verbosity level
                    - the default is 0

            Raises
            ------

            Returns
            -------
                - df
                    - pl.DataFrame
                    - dataframe modified according to the specifications
                    - if score_col is None will contain one additional column
                        - this is a placeholder for score-col
                        - this column has only 0 as entries
                - score_col_use
                    - str
                    - modified version of the input 'score_col'

            Comments
            --------
        """
        
        #get initial shape of dataframe
        df_input_shape = df.shape[0]    #shape if input dataframe (for verbosity)


        #initialize a score_col placeholder if no score-col is provided
        if score_col is None:
            score_col_use = '<score_placeholder>'
            while score_col in df.columns:
                score_col_use += '_'
            df = df.insert_at_idx(df.shape[1], pl.Series(score_col_use, np.zeros(df.shape[0])))
        else:
            score_col_use = score_col
        ##replace 'score_col' with 'score_col_use'
        score_scaling = score_scaling.replace('score_col', 'score_col_use')

        #filter which range of scores to display and remove scores evaluating to nan if desired
        df = df.filter(((pl.col(score_col_use).is_between(min_score, max_score))|(pl.col(score_col_use).is_nan())))
        df_minmaxscore_shape = df.shape[0]  #shape of dataframe after score_col_use boundaries got applied
        
        if remove_nanscore:
            df = df.filter(pl.col(score_col_use).is_not_nan())
        df_nonan_shape = df.shape[0]    #shape of dataframe after nans got removed
    
        if verbose > 0:
            print(
                f'INFO(WB_HypsearchPlot): Removed\n'
                f'    {df_input_shape-df_minmaxscore_shape} row(s) via ({min_score} < {score_col_use} < {max_score}),\n'
                f'    {df_minmaxscore_shape-df_nonan_shape} row(s) containig nans,\n'
                f'    {df_input_shape-df_nonan_shape} row(s) total.\n'
            )
        #apply user defined expression to scale the score-function and thus color-scale
        df = df.with_columns(eval(score_scaling).alias(score_col_use))
        
        return df, score_col_use
    
    def __deal_withnan(self,
        df
        ) -> Tuple[pl.DataFrame,float]:
        """
            - method to deal with nan values in df
            - essentially will replace all nan with a value 0.5 lower than the minimum of the numerical columns in the dataframe
            - necessary to avoid issues while plotting

            Parameters
            ----------
                - df
                    - pl.DataFrame
                    - input dataframe that will be modified (nan will be filled)

            Raises
            ------

            Returns
            -------
                - df
                    - pl.DataFrame
                    - dataframe with the nan filled as 0.5 lower than the minimum of all numerical columns in the input 'df'
                - fill_value
                    - float
                    - the value used to fill the nan

            Comments
            --------
        """

        fill_value = df.min()
        cols = np.array(fill_value.columns)[(np.array(fill_value.dtypes, dtype=str)!='Utf8')]
        fill_value = np.nanmin(fill_value.select(pl.col(cols)).to_numpy())-0.5                  #fill with value slightly lower than minimum of whole dataframe -> Ensures that nan end up in respective part of colormap
        df = df.fill_nan(fill_value=fill_value)
        df = df.fill_null(value=fill_value)

        return df, fill_value

    def __deal_with_inf(self,
        df:pl.DataFrame, score_col:str,
        verbose:int=0,
        ):
        """
            - method that removes rows with infinite values in 'score_col' resulting from 'score_scaling'

            Parameters
            ----------
                -  df
                    - pl.DataFrame
                    - input dataframe that will be modified (inf and -inf will be replaced by nan)
                - score_col
                    - str
                    - name of the score column
                - verbose
                    - int, optional
                    - verbosity level
                    - the default is 0
            Raises
            ------
                
            Returns
            -------
                - df
                    - pl.DataFrame
                    - dataframe with the inf and -inf in score_col removed

            Comment
            -------
                
        """

        df_input_shape = df.shape[0]
        df = df.filter(pl.col(score_col).is_finite())

        df_inf_shape = df.shape[0]

        if verbose > 0:
            print(
                f'INFO(WB_HypsearchPlot): Removed\n'
                f'    {df_input_shape-df_inf_shape} row(s) where {score_col} evaluated to inf or -inf due to "score_scaling",\n'
                # f'    {df_input_shape-df_inf_shape} row(s) total.\n'
            )
        # df = df.with_columns(
        #     pl.when(pl.col(score_col).is_infinite())
        #     .then(float("nan"))
        #     .otherwise(pl.col(score_col))
        #     .keep_name()
        # )

        return df

    def plot(self,
        coordinates:Union[pl.DataFrame,List[dict]],
        id_col:str=None,
        score_col:str=None,
        coords_cols:Union[str,list]=r'^.*$',
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
        max_nretries:int=4,
        verbose:int=None,
        text_kwargs:dict=None, fig_kwargs:dict=None, save_kwargs:dict=None
        ) -> Tuple[Figure, plt.Axes]:
        """
            - method to create a coordinate plot

            Parameters
            ----------
                - coordinates
                    - pl.DataFrame, list(dict)
                    - structure contianing the coordinates to plot
                        - rows denote different runs/models
                        - columns different coordinates
                    - the structure should (but does not have to) contain
                        - an id column used for identification
                        - a score column used for rating different runs/models
                - id_col
                    - str, optional
                    - name of the column to use as an ID
                    - the default is None
                        - will generate an id column by assigning a unique integer to each row
                - score_col
                    - str, optional
                    - name of the column used for scoring the different runs/models
                    - the default is None
                        - no scoring will be used
                        - a dummy column consisting of zeros will be added to 'df'
                - coords_cols
                    - str, list
                    - expression or list specifying the columns in 'df' to use as coordinates
                    - the default is '^.*$'
                        - will select all columns
                - min_score
                    - float, optional
                    - minimum score to plot
                    - everything below will be dropped from 'df'
                    - the default is None
                        - will be set to -np.inf
                - max_score
                    - float, optional
                    - maximum score to plot
                    - everything above will be dropped from 'df'
                    - the default is None
                        - will be set to np.inf
                - remove_nanscore
                    - bool, optional
                    - whether to drop all rows that have a score evaluating to np.nan
                    - the default is False
                - score_scaling
                    - str, optional
                    - str representing a polars expression
                    - will be applied to 'score_col' to scale it
                        - might be useful for better visualization
                    - some useful examples
                        - 'np.log10(pl.col(score_col))'
                            - logarithmic score
                        - 'np.abs(pl.col(score_col))'
                            - absolute score
                        - 'pl.col(score_col)**2'
                            - squared score
                    - the default is 'pl.col(score_col)'
                        - i.e. no modification
                - show_idcol
                    - bool, optional
                    - whether to show a column encoding the ID of a particular run/model
                    - only recommended if the number of models is not too large
                    - overwrites self.show_idcol
                    - the default is None
                        - will default to self.show_idcol
                - interpkind
                    - str, optional
                    - function to use for the interpolation between the different coordinates
                    - argument passed as 'kind' to scipy.interpolate.interp1d()
                    - overwrites self.interpkind
                    - the default is None
                        - will default to self.interpkind
                - res
                    - int, optional
                    - resolution of the interpolated line
                        - i.e. the number of points used to plot each run/model
                    - overwrites self.res
                    - the default is None
                        - defaults to self.res
                - axpos_coord
                    - tuple, int, optional
                    - position of where to place the axis containing the coordinate-plot
                    - specification following the matplotlib convention
                        - will be passed to fig.add_subplot()
                    - overwrites self.axpos_coord
                    - the default is None
                        - defaults to self.axpos_coord
                - axpos_hist
                    - tuple, int, optional
                    - position of where to place the axis containing the histogram of scorings
                    - specification following the matplotlib convention
                        - will be passed to fig.add_subplot()
                    - overwrites self.axpos_hist
                    - the default is None
                        - defaults to self.axpos_hist
                - ticks2display
                    - int, optional
                    - number of ticks to show for numeric coordinates
                    - overwrites self.ticks2display
                    - the default is None
                        - defaults to self.ticks2display
                - tickcolor
                    - str, tuple, optional
                    - color to draw ticks and ticklabels in
                    - if a tuple is passed it has to be a RGBA-tuple
                    - overwrites self.tickcolor
                    - the default is None
                        - defaults to self.tickcolor
                - ticklabelrotation
                    - float, optional
                    - rotation of the ticklabels
                    - overwrites self.ticklabelrotation
                    - the default is None
                        - defaults to self.ticklabelrotation
                - tickformat
                    - str, optional
                    - formatstring for the (numeric) ticklabels
                    - overwrites self.tickformat
                    - the default is None
                        - defaults to self.tickformat
                - nancolor
                    - str, tuple, optional
                    - color to draw failed runs (evaluate to nan) in
                    - if a tuple is passed it has to be a RGBA-tuple
                    - overwrites self.nancolor
                    - the default is None
                        - defaults to self.nancolor
                - nanfrac
                    - float, optional
                    - the fraction of the colormap to use for nan-values (i.e. failed runs)
                        - fraction of 256 (resolution of the colormap)
                    - will also influence the number of bins/binsize used in the performance-histogram
                    - a value between 0 and 1
                    - overwrites self.nanfrac
                    - the default is None
                        - defaults to self.nanfrac
                - linealpha
                    - float, optional
                    - alpha value of the lines representing runs/models
                    - overwrites self.linealpha
                    - the default is None
                        - defaults to self.linealpha
                - linewidth
                    - float, optional
                    - linewidth of the lines representing runs/models
                    - overwrites self.linewidth
                    - the default is None
                        - defaults to self.linewidth
                - base_cmap
                    - str, mcolors.Colormap
                    - colormap to map the scores onto
                    - some space will be allocated for nan, if nans shall be displayed as well
                    - overwrites self.base_cmap
                    - the default is None
                        - defaults to self.base_cmap
                - cbar_over_hist
                    - bool, optional
                    - whether to plot a colorbar instead of the default performance-histogram
                    - the histogram also has a colorbar encoded
                    - overwrites self.cbar_over_hist
                    - the default is None
                        - defaults to self.cbar_over_hist
                - n_jobs
                    - int, optional
                    - number of threads to use when plotting the runs/modes
                    - use for large coordinate-plots
                    - argmument passed to joblib.Parallel
                    - overwrites self.n_jobs
                    - the default is None
                        - defaults to self.n_jobs
                - n_jobs_addaxes
                    - int, optional
                    - number of threads to use when plotting the axes for the different coordinates
                    - use if a lot coordinates shall be plotted
                    - argmument passed to joblib.Parallel
                    - it could be that that a RuntimeError occurs if too many threads try to add axes at the same time
                        - this error should be caught with a try-except statement, but in case it something still goes wrong try setting 'n_jobs_addaxes' to 1
                    - overwrites self.n_jobs_addaxes
                    - the default is None
                        - defaults to self.n_jobs_addaxes
                - sleep
                    - float, optional
                    - time to sleep after finishing each job in plotting runs/models and coordinate-axes
                    - overwrites self.sleep
                    - the default is None
                        - defaults to self.sleep
                - map_suffix
                    - str, optional
                    - suffix to add to the columns created by mapping the coordinates onto the intervals [0,1]
                    - only use if your column-names contain the default ('__map')
                    - overwrites self.map_suffix
                    - the default is None
                        - defaults to self.map_suffix
                - save
                    - str, optional
                    - filename to save the plot to
                    - the default is False
                        - will not save the plot
                - max_nretries
                    - int, optional
                    - maximum number of retries in case a RuntimeError occurs during parallelized plotting
                    - only relevant if 'n_jobs' or 'n_jobs_addaxes' is greater than 1 or -1
                    - the default is 4
                - verbose
                    - int, optional
                    - verbosity level
                    - overwrites self.verbose
                    - the default is None
                        - defaults to self.verbose
                - text_kwargs
                    - dict, optional
                    - kwargs passed to ax.text()
                        - affect the labels of the created subplots for each coordinate
                    - overwrites self.text_kwargs
                    - the default is None
                        - defaults to self.text_kwargs
                - fig_kwargs
                    - dict, optional
                    - kwargs to pass to plt.figure()
                    - the default is None
                        - will not pass any additional arguments
                - save_kwargs
                    - dict, optional
                    - kwargs to pass to plt.savefig()
                    - the default is None
                        - will not pass any additional arguments

            Raises
            ------

            Returns
            -------
                - fig
                    - Figure
                    - created figure
                - axs
                    - plt.Axes
                    - axes corresponding to 'fig'

            Comments
            --------
        """

        #initialize
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
        if text_kwargs is None:         text_kwargs         = self.text_kwargs

        if min_score is None: min_score = -np.inf
        if max_score is None: max_score =  np.inf
        

        if fig_kwargs is None:  fig_kwargs  = {}
        if save_kwargs is None: save_kwargs = {}


        #convert grid to polars DataFrame
        if isinstance(coordinates, pl.DataFrame):
            df = coordinates
        else:
            df = pl.DataFrame(coordinates)

        
        #initialize idcol correctly (generate id if None has been passed)
        if id_col is None:
            id_col = 'id'
            df = df.insert_at_idx(0, pl.Series(id_col, np.arange(0, df.shape[0], 1, dtype=np.int64)))
        
        #initialize score column
        df, score_col_use = self.__init_scorecol(
            df=df,
            score_col=score_col, score_scaling=score_scaling,
            min_score=min_score, max_score=max_score,
            remove_nanscore=remove_nanscore,
            verbose=verbose
        )
        
        #get colormap
        if isinstance(base_cmap, str): cmap = plt.get_cmap(base_cmap)
        else: cmap = base_cmap
        ##if any score evaluates to nan, generate a cmap accordingly
        if df.select(pl.col(score_col_use).is_nan().any()).item():
            cmap = self.make_new_cmap(cmap=cmap, nancolor=nancolor, nanfrac=nanfrac)

        #extract model-names, parameter-columns, ...
        ids = df.select(pl.col(id_col))
        scores = df.select(pl.col(score_col_use))
        df = df.drop(id_col)                 #in case idcol is part of the searched parameters
        df = df.drop(score_col_use)          #in case score_col_use is part of the coords_cols
        df = df.select(pl.col(coords_cols))
        
        #if desired also show the individual model-ids
        if show_idcol: df.insert_at_idx(0,  ids.to_series())
        #only display score, if score_col is provided
        if score_col is not None: df.insert_at_idx(df.shape[1], scores.to_series())
                
        #deal with missing and inifinite values
        df, fill_value = self.__deal_withnan(df)
        df = self.__deal_with_inf(df, score_col, verbose=verbose)


        #coordinates
        coords = df.columns
        coords_map = [c+map_suffix for c in df.columns]
        n_coords = len(coords)-1

        #convert categorical columns to unique integers for plotting - ensures that each coordinate lies within range(0,1)
        df = df.fill_null('None') #replace None with 'None' because otherwise mapping does not work        
        for c in df.columns:
            df = self.col2range01(df=df, colname=c, map_suffix=map_suffix)
            
        #################
        #actual plotting#
        #################

        #temporarily disable tight-layout if enabled by default
        plt.rcParams['figure.autolayout'] = False


        fig = plt.figure(**fig_kwargs)
        
        #axis used for plotting models
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
                        f'INFO(WB_HypsearchPlot.plot_model()):\n'
                        f'    The following error occured while plotting the models: {err}.\n'
                        f'    Retrying to plot. Number of elapsed retries: {nretries}.'
                    )

        #plot one additional y-axis for every single coordinate
        ##try except to retry if a RuntimeError occured
        e = True
        nretries = 0
        while e and nretries < max_nretries:
            try:
                axps = Parallel(n_jobs=n_jobs_addaxes, verbose=verbose, prefer='threads')(
                    delayed(self.add_coordaxes)(
                        ax=ax1,
                        coordinate=coordinate, coordinate_map=coordinate_map,
                        fill_value=fill_value,
                        idx=idx, n_coords=n_coords,
                        ticks2display=ticks2display, tickcolor=tickcolor, ticklabelrotation=ticklabelrotation, tickformat=tickformat,
                        sleep=sleep,
                        text_kwargs=text_kwargs,
                    ) for idx, (coordinate, coordinate_map) in enumerate(zip(df.select(pl.col(coords)), df.select(pl.col(coords_map))))
                )
                e = False
            except RuntimeError as err:
                e = True
                nretries += 1
                if verbose > 0:
                    print(
                        f'INFO(WB_HypsearchPlot.add_coordaxes()):\n'
                        f'    The following error occured while plotting the models: {err}.\n'
                        f'    Retrying to plot. Number of elapsed retries: {nretries}.'
                    )
        
        #generate score label from score_scaling expression
        score_lab = re.sub(r'pl\.col\(score\_col\)', score_col_use, score_scaling)
        score_lab = re.sub(r'(np|pl|pd)\.', '', score_lab)
        
        #if a score column has been passed use it
        if score_col is not None:
            #add colorbar
            if cbar_over_hist:
                norm = mcolors.Normalize(vmin=df.select(pl.col(score_col_use)).min(), vmax=df.select(pl.col(score_col)).max())
                cbar = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=norm), ax=axps[-1], pad=0.0005, drawedges=False, anchor=(0,0))
                cbar.outline.set_visible(False) #hide colorbar outline
                cbar.set_label(score_lab, color=tickcolor)


                cbar.ax.set_zorder(0)
                cbar.ax.set_yticks([])  #use ticks from last subplot

            #generate a histogram including the colorbar
            else:
                self.add_score_distribution(
                    score_col_map=df.select(pl.col(score_col_use+map_suffix)),
                    nanfrac=nanfrac,
                    lab=score_lab, ticklabcolor=tickcolor,
                    cmap=cmap,
                    fig=fig, axpos=axpos_hist,
                )
                plt.subplots_adjust(wspace=0) #will move around colorbar if applied there
        
        if isinstance(save, str): plt.savefig(save, bbox_inches='tight', **save_kwargs)
        # plt.tight_layout()    #moves cbar around

        axs = fig.axes

        plt.rcParams['figure.autolayout'] = True


        return fig, axs
    
class LatentSpaceExplorer:
    """
        - class to plot generated samples from latent variables

        Attributes
        ----------
            - plot_func
                - str, optional
                - function to use for visualizing each individual sample
                - has to be the string equivalent of any method that can be called on plt.Axes
                - the corresponding function has to take at least one argument (the values to plot)
                - the default is 'plot'
            - subplots_kwargs
                - dict, optional
                - kwargs to pass to plt.subplots()
                - the default is None
                    - will be initialized with {}
            - predict_kwargs
                - dict, optional
                - kwargs to pass to generator.predict()
                    - generator is a parameter passed self.plot()
                - the default is None
                    - will be initialized with {}
            - plot_func_kwargs
                - dict, optional
                - kwargs to pass to the function passed to 'plot_func'
                - the default is None
                    - will be initialized with {}
            - verbose
                - int, optional
                - verbosity level
                - the default is 0

        Methods
        -------
            - plot()

        Dependencies
        ------------
            - matplotlib
            - numpy
            - typing

        Comments
        --------
            - a figure shall be created before calling this method
            - if the plot is not showing you might have to call plt.show() after the execution of this method

    """

    def __init__(self,
        plot_func:str='plot',
        subplots_kwargs:dict=None, predict_kwargs:dict=None, plot_func_kwargs:dict=None,
        verbose:int=0
        ) -> None:

        self.plot_func = plot_func

        if subplots_kwargs is None:     self.subplots_kwargs     = {}
        else:                           self.subplots_kwargs     = subplots_kwargs
        if predict_kwargs is None:      self.predict_kwargs      = {}
        else:                           self.predict_kwargs      = predict_kwargs
        if plot_func_kwargs is None:    self.plot_func_kwargs    = {}
        else:                           self.plot_func_kwargs    = plot_func_kwargs

        self.verbose = verbose

        return
    
    def __repr__(self) -> str:
        return (
            f'PlotLatentExamples(\n'
            f'    plot_func={self.plot_func},\n'
            f'    subplots_kwargs={self.subplots_kwargs}, predict_kwargs={self.predict_kwargs}, plot_func_kwargs={self.plot_func_kwargs},\n'
            f'    verbose={self.verbose},\n'
            f')'
        )

    def plot_dbe(self,
        X:np.ndarray, y:np.ndarray,
        res:int=100, k:int=1,
        ax:plt.Axes=None,
        contourf_kwargs:dict=None,
        ) -> None:
        """
            - method to plot estimated desicion-boundaries of data
            - uses voronoi diagrams to to do so
                - estimates the decision boundaries using KNN with k=1
                - Source: https://stackoverflow.com/questions/37718347/plotting-decision-boundary-for-high-dimension-data
                    - last access: 17.05.2023
            
            Parameters
            ----------
                - X
                    - np.ndarray
                    - 2d array containing the features of the data
                        - i.e. 2 features
                - y
                    - np.ndarray
                    - 1d array of shape X.shape[0]
                    - labels corresponding to X
                - res
                    - int, optional
                    - resolution of the estimated boundary
                    - the default is 100
                - k
                    - int, optional
                    - number of neighbours to use in the KNN estimator
                    - the default is 1
                - ax
                    - plt.Axes
                    - axes to add the density estimate to
                    - the default is None
                        - will call plt.contourf() instead of ax.contourf()
                - contourf_kwargs
                    - dict, optional
                    - kwargs to pass to .contourf() function
                    - the default is None
                        - will be initialized with {'alpha':0.5, 'zorder':-1}


            Raises
            ------
                - ValueError
                    - if either 'X' or 'y' are not passed in coorect shapes

            Returns
            -------

            Dependencies
            ------------
                - matplotlib
                - numpy
                - sklearn

            Comments
            --------

        """
        
        #initialize parameters
        if contourf_kwargs is None: contourf_kwargs = {'alpha':0.5, 'zorder':-1}

        y = y.flatten()

        #check shapes
        if X.shape[1] != 2:
            raise ValueError(f'"X" has to contain 2 features. I.e. it has to be of shape (n_samples,2) and not {X.shape}')
        if y.shape[0] != X.shape[0]:
            raise ValueError(f'"y" has to be a 1d version of "X" containing the labels corresponding to the samples. I.e. it has to be of shape (n_samples,) and not {y.shape}')

        #get background model
        background_model = KNeighborsClassifier(n_neighbors=k)
        background_model.fit(X, y)
        xx, yy = np.meshgrid(
            np.linspace(X[:,0].min(), X[:,0].max(), res),
            np.linspace(X[:,1].min(), X[:,1].max(), res),
        )

        voronoiBackground = background_model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape((res, res))

        #(add) plot
        if ax is not None:
            ax.contourf(xx, yy, voronoiBackground, **contourf_kwargs)
        else:
            plt.contourf(xx, yy, voronoiBackground, **contourf_kwargs)

        return

    def corner_plot(self,
        X:np.ndarray, y:np.ndarray=None,
        ) -> Tuple[Figure,plt.Axes]:
        from astroLuSt.ai.mle import MLE

        if y is None: y = 'tab:blue'
        fig, axs = MLE().corner_plot(
            X=X, y=y, featurenames=[f'z[{idx}]' for idx in range(X.shape[1])]
        )

        return

    def generated_1d(self,
        generator:Callable,
        z0:list,
        zi_f:Union[list,int],
        z0_idx:int=0,
        plot_func:str=None,
        subplots_kwargs:dict=None, predict_kwargs:dict=None, plot_func_kwargs:dict=None,
        verbose:int=None,
        ) -> Tuple[Figure, List[plt.Axes]]:
        """
            - method to actually generate a plot showing samples out of the latent space while varying 2 latent variables

            Parameters
            ----------
                - generator
                    - Callable class
                    - has to implement a predict method
                    - will be called to generate samples from the provided latent variables
                        - the latent variables are a list of the length len(zi_f)+1
                            - i.e. the zi_f fixed variables + the iterated z0 latent variable
                - z0
                    - list
                    - values of one of the latent dimensions interpretable by generator
                    - will be iterated over and used to generate a grid
                    - has to be equally spaced
                - zi_f
                    - list, int
                    - fixed variables of all latent dimensions which are not z0
                    - if a list gets passed
                        - has to contain as many entries as the list that get interpreted by generator.predict - 1
                        - each entry represents one latent dimensions value in order
                    - if an integer gets passed
                        - has to be of same value as the length of the list that get interpreted by generator.predict - 1
                        - will be initialized with a list of zeros length zi_f
                - z0_idx
                    - int, optional
                    - index of where in the list of latent dimensions z0 is located
                    - has to differ from z1_idx
                    - the deafult is 0
                        - will set z0 as the first element of the latent vector 
                - subplots_kwargs
                    - dict, optional
                    - kwargs to pass to plt.subplots()
                    - overwrites self.subplot_kwargs
                    - the default is None
                        - will default to self.subplot_kwargs
                - predict_kwargs
                    - dict, optional
                    - kwargs to pass to generator.predict()
                        - generator is a parameter passed self.plot()
                    - overwrites self.subplot_kwargs
                    - the default is None
                        - will default to self.subplot_kwargs
                - plot_func_kwargs
                    - dict, optional
                    - kwargs to pass to the function passed to 'plot_func'
                    - overwrites self.plot_func_kwargs
                    - the default is None
                        - will default to self.plot_func_kwargs
                - verbose
                    - int, optional
                    - verbosity level
                    - overwrites self.verbose
                    - the default is None
                        - will default to self.verbose
                    
            Raises
            ------
                - ValueError
                    - if 'z0' is not equally spaced

            Returns
            -------
                - fig
                    - Figure
                    - created figure
                - axs
                    - list(plt.Axes)
                    - axes corresponding to fig
                    - contains an axis to set labels and title as last entry

            Comments
            --------
                - if you want to show something in the background simply follow this structure:
                
                >>> PLE = PlotLatentExamples(...)
                >>> fig, axs = PLE.plot_1d(...)
                >>> axs[-1].plot(...)
                >>> plt.show()
                
                - you can also use function that take plt.Axes as parameters. The following example plots some decision boundary estmiate in the background of the latent space examples:
                
                >>> PLE = PlotLatentExamples(...)
                >>> fig, axs = PLE.plot_1d(...)
                >>> plot_dbe(..., ax=axs[-1])
                >>> plt.show()            
        """
        #check shapes
        if len(np.unique(np.diff(z0))) > 1:
            raise ValueError(f'"z0" has to be equally spaced!')
        
        #initialize properly
        z0         = np.array(z0)
        if isinstance(zi_f, int):
            zi_f   = np.zeros(zi_f)
        else:
            zi_f       = zi_f
        if plot_func is None:           plot_func           = self.plot_func
        if subplots_kwargs is None:     subplots_kwargs     = self.subplots_kwargs
        if predict_kwargs is None:      predict_kwargs      = self.predict_kwargs
        if plot_func_kwargs is None:    plot_func_kwargs    = self.plot_func_kwargs
        if verbose is None:             verbose             = self.verbose
        
        #get indices of fixed zi
        zi_f_idxs = [i for i in range(len(zi_f)+1) if i != z0_idx]



        #temporarily disable autolayout
        plt.rcParams['figure.autolayout'] = False


        fig, axs = plt.subplots(nrows=1, ncols=z0.shape[0], **subplots_kwargs)
        

        tit = ', '.join([f'z{fzi_idx}: {fzi}' for fzi_idx, fzi in zip(zi_f_idxs, zi_f)])

        #subplot for axis labels
        ax0 = fig.add_subplot(111, zorder=-1)
        ax0.set_title(f'Latent Space\n({tit})')
        ax0.set_xlabel(f'z[{z0_idx}]')
        ax0.set_yticks([])
        #set ticks and ticklabels correctly
        ax0.set_xlim(np.min(z0)-np.mean(np.diff(z0))/2, np.max(z0)+np.mean(np.diff(z0))/2)
        ax0.xaxis.set_major_locator(MaxNLocator(len(z0), prune='both'))        


        for col, z0i in enumerate(z0):

            #construct latent-vector - fixed_zi never change, variable_zi get iterated over
            z = np.zeros((np.shape(zi_f)[0]+1))
            z[z0_idx] = z0i
            z[zi_f_idxs] = zi_f
            z_sample = np.array([z])

            try:
                x_decoded = generator.predict(z_sample, **predict_kwargs)
            except Exception as e:
                raise ValueError(f'"generator" has to implement a "predict()" method that takes a list of len(zi_f)+1 parameters as input!')


            if x_decoded.shape[0] == 1: x_decoded = x_decoded.flatten()


            eval(f'axs[col].{plot_func}(x_decoded, **plot_func_kwargs)')

            #hide labels of latent samples
            # axs[col].set_title(z0i)
            axs[col].set_xlabel('')
            axs[col].set_xticks([])
            axs[col].set_yticks([])
            
            #set subplot background transparent (to show content of ax0)
            axs[col].patch.set_alpha(0.0)

        #get axes
        axs = fig.axes

        plt.subplots_adjust(wspace=0, hspace=0)
        
        plt.rcParams['figure.autolayout'] = True

        return fig, axs

    def generated_2d(self,
        generator:Callable,
        z0:list, z1:list,
        zi_f:Union[list,int],
        z0_idx:int=0, z1_idx:int=1,
        plot_func:str=None,
        subplots_kwargs:dict=None, predict_kwargs:dict=None, plot_func_kwargs:dict=None,
        verbose:int=None,
        ) -> Tuple[Figure, List[plt.Axes]]:
        """
            - method to actually generate a plot showing samples out of the latent space while varying 2 latent variables

            Parameters
            ----------
                - generator
                    - Callable class
                    - has to implement a predict method
                    - will be called to generate samples from the provided latent variables
                        - the latent variables are a list of the length len(zi_f)+2 
                            - i.e. the zi_f fixed variables + the iterated z0 and z1 variables 
                - z0
                    - list
                    - values of one of the latent dimensions interpretable by generator
                    - will be iterated over and used to generate a grid
                    - has to be equally spaced
                - z1
                    - list
                    - values of the second latent dimension interpretable by generator
                    - will be iterated over and used to generate a grid
                    - has to be equally spaced
                - zi_f
                    - list, int
                    - fixed variables of all latent dimensions which are not 'z0' and 'z1'
                    - if a list gets passed
                        - has to contain as many entries as the list that get interpreted by generator.predict - 2
                        - each entry represents one latent dimensions value in order
                    - if an integer gets passed
                        - has to be of same value as the length of the list that get interpreted by generator.predict - 2
                        - will be initialized with a list of zeros length zi_f
                - z0_idx
                    - int, optional
                    - index of where in the list of latent dimensions z0 is located
                    - has to differ from z1_idx
                    - the deafult is 0
                        - will set z0 as the first element of the latent vector 
                - z1_idx
                    - int, optional
                    - index of where in the list of latent dimensions z1 is located
                    - has to differ from z0_idx
                    - the deafult is 1
                        - will set z1 as the second element of the latent vector 
                    - int, optional
                - subplots_kwargs
                    - dict, optional
                    - kwargs to pass to plt.subplots()
                    - overwrites self.subplot_kwargs
                    - the default is None
                        - will default to self.subplot_kwargs
                - predict_kwargs
                    - dict, optional
                    - kwargs to pass to generator.predict()
                        - generator is a parameter passed self.plot()
                    - overwrites self.subplot_kwargs
                    - the default is None
                        - will default to self.subplot_kwargs
                - plot_func_kwargs
                    - dict, optional
                    - kwargs to pass to the function passed to 'plot_func'
                    - overwrites self.plot_func_kwargs
                    - the default is None
                        - will default to self.plot_func_kwargs
                - verbose
                    - int, optional
                    - verbosity level
                    - overwrites self.verbose
                    - the default is None
                        - will default to self.verbose
                    
            Raises
            ------
                - ValueError
                    - if 'z0' or 'z1' is not equally spaced
                    - if 'z0_idx' and 'z1_idx' have the same value

            Returns
            -------
                - fig
                    - Figure
                    - created figure
                - axs
                    - list(plt.Axes)
                    - axes corresponding to fig
                    - contains an axis to set labels and title as last entry

            Comments
            --------
                - if you want to show something in the background simply follow this structure:
                
                >>> PLE = PlotLatentExamples(...)
                >>> fig, axs = PLE.plot_2d(...)
                >>> axs[-1].plot(...)
                >>> plt.rcParams['figure.autolayout'] = False #to keep no whitespace between generated samples
                >>> plt.show()
                
                - you can also use function that take plt.Axes as parameters. The following example plots some decision boundary estmiate in the background of the latent space examples:
                
                >>> PLE = PlotLatentExamples(...)
                >>> fig, axs = PLE.plot_2d(...)
                >>> plot_dbe(..., ax=axs[-1])
                >>> plt.rcParams['figure.autolayout'] = False #to keep no whitespace between generated samples
                >>> plt.show()
        """

        #check shapes
        if np.any((np.diff(np.diff(z0))) > 1e-8) or np.any(np.diff((np.diff(z1))) > 1e-8):
            warnings.warn(f'"z0" and "z1" have to be equally spaced!')
        if z0_idx == z1_idx:
            raise ValueError(
                f'"z0_idx" and "z1_idx" have to have different values!\n'
                f'    If you want to only vary one latent dimension use plot_1d()!'
            )

            
        #initialize properly
        z0         = np.array(z0)
        z1         = np.array(z1)
        if isinstance(zi_f, int):
            zi_f   = np.zeros(zi_f)
        else:
            zi_f       = zi_f
        if plot_func is None:           plot_func           = self.plot_func
        if subplots_kwargs is None:     subplots_kwargs     = self.subplots_kwargs
        if predict_kwargs is None:      predict_kwargs      = self.predict_kwargs
        if plot_func_kwargs is None:    plot_func_kwargs    = self.plot_func_kwargs
        if verbose is None:             verbose             = self.verbose
        

        #get indices of fixed zi
        zi_f_idxs = [i for i in range(len(zi_f)+2) if i not in [z0_idx, z1_idx]]



        #temporarily disable autolayout
        plt.rcParams['figure.autolayout'] = False


        fig, axs = plt.subplots(nrows=z1.shape[0], ncols=z0.shape[0], **subplots_kwargs)
        

        tit = ', '.join([f'z{fzi_idx}: {fzi}' for fzi_idx, fzi in zip(zi_f_idxs, zi_f)])

        z0_step = np.max(z0)-np.min(z0)
        z1_step = np.max(z1)-np.min(z1)

        #subplot for axis labels
        ax0 = fig.add_subplot(111, zorder=-1)
        ax0.set_title(f'Latent Space\n({tit})')
        ax0.set_xlabel(f'z[{z0_idx}]')
        ax0.set_ylabel(f'z[{z1_idx}]')

        #set ticks and ticklabels correctly
        ax0.set_ylim(np.min(z1)-np.mean(np.diff(z1))/2, np.max(z1)+np.mean(np.diff(z1))/2)
        ax0.set_xlim(np.min(z0)-np.mean(np.diff(z0))/2, np.max(z0)+np.mean(np.diff(z0))/2)
        ax0.yaxis.set_major_locator(MaxNLocator(len(z1), prune='both'))
        ax0.xaxis.set_major_locator(MaxNLocator(len(z0), prune='both'))

        for row, z1i in enumerate(z1[::-1]):
            for col, z0i in enumerate(z0):

                #construct latent-vector - fixed_zi never change, variable_zi get iterated over
                z = np.zeros((np.shape(zi_f)[0]+2))
                z[z0_idx] = z0i
                z[z1_idx] = z1i
                z[zi_f_idxs] = zi_f



                z_sample = np.array([z])

                try:
                    x_decoded = generator.predict(z_sample, **predict_kwargs)
                except Exception as e:
                    raise ValueError(f'"generator" has to implement a "predict()" method that takes a list of len(zi_f)+2 parameters as input!')


                if x_decoded.shape[0] == 1: x_decoded = x_decoded.flatten()


                eval(f'axs[row, col].{plot_func}(x_decoded, **plot_func_kwargs)')

                #hide labels of latent samples
                # axs[row, col].set_title(f'{z0i},{z1i}')
                axs[row, col].set_xlabel('')
                axs[row, col].set_ylabel('')
                axs[row, col].set_xticks([])
                axs[row, col].set_yticks([])

                #set subplot background transparent (to show content of ax0)
                axs[row, col].patch.set_alpha(0.0)

        #get axes
        axs = fig.axes

        plt.subplots_adjust(wspace=0, hspace=0)
        
        plt.rcParams['figure.autolayout'] = True

        return fig, axs


#%%functions
def plot_dbe(
    X:np.ndarray, y:np.ndarray,
    res:int=100, k:int=1,
    ax:plt.Axes=None,
    contourf_kwargs:dict=None,
    ) -> None:        
    """
        - function to plot estimated desicion-boundaries of data
        - uses voronoi diagrams to to do so
            - estimates the decision boundaries using KNN with k=1
            - Source: https://stackoverflow.com/questions/37718347/plotting-decision-boundary-for-high-dimension-data
                - last access: 17.05.2023
        
        Parameters
        ----------
            - X
                - np.ndarray
                - 2d array containing the features of the data
                    - i.e. 2 features
            - y
                - np.ndarray
                - 1d array of shape X.shape[0]
                - labels corresponding to X
            - res
                - int, optional
                - resolution of the estimated boundary
                - the default is 100
            - k
                - int, optional
                - number of neighbours to use in the KNN estimator
                - the default is 1
            - ax
                - plt.Axes
                - axes to add the density estimate to
                - the default is None
                    - will call plt.contourf() instead of ax.contourf()
            - contourf_kwargs
                - dict, optional
                - kwargs to pass to .contourf() function
                - the default is None
                    - will be initialized with {'alpha':0.5, 'zorder':-1}


        Raises
        ------
            - ValueError
                - if either 'X' or 'y' are not passed in coorect shapes

        Returns
        -------

        Dependencies
        ------------
            - matplotlib
            - numpy
            - sklearn

        Comments
        --------

    """
    
    #initialize parameters
    if contourf_kwargs is None: contourf_kwargs = {'alpha':0.5, 'zorder':-1}

    y = y.flatten()

    #check shapes
    if X.shape[1] != 2:
        raise ValueError(f'"X" has to contain 2 features. I.e. it has to be of shape (n_samples,2) and not {X.shape}')
    if y.shape[0] != X.shape[0]:
        raise ValueError(f'"y" has to be a 1d version of "X" containing the labels corresponding to the samples. I.e. it has to be of shape (n_samples,) and not {y.shape}')

    #get background model
    background_model = KNeighborsClassifier(n_neighbors=k)
    background_model.fit(X, y)
    xx, yy = np.meshgrid(
        np.linspace(X[:,0].min(), X[:,0].max(), res),
        np.linspace(X[:,1].min(), X[:,1].max(), res),
    )

    voronoiBackground = background_model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape((res, res))

    #(add) plot
    if ax is not None:
        ax.contourf(xx, yy, voronoiBackground, **contourf_kwargs)
    else:
        plt.contourf(xx, yy, voronoiBackground, **contourf_kwargs)

    return

def corner_plot(
    X:np.ndarray, y:Union[np.ndarray,str]=None, featurenames:np.ndarray=None,
    mus:np.ndarray=None, sigmas:np.ndarray=None, corrmat:np.ndarray=None,
    bins:int=100,
    equal_range:bool=False, asstandardnormal:bool=False,
    fig:Figure=None,
    sctr_kwargs:dict=None
    ) -> Tuple[Figure,plt.Axes]:
    """
        - function to create a corner plot of a given set of input data X
    
        Parameters
        ----------
            - X
                - np.ndarray
                - contains samples as rows
                - contains features as columns
            - y
                - np.ndarray, str, optional
                - contains labels corresponding to X
                - if a np.ndarray is passed
                    - will be used as the colormap
                - if a string is passed
                    - will be interpreted as the actual color
                - the default is None
                    - will default to 'tab:blue'
            - featurenames
                - np.ndarray, optional
                - names to give to the features present in X
                - the default is None
                    - will initialize with 'Feature i', where i is the index the feature appears in X
            - mus
                - np.ndarray, optional
                - contains the mean value estimates corresponding to X
                - the default is None
                    - will be ignored
            - sigmas
                - np.ndarray, optional
                - contains the standard deviation estimates corresponding to X
                - the default is None
                    - will be ignored
            - corrmat
                - np.ndarray, optional
                - correlation matrix for X
                - has to have shape (X.shape[1],X.shape[1])
                - the default is None
                    - will infer the correlation coefficients
            - bins
                - int, optional
                - number of bins to use in plt.histogram
                - also the resolution for the estimation of the normal distribution
                - the default is 100
            - equal_range
                - bool, optional
                - whether to plot the data with equal x- and y-limits
                - the default is False
            - asstandardnormal
                - bool, optional
                - whether to plot the data rescaled to zero mean and unit variance
                - the default is False
            - fig
                - Figure, optional
                - figure to plot into
                - the default is None
                    - will create a new figure

        Raises
        ------

        Returns
        -------
            - fig
                - Figure
                - the created matplotlib figure
            - axs
                - plt.Axes
                - axes corresponding to 'fig'

        Dependencies
        ------------
            - scipy
            - matplotlib
            - numpy
            - typing

        Comments
        --------

    """


    #initialize correctly
    if y is None:
        y = 'tab:blue'
    if featurenames is None: featurenames = [f"Feature {i}" for i in np.arange(X.shape[1])]

    if mus is None:
        mus = [None]*len(X)
    if sigmas is None:
        sigmas = [None]*len(X)
    if corrmat is None:
        corrmat = np.corrcoef(X)
    if sctr_kwargs is None:
        sctr_kwargs = {'s':1, 'alpha':0.5, 'zorder':2}


    #plot
    if fig is None: fig = plt.figure()
    nrowscols = X.shape[1]


    idx = 0
    for idx1, (d1, l1, mu1, sigma1) in enumerate(zip(X.T, featurenames, mus, sigmas)):
        for idx2, (d2, l2, mu2, sigma2) in enumerate(zip(X.T, featurenames, mus, sigmas)):
            idx += 1

            if asstandardnormal and mu1 is not None and sigma1 is not None:
                d1 = (d1-mu1)/sigma1
                d2 = (d2-mu2)/sigma2
                mu1, mu2 = 0, 0
                sigma1, sigma2 = 1, 1
            
            #plotting 2D distributions
            if idx1 > idx2:

                ax1 = fig.add_subplot(nrowscols, nrowscols, idx)
                
                if mu1 is not None: ax1.axhline(mu1, color="tab:orange", linestyle="--")
                if mu2 is not None: ax1.axvline(mu2, color="tab:orange", linestyle="--")

                #data
                sctr = ax1.scatter(
                    d2, d1,
                    c=y,               
                    **sctr_kwargs,
                )


                
                #normal distribution estimate
                if mu1 is not None and sigma1 is not None:
                    
                    covmat = np.cov(np.array([d1,d2]))

                    xvals = np.linspace(np.nanmin([d1, d2]), np.nanmax([d1, d2]), bins)
                    yvals = np.linspace(np.nanmin([d1, d2]), np.nanmax([d1, d2]), bins)
                    xx, yy = np.meshgrid(xvals, yvals)
                    mesh = np.dstack((xx, yy))
                    
                    norm = stats.multivariate_normal(
                        mean=np.array([mu1, mu2]),
                        cov=covmat,
                        allow_singular=True
                    )
                    cont = ax1.contour(yy, xx, norm.pdf(mesh), cmap="gray", zorder=1)
                

                #labelling
                if idx1 == nrowscols-1:
                    ax1.set_xlabel(l2)
                else:
                    ax1.set_xticklabels([])
                if idx2 == 0:
                    ax1.set_ylabel(l1)
                else:
                    ax1.set_yticklabels([])
                ax1.tick_params()

                if not equal_range:
                    ax1.set_xlim(np.nanmin(d2), np.nanmax(d2))
                    ax1.set_ylim(np.nanmin(d1), np.nanmax(d1))

                #add corrcoeff in legend
                ax1.errorbar(np.nan, np.nan, color="none", label=r"$r_\mathrm{P}=%.4f$"%(corrmat[idx1, idx2]))
                ax1.legend()

            #plotting 1d histograms
            elif idx1 == idx2:
                if idx != 1:
                    orientation = "horizontal"
                else:
                    orientation = "vertical"


                axhist = fig.add_subplot(nrowscols, nrowscols, idx)

                axhist.hist(d1, bins=bins, orientation=orientation, density=True)

                #normal distribution estimate
                if mu1 is not None and sigma1 is not None:
                    xvals = np.linspace(np.nanmin(d1), np.nanmax(d1), bins)
                    normal = stats.norm.pdf(xvals, mu1, sigma1)
                    
                    if orientation == "horizontal":
                        axhist.axhline(mu1, color="tab:orange", linestyle="--", label=r"$\mu=%.2f$"%(mu1))
                        axhist.plot(normal, xvals)
                        if not equal_range:
                            axhist.set_ylim(np.nanmin(d1), np.nanmax(d1))
                        
                        axhist.xaxis.set_ticks_position("top")
                        axhist.set_yticklabels([])
                    
                    elif orientation == "vertical":
                        axhist.plot(xvals, normal)
                        axhist.axvline(mu1, color="tab:orange", linestyle="--", label=r"$\mu=%.2f$"%(mu1))
                        if not equal_range:
                            axhist.set_xlim(np.nanmin(d1), np.nanmax(d1))
                        
                        axhist.yaxis.set_ticks_position("right")
                        axhist.set_xticklabels([])
                
                    axhist.errorbar(np.nan, np.nan, color="none", label=r"$\sigma=%.2f$"%(sigma1))
                    axhist.legend()
                

                axhist.tick_params()
    
    #make x and y limits equal if requested
    if equal_range and not asstandardnormal:
        for idx, ax in enumerate(fig.axes):
            
            #first 1D histogram
            if idx == 0:
                xymin = np.nanmin([fig.axes[idx+1].get_xlim(), fig.axes[idx+1].get_ylim()])
                xymax = np.nanmax([fig.axes[idx+1].get_xlim(), fig.axes[idx+1].get_ylim()])
                ax.set_xlim(xymin, xymax)

            #all other 1D histograms
            elif idx > 0 and (idx+1)%nrowscols == 0:
                xymin = np.nanmin([fig.axes[idx-1].get_xlim(), fig.axes[idx-1].get_ylim()])
                xymax = np.nanmax([fig.axes[idx-1].get_xlim(), fig.axes[idx-1].get_ylim()])
                ax.set_ylim(xymin, xymax)

            #2D histograms
            else:
                xymin = np.nanmin([fig.axes[idx].get_xlim(), fig.axes[idx].get_ylim()])
                xymax = np.nanmax([fig.axes[idx].get_xlim(), fig.axes[idx].get_ylim()])

                ax.set_xlim(xymin, xymax)
                ax.set_ylim(xymin, xymax)
    
    axs = fig.axes

    return fig, axs