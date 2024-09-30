

#%%imports
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import matplotlib.colors as mcolors
import numpy as np
import polars as pl
from typing import Tuple, Union, List

from astroLuSt.visualization.plotting import generate_colors


#%%definitions
class GANTT:
    """
    - class to create a Workload- and GANTT-Chart given some taballaric project data

    Attributes
    ----------
        - `start_col`
            - str, int, optional
            - the column containing timestamps of when the tasks started
            - if int
                - will be interpreted as index of the column
            - otherwise
                - will be interpreted as column name
            - the default is `None`
                - will be infered by using `end_col` and `dur_col` once `self.plot_gantt()` or `self.plot_workload()` are called
                - therefore at least two `start_col`, `end_col`, `dur_col` have to be not `None`
        - `end_col`
            - str, int, optional
            - the column containing timestamps of when the tasks started
            - if int
                - will be interpreted as index of the column
            - otherwise
                - will be interpreted as column name
            - the default is `None`
                - will be infered by using `start_col` and `dur_col` once `self.plot_gantt()` or `self.plot_workload()` are called
                - therefore at least two `start_col`, `end_col`, `dur_col` have to be not `None`
        - `dur_col`
            - str, int, optional
            - the column containing timestamps of when the tasks started
            - if int
                - will be interpreted as index of the column
            - otherwise
                - will be interpreted as column name
            - the default is `None`
                - will be infered by using `start_col` and `end_col` once `self.plot_gantt()` or `self.plot_workload()` are called
                - therefore at least two `start_col`, `end_col`, `dur_col` have to be not `None`
        - `comp_col`
            - str, int, optional
            - the column containing values describing how much of a task is completed
            - if int
                - will be interpreted as index of the column
            - otherwise
                - will be interpreted as column name
            - the default is `None`
                - will be set to 1 (100%) for all entries in `df` once `self.plot_gantt()` or `self.plot_workload()` are called
        - `start_slope_col`
            - str, int, optional
            - the column containing values describing how steep the incline at the starting side of each task is for the corresponding workload-curve
            - if int
                - will be interpreted as index of the column
            - otherwise
                - will be interpreted as column name
            - the default is `None`
                - will be set to 1 for all entries in `df`
        - `end_slope_col`
            - str, int, optional
            - the column containing values describing how steep the incline at the ending side of each task is for the corresponding workload-curve
            - if int
                - will be interpreted as index of the column
            - otherwise
                - will be interpreted as column name
            - the default is `None`
                - will be set to 1 for all entries in `df`
        - `weight_col`
            - str, int, optional
            - the column containing values describing how much each task contributes to the overall workload
            - if int
                - will be interpreted as index of the column
            - otherwise
                - will be interpreted as column name
            - the default is `None`
                - will be set to 1 for all entries in `df`
        - `color_by`
            - str, int, optional
            - the column by which to color the bars for each task
            - if int
                - will be interpreted as index of the column
            - otherwise
                - will be interpreted as column name
            - the default is 0
        - `sort_by`
            - str, int, optional
            - the column by which to sort the bars for each task
                - i.e. the y-axis of the chart
            - if int
                - will be interpreted as index of the column
            - otherwise
                - will be interpreted as column name
            - the default is 0
        - `cmap`
            - str, mcolors.Colormap, optional
            - colormap to use for coloring bars in the GANTT-chart and curves in the workload-chart
            - colormap applied according to `color_by`
            - the default is `nipy_spectral`
        - `total_color`
            - list, optional
            - has to have length of 4
                - RGBA-tuple
            - color to use for the total workload
            - the default is `None`
                - will be set to `[0,0,0,1]` (black)
        - `res`
            - int, optional
            - resolution of the workload-curves
                - i.e. for how many datapoints to calculate the workload
            - the default is 100
        - `time_scaling`
            - float, optional
            - factor to scale integers converted to datetime objects
            - will affect the steepness of the individual workload-curves 
            - the default is 1E-12
        - `verbose`
            - int, optional
            - verbosity level
            - the default is 0
        - `plot_kwargs`
            - dict, optional
            - kwargs to be passed to `.plot()`
            - the default is `None`
                - will be set to `{}`
        - `fill_between_kwargs`
            - dict, optional
            - kwargs to be passed to `.fill_between()`
            - the default is `None`
                - will be set to `{'alpha':0.5}`                
        - `axvline_kwargs`
            - dict, optional
            - kwargs to be passed to `.axvline()`
            - the default is `None`
                - will be set to `{'color':'k', 'linestyle':'--'}`
        - `text_kwargs`
            - dict, optional
            - kwargs to be passed to `.text()`
            - the default is `None`
                - will be set to `{'ha':'left', 'va':'bottom', 'y':0}`
        - `grid_kwargs`
            - dict, optional
            - kwargs to be passed to `ax.grid()`
            - the default is `None`
                - will be set to `{'visible':True, 'axis':'x'}`    

        Methods
        -------
            - `__get_missing()`
            - `__get_cmap()`
            - `sigmoid()`
            - `workload_curve()`
            - `plot_gantt()`
            - `plot_workload()`
            - `plot()`
        
        Dependencies
        ------------
            - datetime
            - matplotlib
            - numpy
            - polars
            - typing

        Comments
        --------
    
    """
    
    def  __init__(self,
        start_col:Union[str,int]=None, end_col:Union[str,int]=None, dur_col:Union[str,int]=None, comp_col:Union[str,int]=None,
        start_slope_col:Union[str,int]=None, end_slope_col:Union[str,int]=None,
        weight_col:Union[str,int]=None,
        color_by:Union[str,int]=0, sort_by:Union[str,int]=0,                  
        cmap:Union[str,mcolors.Colormap]=None, total_color=None,
        res:int=100,
        time_scaling:float=1E-12,
        plot_kwargs:dict=None, fill_between_kwargs:dict=None, axvline_kwargs:dict=None, text_kwargs:dict=None, grid_kwargs:dict=None,
        verbose:int=0,
        ) -> None:

        self.start_col          = start_col
        self.end_col            = end_col
        self.dur_col            = dur_col
        self.comp_col           = comp_col
        self.start_slope_col    = start_slope_col
        self.end_slope_col      = end_slope_col
        self.weight_col         = weight_col
        self.color_by           = color_by
        self.sort_by            = sort_by

        if cmap is None:                self.cmap                   = 'nipy_spectral'
        else:                           self.cmap                   = cmap
        if total_color is None:         self.total_color            = [0,0,0,1]
        else:                           self.total_color            = total_color
        if plot_kwargs is None:         self.plot_kwargs            = {}
        else:                           self.plot_kwargs            = plot_kwargs 
        if fill_between_kwargs is None: self.fill_between_kwargs    = {'alpha':0.5}
        else:                           self.fill_between_kwargs    = fill_between_kwargs 
        if axvline_kwargs is None:      self.axvline_kwargs         = {'color':'k', 'linestyle':'--'}
        else:                           self.axvline_kwargs         = axvline_kwargs 
        if text_kwargs is None:         self.text_kwargs            = {'ha':'left', 'va':'bottom', 'y':0}
        else:                           self.text_kwargs            = text_kwargs 
        if grid_kwargs is None:         self.grid_kwargs            = {'visible':True, 'axis':'x'}
        else:                           self.grid_kwargs            = grid_kwargs 
        self.res = res
        self.time_scaling = time_scaling
        self.verbose = verbose

        return
    
    def __repr__(self) -> str:
        
        return (
            f'GANTT(\n'
            f'    start_col={repr(self.start_col)}, end_col={repr(self.end_col)}, dur_col={repr(self.dur_col)}, comp_col={repr(self.comp_col)},\n'
            f'    start_slope_col={repr(self.start_slope_col)}, end_slope_col={repr(self.end_slope_col)},\n'
            f'    weight_col={repr(self.weight_col)},\n'
            f'    color_by={repr(self.color_by)}, sort_by={repr(self.sort_by)},\n'
            f'    cmap={repr(self.cmap)}, total_color={repr(self.total_color)},\n'
            f'    res={repr(self.res)},\n'
            f'    time_scaling={repr(self.time_scaling)},\n'
            f'    verbose={repr(self.verbose)},\n'
            f'    plot_kwargs={repr(self.plot_kwargs)}, fill_between_kwargs={repr(self.fill_between_kwargs)}, axvline_kwargs={repr(self.axvline_kwargs)}, text_kwargs={repr(self.text_kwargs)}, grid_kwargs={repr(self.grid_kwargs)},\n'
            f')'
        )
    
    def __get_missing(self,
        df:pl.DataFrame,
        start_col:Union[str,int]=None, end_col:Union[str,int]=None, dur_col:Union[str,int]=None, comp_col:Union[str,int]=None,
        start_slope_col:Union[str,int]=None, end_slope_col:Union[str,int]=None,
        weight_col:Union[str,int]=None,
        verbose:int=None,
        ) -> Tuple[pl.DataFrame,pl.Series,pl.Series,pl.Series,pl.Series,pl.Series,pl.Series,str,str,str,str,str,str]:
        """
            - private method to infer missing columns in `df`

            Parameters
            ----------
                - `df`
                    - pl.DataFrame
                    - dataframe containing all tasks to plot in the GANTT chart
                - `start_col`
                    - str, int, optional
                    - the column containing timestamps of when the tasks started
                    - if int
                        - will be interpreted as index of the column
                    - otherwise
                        - will be interpreted as column name
                    - the default is `None`
                        - will be infered by using `end_col` and `dur_col`
                        - therefore at least two `start_col`, `end_col`, `dur_col` have to be not `None`
                - `end_col`
                    - str, int, optional
                    - the column containing timestamps of when the tasks started
                    - if int
                        - will be interpreted as index of the column
                    - otherwise
                        - will be interpreted as column name
                    - the default is `None`
                        - will be infered by using `start_col` and `dur_col`
                        - therefore at least two `start_col`, `end_col`, `dur_col` have to be not `None`
                - `dur_col`
                    - str, int, optional
                    - the column containing timestamps of when the tasks started
                    - if int
                        - will be interpreted as index of the column
                    - otherwise
                        - will be interpreted as column name
                    - the default is `None`
                        - will be infered by using `start_col` and `end_col`
                        - therefore at least two `start_col`, `end_col`, `dur_col` have to be not `None`
                - `comp_col`
                    - str, int, optional
                    - the column containing values describing how much of a task is completed
                    - if int
                        - will be interpreted as index of the column
                    - otherwise
                        - will be interpreted as column name
                    - the default is `None`
                        - will be set to 1 (100%) for all entries in `df`
                - `start_slope_col`
                    - str, int, optional
                    - the column containing values describing how steep the incline at the starting side of each task is
                    - if int
                        - will be interpreted as index of the column
                    - otherwise
                        - will be interpreted as column name
                    - the default is `None`
                        - will be set to 1 for all entries in `df`
                - `end_slope_col`
                    - str, int, optional
                    - the column containing values describing how steep the incline at the ending side of each task is
                    - if int
                        - will be interpreted as index of the column
                    - otherwise
                        - will be interpreted as column name
                    - the default is `None`
                        - will be set to 1 for all entries in `df`
                - `weight_col`
                    - str, int, optional
                    - the column containing values describing how much each task contributes to the overall workload
                    - if int
                        - will be interpreted as index of the column
                    - otherwise
                        - will be interpreted as column name
                    - the default is `None`
                        - will be set to 1 for all entries in `df`
                - `verbose`
                    - int, optional
                    - verbosity level
                    - will override `self.verbose`
                    - the default is `None`
                        - will fall back to `self.verbose`

            Raises
            ------
                - `ValueError`
                    - if more than two of `start_col`, `end_col`, `dur_col` are `None`

            Returns
            -------
                - `df`
                    - pl.DataFrame
                    - input dataframe with infered columns added
                - `start`
                    - pl.Series
                    - infered starting timestamps if `None` provided
                    - otherwise the passed starting timestamps
                - `end`
                    - pl.Series
                    - infered ending timestamps if `None` provided
                    - otherwise the passed ending timestamps
                - `duration`
                    - pl.Series
                    - infered durations if `None` provided
                    - otherwise the passed starting durations
                - `completed`
                    - pl.Series
                    - if `None` was provided to `comp_col`
                        - series of ones
                    - otherwise the passed completion fractions
                - `start_slope`
                    - pl.Series
                    - if `None` was provided
                        - series of ones
                    - otherwise the passed start slopes
                - `end_slope`
                    - pl.Series
                    - if `None` was provided
                        - series of ones
                    - otherwise the passed end slopes
                - `weight`
                    - pl.Series
                    - if `None` was provided
                        - series of ones
                    - otherwise the passed weights
                - `col_start`
                    - str
                    - updated column name in `df` of `col_start`
                - `col_end`
                    - str
                    - updated column name in `df` of `col_end`
                - `col_dur`
                    - str
                    - updated column name in `df` of `col_dur`
                - `col_comp`
                    - str
                    - updated column name in `df` of `col_comp`
                - `col_start_slope`
                    - str
                    - updated column name in `df` of `col_start_slope`
                - `col_end_slope`
                    - str
                    - updated column name in `df` of `col_end_slope`
                - `col_weight`
                    - str
                    - updated column name in `df` of `col_weight`


            Comments
            --------      
        """

        if verbose is None: verbose = self.verbose

        #initialize non-essential columns accordingly
        if comp_col is None:
            comp_col = '<completion placeholder>'
            df = df.with_columns(pl.lit(1).alias(comp_col))
        if start_slope_col is None:
            start_slope_col = '<start_slope placeholder>'
            df = df.with_columns(pl.lit(1).alias(start_slope_col))
        if end_slope_col is None:
            end_slope_col = '<end_slope placeholder>'
            df = df.with_columns(pl.lit(1).alias(end_slope_col))
        if weight_col is None:
            weight_col = '<weight placeholder>'
            df = df.with_columns(pl.lit(1).alias(weight_col))

        start_slope = df[start_slope_col]
        end_slope   = df[end_slope_col]
        weight      = df[weight_col]


        if isinstance(start_col, int):          col_start       = df.columns[start_col]
        else:                                   col_start       = start_col
        if isinstance(end_col, int):            col_end         = df.columns[end_col]
        else:                                   col_end         = end_col
        if isinstance(dur_col, int):            col_dur         = df.columns[dur_col]
        else:                                   col_dur         = dur_col
        if isinstance(comp_col, int):           col_comp        = df.columns[comp_col]
        else:                                   col_comp        = comp_col
        if isinstance(start_slope_col, int):    col_start_slope = df.columns[start_slope_col]
        else:                                   col_start_slope = start_slope_col
        if isinstance(end_slope_col, int):      col_end_slope   = df.columns[end_slope_col]
        else:                                   col_end_slope   = end_slope_col
        if isinstance(weight_col, int):         col_weight      = df.columns[weight_col]
        else:                                   col_weight      = weight_col


        #determine how much of the task has been finished already and duration of whole task
        ##duration missing
        if col_start is not None and col_end is not None:
            completed = (((df[col_end]-df[col_start])).cast(int) * df[col_comp]).cast(pl.Datetime)
            duration  = (df[col_end]-df[col_start]).cast(pl.Datetime)
            start = df[col_start]
            end = df[col_end]
            if verbose > 1:
                print(
                    f'INFO(GANTT.__get_missing):\n'
                    f'    Infering `duration` from `start` and `end`'
                )
        ##end missing
        elif col_start is not None and col_end is None and col_dur is not None:
            completed = (df[col_dur].cast(int) * df[col_comp]).cast(pl.Time)
            duration  = df[col_dur].cast(pl.Time)
            start = df[col_start]
            end = df[col_start] + df[col_dur]
            if verbose > 1:
                print(
                    f'INFO(GANTT.__get_missing):\n'
                    f'    Infering `end` from `start` and `duration`'
                )
        ##start missing
        elif col_start is None and col_end is not None and col_dur is not None:
            completed = (df[col_dur].cast(int) * df[col_comp]).cast(pl.Time)
            duration  = df[col_dur].cast(pl.Time)
            start = df[col_end] - df[col_dur]
            end = df[col_end]
            if verbose > 1:
                print(
                    f'INFO(GANTT.__get_missing):\n'
                    f'    Infering `start` from `end` and `duration`'
                )
        else:
            raise ValueError('At least two of `col_start`, `col_end`, `col_dur` have to be not `None`!')

        #rename series to have correct names
        start.rename(       col_start,          in_place=True)
        end.rename(         col_end,            in_place=True)
        duration.rename(    col_dur,            in_place=True)
        completed.rename(   col_comp,           in_place=True)
        start_slope.rename( col_start_slope,    in_place=True)
        end_slope.rename(   col_end_slope,      in_place=True)
        weight.rename(      col_weight,         in_place=True)

        return (
            df,
            start, end, duration, completed, start_slope, end_slope, weight,
            col_start, col_end, col_dur, col_comp, col_start_slope, col_end_slope, col_weight
        )

    def __generate_cmap(self,
        df:pl.DataFrame,
        col_cmap:str, cmap:Union[str,mcolors.Colormap],
        ) -> np.ndarray:
        """
            - private method to generate a list of colors for unique elements in `df[col_map]`

            Parameters
            ----------
                - `df`
                    - pl.DataFrame
                    - dataframe containing `col_cmap` as column
                - `col_cmap`
                    - str
                    - column name to be considered for the coloring
                    - will assign one color to each unique element in `df[col_cmap]`
                - `cmap`
                    - str, mcolors.Colormap
                    - colormap to use for generating the colors

            Raises
            ------

            Returns
            -------
                - `colors`
                    - np.ndarray
                    - generated colors corresponding to unique elements in `df[col_cmap]`
                        - each unique element will have a unique color
                        - the order of `colors` is the same as the one of `df[col_cmap]`

            Comments
            --------
        """

        nuniquecolors = df[col_cmap].n_unique()
        ncolors = df[col_cmap].shape[0]

        gen_colors = generate_colors(
            classes=nuniquecolors+2,
            cmap=cmap
        )[1:-1]

        #init output
        colors = np.zeros((ncolors,4))
        uniques, indices = np.unique(df[col_cmap], return_index=True)
        
        #replace output with corresponding colors
        for u, idx in zip(uniques, indices):
            colors[(df[col_cmap].to_numpy()==u)] += gen_colors[idx]
        
        return colors

    def sigmoid(self,
        x:np.ndarray,
        slope:np.ndarray=1, shift:np.ndarray=0
        ) -> np.ndarray:
        """
            - method to calculate a generalized sigmoid function

            Parameters
            ----------
                - `x`
                    - np.ndarray
                    - independent variable to calculate simoid of
                - `slope`
                    - np.ndarray, optional
                    - parameter changing the slope of the sigmoid
                    - the default is 1
                - `shift`
                    - np.ndarray, optional
                    - parameter to shift the sigmoid in direction of `x`
                    - the default is 0

            Raises
            ------

            Returns
            -------
                - `sigma`
                    - np.ndarray
                    - generalized sigmoid evaluated on `x`

            Comments
            --------

        """

        Q1 = 1 + np.e**(slope*(-(x - shift)))
        sigma = 1/Q1

        return sigma
  
    def workload_curve(self,
        x:pl.Series,
        start:pl.Series, end:pl.Series, duration:pl.Series, completed:pl.Series,
        start_slope:pl.Series, end_slope:pl.Series,
        weight:pl.Series,
        time_scaling:float=None,
        ) -> Tuple[np.ndarray,np.ndarray]:
        """
            - method to calculate a specific tasks workload curve (workload over time)
            - combines two sigmoids with opposite slopes
        
            Parameters
            ----------
                - `x`
                    - np.ndarray
                    - the x-values to be evaluated
                        - usually times
                - `start`
                    - pl.Series
                    - time at which the tasks start
                - `end`
                    - pl.Series
                    - time at which the tasks end
                - `duration`
                    - pl.Series
                    - duration of each task
                - `completed`
                    - pl.Series
                    - factor describing how much of each task is completed
                    - not needed in function
                - `start_slope`
                    - pl.Series
                    - how steep the starting phase should be for each task
                - `end_slope`
                    - pl.Series
                    - how steep the ending phase (reflection phase) should be for each task
                - `weight`
                    - pl.Series
                    - weights describing how much each task contributes to the overall workload
                - `time_scaling`
                    - float, optional
                    - factor to scale integers converted to datetime objects
                    - will affect the steepness of the individual curves
                    - the default is `None`
                        - will fall back to `self.time_scaling`

            Raises
            ------

            Returns
            -------
                - `workload`
                    - np.ndarray
                    - workload curve for each task
                - `total_workload`
                    - np.ndarray
                    - workload curve for the weighted sum of all tasks in dependence of time

            Comments
            --------
        """

        if time_scaling is None: time_scaling = self.time_scaling

        #get workload-curves
        ##no offset by start, because start is the zeropoint
        time_eval = np.array(len(start)*[x.cast(int)])
        l_start = self.sigmoid(time_eval.T, slope= start_slope.cast(float).to_numpy()*time_scaling, shift=(start.cast(int).to_numpy()))
        l_end   = self.sigmoid(time_eval.T, slope=-end_slope.cast(float).to_numpy()  *time_scaling, shift=(start.cast(int).to_numpy()+duration.cast(int).to_numpy()))
        ##weight curves
        workload = (l_start+l_end)
        workload -= workload.min()
        workload /= np.nanmax(workload, axis=0)
        workload *= weight.to_numpy()

        #total workload
        total_workload = np.sum(workload, axis=1)
        
        #normalize workload
        workload /= np.nanmax(total_workload)
        total_workload /= np.nanmax(total_workload)
        
        return workload, total_workload


    def plot_gantt(self,
        df:pl.DataFrame,
        ax:plt.Axes=None,
        start_col:Union[str,int]=None, end_col:Union[str,int]=None, dur_col:Union[str,int]=None, comp_col:Union[str,int]=None,
        color_by:Union[str,int]=None, sort_by:Union[str,int]=None,
        cmap:Union[str,mcolors.Colormap]=None,
        verbose:int=None,
        axvline_kwargs:dict=None, text_kwargs:dict=None, grid_kwargs:dict=None,
        ) -> None:
        """
            - method to create a GANTT chart given information stored in a pl.DataFrame

            Parameters
            ----------
                - `df`
                    - pl.DataFrame
                    - dataframe containing all tasks to plot in the GANTT chart
                - `ax`
                    - plt.Axes, optional
                    - axes to plot the graph onto
                    - if `None`
                        - will call `plt.barh()` ect. instead of `ax.barh()`
                    - the default is `None`
                - `start_col`
                    - str, int, optional
                    - the column containing timestamps of when the tasks started
                    - if int
                        - will be interpreted as index of the column
                    - otherwise
                        - will be interpreted as column name
                    - overrides `self.start_col`
                    - the default is `None`
                        - will fall back to `self.start_col`
                        - if that is also `None`
                            - will be infered by using `end_col` and `dur_col`
                            - therefore at least two `start_col`, `end_col`, `dur_col` have to be not `None`
                - `end_col`
                    - str, int, optional
                    - the column containing timestamps of when the tasks started
                    - if int
                        - will be interpreted as index of the column
                    - otherwise
                        - will be interpreted as column name
                    - overrides `self.end_col`
                    - the default is `None`
                        - will fall back to `self.end_col`
                        - if that is also `None`
                            - will be infered by using `start_col` and `dur_col`
                            - therefore at least two `start_col`, `end_col`, `dur_col` have to be not `None`
                - `dur_col`
                    - str, int, optional
                    - the column containing timestamps of when the tasks started
                    - if int
                        - will be interpreted as index of the column
                    - otherwise
                        - will be interpreted as column name
                    - overrides `self.dur_col`
                    - the default is `None`
                        - will fall back to `self.dur_col`
                        - if that is also `None`
                            - will be infered by using `start_col` and `end_col`
                            - therefore at least two `start_col`, `end_col`, `dur_col` have to be not `None`
                - `comp_col`
                    - str, int, optional
                    - the column containing values describing how much of a task is completed
                    - if int
                        - will be interpreted as index of the column
                    - otherwise
                        - will be interpreted as column name
                    - overrides `self.comp_col`
                    - the default is `None`
                        - will fall back to `self.comp_col`
                        - if that is also `None`
                            - will be set to 1 (100%) for all entries in `df`
                - `color_by`
                    - str, int, optional
                    - the column by which to color the bars for each task
                    - if int
                        - will be interpreted as index of the column
                    - otherwise
                        - will be interpreted as column name
                    - overrides `self.color_by`
                    - the default is `None`
                        - will fall back to `self.color_by`
                - `sort_by`
                    - str, int, optional
                    - the column by which to sort the bars for each task
                        - i.e. the y-axis of the chart
                    - if int
                        - will be interpreted as index of the column
                    - otherwise
                        - will be interpreted as column name
                    - overrides `self.sort_by`
                    - the default is `None`
                        - will fall back to `self.sort_by`
                - `cmap`
                    - str, mcolor.Colormap, optional
                    - colormap to use for coloring bars by `color_by`
                    - overrides `self.cmap`
                    - the default is `None`
                        - will fall back to `self.cmap`
                - `verbose`
                    - int, optional
                    - verbosity level
                    - will override `self.verbose`
                    - the default is `None`
                        - will fall back to `self.verbose`
                - `axvline_kwargs`
                    - dict, optional
                    - kwargs to be passed to `.axvline()`
                    - overrides `self.axvline_kwargs`
                    - the default is `None`
                        - will fall back to `self.axvline_kwargs`
                - `text_kwargs`
                    - dict, optional
                    - kwargs to be passed to `.text()`
                    - overrides `self.text_kwargs`
                    - the default is `None`
                        - will fall back to `self.text_kwargs`
                - `grid_kwargs`
                    - dict, optional
                    - kwargs to be passed to `ax.grid()`
                    - overrides `self.grid_kwargs`
                    - the default is `None`
                        - will fall back to `self.grid_kwargs`

            Raises
            ------

            Returns
            -------

            Comments
            --------
        """

        #default values
        if start_col is None:       start_col       = self.start_col
        if end_col is None:         end_col         = self.end_col
        if dur_col is None:         dur_col         = self.dur_col
        if comp_col is None:        comp_col        = self.comp_col
        if color_by is None:        color_by        = self.color_by
        if sort_by is None:         sort_by         = self.sort_by
        if cmap is None:            cmap            = self.cmap
        if verbose is None:         verbose         = self.verbose
        if axvline_kwargs is None:  axvline_kwargs  = self.axvline_kwargs
        if text_kwargs is None:     text_kwargs     = self.text_kwargs
        if grid_kwargs is None:     grid_kwargs     = self.grid_kwargs

        if isinstance(color_by, int):   col_cmap  = df.columns[color_by]
        else:                           col_cmap  = color_by
        if isinstance(sort_by, int):    col_sort  = df.columns[sort_by]
        else:                           col_sort  = sort_by

        #generate colormap for plot
        colors = generate_colors(
            classes=df[col_cmap].n_unique()+2,
            cmap=cmap
        )[1:-1]

        df, \
        start, end, duration, completed, start_slope, end_slope, weight,\
        col_start, col_end, col_dur, col_comp, col_start_slope, col_end_slope, \
        col_weight \
            = self.__get_missing(
                df=df, start_col=start_col, end_col=end_col, dur_col=dur_col, comp_col=comp_col,
                start_slope_col=None, end_slope_col=None,
                weight_col=None,
                verbose=verbose,        
        )

        #plot onto axis
        if ax is not None:
            #add bar for each task
            bars_d = ax.barh(y=df[col_sort], width=duration,  left=start, color=colors, alpha=0.5, zorder=0)
            bars_c = ax.barh(y=df[col_sort], width=completed, left=start, color=colors, alpha=1.0, zorder=1)
            
            #add labels to bars
            ax.bar_label(bars_d, labels=[f'{p:g}%' for p in df[col_comp]*100])


            #vertical line for today
            today = np.datetime64(datetime.now())
            ax.axvline(today, **axvline_kwargs)
            ax.text(x=today, s=' Today', **text_kwargs)

            #correct y_axis for sorting
            ax.invert_yaxis()
            ax.set_xticks(ax.get_xticks())
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

            #add gridlines
            ax.grid(**grid_kwargs)

            #labelling
            ax.set_xlabel('Time')
            ax.set_ylabel(col_sort)


        #plot in current figure
        else:
            #add bar for each task
            plt.barh(y=df[col_sort], width=duration,  left=start, color=colors, alpha=0.5, zorder=0)
            plt.barh(y=df[col_sort], width=completed, left=start, color=colors, alpha=1.0, zorder=1)
            
            #vertical line for today
            plt.axvline(np.datetime64(datetime.now()), **axvline_kwargs)

        return 
    
    def plot_workload(self,
        df:pl.DataFrame,
        ax:plt.Axes=None,
        start_col:Union[str,int]=None, end_col:Union[str,int]=None, dur_col:Union[str,int]=None, comp_col:Union[str,int]=None,
        start_slope_col:Union[str,int]=None, end_slope_col:Union[str,int]=None,
        weight_col:Union[str,int]=None,
        color_by:Union[str,int]=None, sort_by:Union[str,int]=None,
        cmap:Union[str,mcolors.Colormap]=None, total_color:list=None,
        res:int=None,
        time_scaling:float=None,
        verbose:int=None,
        plot_kwargs:dict=None, fill_between_kwargs:dict=None, axvline_kwargs:dict=None, text_kwargs:dict=None, grid_kwargs:dict=None
        ) -> None:
        """
            - method to generate a workload plot of the tasks provided in `df`
            - each task will be represented by two generalized sigmoids
                - one for the starting phase
                - other for the ending phase
            
            Parameters
            ----------
                - `df`
                    - pl.DataFrame
                    - dataframe containing all tasks to plot in the workload chart
                - `ax`
                    - plt.Axes, optional
                    - axes to plot the graph onto
                    - if `None`
                        - will call `plt.barh()` ect. instead of `ax.barh()`
                    - the default is `None`
                - `start_col`
                    - str, int, optional
                    - the column containing timestamps of when the tasks started
                    - if int
                        - will be interpreted as index of the column
                    - otherwise
                        - will be interpreted as column name
                    - overrides `self.start_col`
                    - the default is `None`
                        - will fall back to `self.start_col`
                        - if that is also `None`
                            - will be infered by using `end_col` and `dur_col`
                            - therefore at least two `start_col`, `end_col`, `dur_col` have to be not `None`
                - `end_col`
                    - str, int, optional
                    - the column containing timestamps of when the tasks started
                    - if int
                        - will be interpreted as index of the column
                    - otherwise
                        - will be interpreted as column name
                    - overrides `self.end_col`
                    - the default is `None`
                        - will fall back to `self.end_col`
                        - if that is also `None`
                            - will be infered by using `start_col` and `dur_col`
                            - therefore at least two `start_col`, `end_col`, `dur_col` have to be not `None`
                - `dur_col`
                    - str, int, optional
                    - the column containing timestamps of when the tasks started
                    - if int
                        - will be interpreted as index of the column
                    - otherwise
                        - will be interpreted as column name
                    - overrides `self.dur_col`
                    - the default is `None`
                        - will fall back to `self.dur_col`
                        - if that is also `None`
                            - will be infered by using `start_col` and `end_col`
                            - therefore at least two `start_col`, `end_col`, `dur_col` have to be not `None`
                - `comp_col`
                    - str, int, optional
                    - the column containing values describing how much of a task is completed
                    - if int
                        - will be interpreted as index of the column
                    - otherwise
                        - will be interpreted as column name
                    - overrides `self.comp_col`
                    - the default is `None`
                        - will fall back to `self.comp_col`
                        - if that is also `None`
                            - will be set to 1 (100%) for all entries in `df`
                - `start_slope_col`
                    - str, int, optional
                    - the column containing values describing how steep the incline at the starting side of each task is
                    - if int
                        - will be interpreted as index of the column
                    - otherwise
                        - will be interpreted as column name
                    - overrides `self.start_slope_col`
                    - the default is `None`
                        - will fall back to `self.start_slope_col`
                        - if that is also `None`
                            - will be set to 1 for all entries in `df`
                - `end_slope_col`
                    - str, int, optional
                    - the column containing values describing how steep the incline at the ending side of each task is
                    - if int
                        - will be interpreted as index of the column
                    - otherwise
                        - will be interpreted as column name
                    - overrides `self.end_slope_col`
                    - the default is `None`
                        - will fall back to `self.end_slope_col`
                        - if that is also `None`
                            - will be set to 1 for all entries in `df`
                - `weight_col`
                    - str, int, optional
                    - the column containing values describing how much each task contributes to the overall workload
                    - if int
                        - will be interpreted as index of the column
                    - otherwise
                        - will be interpreted as column name
                    - overrides `self.weight_col`
                    - the default is `None`
                        - will fall back to `self.weight_col`
                        - if that is also `None`
                            - will be set to 1 for all entries in `df`
                - `color_by`
                    - str, int, optional
                    - the column by which to color the bars for each task
                    - if int
                        - will be interpreted as index of the column
                    - otherwise
                        - will be interpreted as column name
                    - overrides `self.color_by`
                    - the default is `None`
                        - will fall back to `self.color_by`
                - `sort_by`
                    - str, int, optional
                    - not used in this method
                    - the column by which to sort the bars for each task in `self.plot_gantt()`
                        - i.e. the y-axis of the chart
                    - if int
                        - will be interpreted as index of the column
                    - otherwise
                        - will be interpreted as column name
                    - overrides `self.sort_by`
                    - the default is `None`
                        - will fall back to `self.sort_by`
                - `cmap`
                    - str, mcolors.Colormap, optional
                    - colormap to use for coloring workload-curves by `color_by`
                    - overrides `self.cmap`
                    - the default is `None`
                        - will fall back to `self.cmap`
                - `total_color`
                    - list, optional
                    - has to have length of 4
                        - RGBA-tuple
                    - color to use for the total workload
                    - overrides `self.total_color`
                    - the default is `None`
                        - will fall back to `self.total_color`
                - `res`
                    - int, optional
                    - resolution of the workload-curves
                        - i.e. for how many datapoints to calculate the workload
                    - overrides `self.res`
                    - the default is `None`
                        - will fall back to `self.res`
                - `time_scaling`
                    - float, optional
                    - factor to scale integers converted to datetime objects
                    - will affect the steepness of the individual curves
                    - the default is `None`
                        - will fall back to `self.time_scaling`                        
                - `verbose`
                    - int, optional
                    - verbosity level
                    - will override `self.verbose`
                    - the default is `None`
                        - will fall back to `self.verbose`
                - `plot_kwargs`
                    - dict, optional
                    - kwargs to be passed to `.plot()`
                    - overrides `self.plot_kwargs`
                    - the default is `None`
                        - will fall back to `self.plot_kwargs`
                - `fill_between_kwargs`
                    - dict, optional
                    - kwargs to be passed to `.fill_between()`
                    - overrides `self.fill_between_kwargs`
                    - the default is `None`
                        - will fall back to `self.fill_between_kwargs`
                - `axvline_kwargs`
                    - dict, optional
                    - kwargs to be passed to `.axvline()`
                    - overrides `self.axvline_kwargs`
                    - the default is `None`
                        - will fall back to `self.axvline_kwargs`
                - `text_kwargs`
                    - dict, optional
                    - kwargs to be passed to `.text()`
                    - overrides `self.text_kwargs`
                    - the default is `None`
                        - will fall back to `self.text_kwargs`
                - `grid_kwargs`
                    - dict, optional
                    - kwargs to be passed to `ax.grid()`
                    - overrides `self.grid_kwargs`
                    - the default is `None`
                        - will fall back to `self.grid_kwargs`

            Raises
            ------

            Returns
            -------

            Comments
            --------
        """

        #default values
        if start_col is None:           start_col           = self.start_col
        if end_col is None:             end_col             = self.end_col
        if dur_col is None:             dur_col             = self.dur_col
        if comp_col is None:            comp_col            = self.comp_col
        if start_slope_col is None:     start_slope_col     = self.start_slope_col
        if end_slope_col is None:       end_slope_col       = self.end_slope_col
        if weight_col is None:          weight_col          = self.weight_col
        if color_by is None:            color_by            = self.color_by
        if sort_by is None:             sort_by             = self.sort_by
        if cmap is None:                cmap                = self.cmap   
        if total_color is None:         total_color         = self.total_color
        if res is None:                 res                 = self.res     
        if time_scaling is None:        time_scaling        = self.time_scaling
        if verbose is None:             verbose             = self.verbose
        if plot_kwargs is None:         plot_kwargs         = self.plot_kwargs
        if fill_between_kwargs is None: fill_between_kwargs = self.fill_between_kwargs
        if axvline_kwargs is None:      axvline_kwargs      = self.axvline_kwargs
        if text_kwargs is None:         text_kwargs         = self.text_kwargs
        if grid_kwargs is None:         grid_kwargs         = self.grid_kwargs

        if isinstance(color_by, int):   col_cmap  = df.columns[color_by]
        else:                           col_cmap  = color_by
        if isinstance(sort_by, int):    col_sort  = df.columns[sort_by]
        else:                           col_sort  = sort_by


        df, \
        start, end, duration, completed, start_slope, end_slope, weight, \
        col_start, col_end, col_dur, col_comp, col_start_slope, col_end_slope, \
        col_weight, \
            = self.__get_missing(
                df=df, start_col=start_col, end_col=end_col, dur_col=dur_col, comp_col=comp_col,
                start_slope_col=start_slope_col, end_slope_col=end_slope_col,
                weight_col=weight_col,
                verbose=verbose,
        )

        #get increments in time to reach res datapoints
        ls_interval = (end.max()-start.min())/timedelta(res)

        #time only needed for plotting
        time      = pl.date_range(start.min(), end.max(), interval=timedelta(ls_interval))

        workload, total_workload = self.workload_curve(
            x=time,
            start=start, end=end, duration=duration, completed=completed,
            weight=weight, start_slope=start_slope, end_slope=end_slope,
            time_scaling=time_scaling,
        )

        #time corresponding to fraction of task-completion
        t_comp = (start.cast(int)+completed.cast(int)).cast(pl.Datetime)

        workloads_to_plot = [
            total_workload,
            *workload.T,
        ]
        tcomps_to_plot = [
            np.inf,
            *t_comp.cast(int),
        ]


        #generate colormap for plot
        colors = self.__generate_cmap(
            df,
            col_cmap=col_cmap, cmap=cmap,
        )
        colors = np.append([total_color], colors, axis=0) #append black for total curve

        #plot onto axis
        if ax is not None:
            #plot curves
            for idx, (c, wtp, tctp) in enumerate(zip(colors, workloads_to_plot, tcomps_to_plot)):
                #boolean of what is completed (fill that)
                wherebool = (time.cast(int).to_numpy() < tctp)
                ax.plot(time, wtp, color=c, **plot_kwargs)
                if idx > 0:
                    ax.fill_between(time, wtp, color=c, where=wherebool, **fill_between_kwargs)

            #vertical line for today
            today = np.datetime64(datetime.now())
            ax.axvline(today, **axvline_kwargs)
            ax.text(x=today, s=' Today', **text_kwargs)

            #add gridlines
            ax.grid(**grid_kwargs)

            #labelling
            ax.tick_params(labelbottom=False)
            ax.set_ylabel('Relative Workload [-]')
        
        #plot onto current figure
        else:
            #plot curves
            for idx, (c, wtp, tctp) in enumerate(zip(colors, workloads_to_plot, tcomps_to_plot)):
                #boolean of what is completed (fill that)
                wherebool = (time.cast(int).to_numpy() < tctp)
                plt.plot(time, wtp, color=c, **plot_kwargs)
                if idx > 0:
                    plt.fill_between(time, wtp, color=c, where=wherebool, **fill_between_kwargs)

        return
    
    def plot(self,
        X:Union[pl.DataFrame,List[dict],str],
        plot_gantt_kwargs:dict=None, plot_workload_kwargs:dict=None,
        ) -> Tuple[Figure,plt.Axes]:
        """
            - method to create a combined workload-GANTT plot (2 panels) given tasks stored in `X`

            Parameters
            ----------
                - X
                    - pl.DataFrame, list, str
                    - object containing informations to individual tasks of your project
                    - if pl.DataFrame
                        - will be used as such
                    - if list
                        - has to contain dicts
                        - will be converted to pl.DataFrame
                            - i.e. `pl.DataFrame(X)` will be called
                    - if str
                        - will be interpreted as filename to csv file
                        - will load file into a pl.DataFrame
                        - i.e. `pl.read_csv(X)` will be called
                - `plot_gantt_kwargs`
                    - dict, optional
                    - kwargs to pass to `self.plot_gantt()`
                    - the default is `None`
                        - will be set to `{}`
                - `plot_workload_kwargs`
                    - dict, optional
                    - kwargs to pass to `self.plot_workload()`
                    - the default is `None`
                        - will be set to `{}`

            Raises
            ------

            Returns
            -------
                - `fig`
                    - matplotlib Figure
                    - created figure
                - `axs`
                    - plt.Axis
                    - axis corresponding to `fig`

            Comments
            --------

        """

        #initialize
        if plot_gantt_kwargs is None:    plot_gantt_kwargs = {}
        if plot_workload_kwargs is None: plot_workload_kwargs = {}

        #get correct type for X
        if isinstance(X, pl.DataFrame): df = X
        elif isinstance(X, str):        df = pl.read_csv(X)
        else:                           df = pl.DataFrame(X)

        #plot
        fig = plt.figure()
        ax2 = fig.add_subplot(212)
        ax1 = fig.add_subplot(211, sharex=ax2)

        #plotting
        self.plot_workload(df=df, ax=ax1, **plot_workload_kwargs)
        self.plot_gantt(   df=df, ax=ax2, **plot_gantt_kwargs)

        axs = fig.axes

        return fig, axs