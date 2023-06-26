

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
class GANTT_:
    """
        - class to genereate a variation of a GANTT-chart
        
        Attributes
        ----------
            - `time`
                - np.ndarray
                - array of the period of time where the project takes place
                - has to be an array of datetime-objects
        
        Infered Attributes
        ------------------
            - `starts`
                - np.ndarray
                - array of datetime-objects
                - contains the starting point relative to `time.min()`
                - i.e. the first task starts usually at `start = 0`
            - `ends`
                - np.ndarray
                - array of datetime-objects
                - contains the endpoint relative to `time.min()`
            - `tasks`
                - np.ndarray
                - relative workload for the tasks to be done (dependent on time)
            - `tasknames`
                - list
                - nametag for each task
            - `weights`
                - np.ndarray
                - array of weights weighting the importance of each task
            - `percent_complete`
                - np.ndarray
                - array of percentages
                - defines how much of a specific task is completed

        Methods
        -------
            - `sigmoid()`
            - `task_func()`
            - `task()`
            - `make_graph()`
            - `make_classic_gantt()`

        Dependencies
        ------------
            - datetime
            - matplotlib
            - numpy

        Comments
        --------
            - `whole_area` (argument of `self.task()`), will be adjusted for each graph separately
                - make sure you set `whole_area` to the value you need, when passing a task to the class

    """

    def __init__(self,
        time:np.ndarray
        ) -> None:
        
        if not all([isinstance(t, datetime) for t in time]):
            raise TypeError("`time` has to be an array containing `datetime` objects!")
        self.time = time
        self.starts = np.array([])
        self.ends = np.array([])
        self.tasks = np.array([])
        self.tasknames = []
        self.weights = np.ones(len(self.tasks))

        self.percent_complete = np.array([])
        self.percent_idxi = np.array([], dtype=int)

    def __repr__(self) -> str:
        return (
            f'GANTT(\n'
            f'    time={repr(self.time)}'
            f')'
        )

    def sigmoid(self,
        x:np.ndarray,
        slope:np.ndarray=1, shift:np.ndarray=0
        ) -> np.ndarray:
        """
            - method to calculate a sigmoid function

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
                    - np.ndarray
                    - parameter to shift the sigmoid in direction of `x`
                    - the default is 0

            Raises
            ------

            Returns
            -------
                - `sigma`
                    - np.ndarray
                    - sigmoid evaluated on `x`

            Comments
            --------

        """

        Q1 = 1 + np.e**(slope*(-(x - shift)))
        sigma = 1/Q1
        return sigma

    def task_func(self,
        time:np.ndarray,
        start:float, end:float,
        start_slope:float=1, end_slope:float=1
        ) -> np.ndarray:
        """
            - method to calculate a specific tasks workload curve
            - combines two sigmoids with opposite signs in the exponent to do that
        
            Parameters
            ----------
                - `time`
                    - np.ndarray
                    - the times (relative to the starting time) used for the tasks
                        - i.e. an array starting with 0
                - `start`
                    - float
                    - time at which the tasks starts
                    - relative to the zero point in time (i.e. `time.min() = 0`)
                - `end`
                    - float
                    - time at which the task ends
                    - relative to the zero point in time (i.e. `time.min() = 0`)
                - `start_slope`
                    - float, optional
                    - how steep the starting phase should be
                    -  the default is 1
                - `end_slope`
                    - float
                    - how steep the ending phase (reflection phase) should be
                    - the default is 1

            Raises
            ------

            Returns
            -------
                - task
                    - np.ndarray
                    - workload curve for `time`

            Comments
            --------
        """

        start_phase = self.sigmoid(time, start_slope, start)
        end_phase   = self.sigmoid(-time, end_slope, -end)
        task = start_phase + end_phase
        task -= task.min()
        task /= task.max()
        
        return task

    def task_integral(self,
        time,
        start:float, end:float,
        start_slope:float=1, end_slope:float=1,
        percent_complete:float=0,
        whole_area:bool=True
        ) -> int:
        """
            - method to estimate the index corresponding to the `percent_completed`
                - To do this:
                    - calculates the intergal of the function used to define a task (`self.task_func()`)
                    - i.e. 2 sigmoids with opposite signs befor the exponent
                - algebraic solution (latex formatting):
                    
                    ```Latex
                    >>> \\begin{align}
                    >>>     \\frac{\ln(e^{s_1a_1} + e^{s_1x})}{s_1}
                    >>>         - \\frac{\ln(e^{s_2a_2 - s_2x} + 1)}{s_2}
                    >>>         - \\mathcal{C}
                    >>> \\end{align}
                    ```

            Parameters
            ----------
                - `time`
                    - np.ndarray
                    - the times (relative to the starting time) used for the tasks
                        - i.e. an array starting with 0
                - `start`
                    - float
                    - time at which the tasks starts
                    - relative to the zero point in time (i.e. `time.min() = 0`)
                - `end`
                    - float
                    - time at which the task ends
                    - relative to the zero point in time (i.e. `time.min() = 0`)
                - `start_slope`
                    - float, optional
                    - how steep the starting phase should be
                    -  the default is 1
                - `end_slope`
                    - float
                    - how steep the ending phase (reflection phase) should be
                    - the default is 1
                - `percent_complete`
                    - float, optional
                    - percentage describing how much of the task is completed
                    - number between 0 and 1
                    - the default is 0
                - `whole_area`
                    - bool, optional
                    - whether to consider the whole area beneath the curves as 100% of the task
                        - this will especially in low percentages NOT line up with the actual GANTT-chart (second subplot) 
                    - otherwise will consider the area between `start` and `end` as 100% of the task
                        - will be exactly aligned with the GANTT-chart
                        - but it might be that 0% already has some area colored in
                    - the default is `True`

            Raises
            ------

            Returns
            -------
                - `percent_idx`
                    - int
                    - index time that is closest to `percent_complete` of the whole time interval

            Comments
            --------

        """

        def ln1(bound):
            return np.log(np.e**(start_slope*bound) + np.e**(start_slope*start))
        def ln2(bound):
            return np.log(np.e**(end_slope*end-end_slope*bound)   + 1)

        #integral over whole interval
        if whole_area:
            #upper bound
            T1 = ln1(time.max())
            T2 = ln2(time.max())
            #upper bound
            T3 = ln1(time.min())
            T4 = ln2(time.min())
        else:
            #upper bound
            T1 = ln1(end)
            T2 = ln2(end)
            #lower bound
            T3 = ln1(start)
            T4 = ln2(start)
        int_whole_interval = T1/start_slope - T2/end_slope - (T3/start_slope - T4/end_slope)
        
        #ingetrals for all time-points
        T1_i = ln1(time)
        T2_i = ln2(time)
        int_times = T1_i/start_slope - T2_i/end_slope - (T3/start_slope - T4/end_slope)

        #get index of time that is the closest to percent_complete from the total time interval
        percent_idx = np.argmin(np.abs(int_times - int_whole_interval*percent_complete))

        return percent_idx

    def task(self,
        start:float, end:float,
        start_slope:float=1, end_slope:float=1,
        taskname:str=None,
        weight:float=1, percent_complete:float=0,
        whole_area:bool=True,
        testplot:bool=False
        ) -> np.ndarray:
        """
            - method to define a specific task
            - will add that task to 'self.tasks' as well

            Parameters
            ----------
                - `start`
                    - float
                    - time at which the tasks starts
                    - relative to the zero point in time (i.e. `time.min() = 0`)
                - `end`
                    - float
                    - time at which the task ends
                    - relative to the zero point in time (i.e. `time.min() = 0`)
                - `start_slope`
                    - float, optional
                    - how steep the starting phase should be
                    -  the default is 1
                - `end_slope`
                    - float
                    - how steep the ending phase (reflection phase) should be
                    - the default is 1
                - `taskname`
                    - str, optional
                    - name of the task added
                    - the default is `None`
                        - Will generate `'Taskn'`, where `n` is the current number of tasks + 1
                - `weight`
                    - float, optional
                    - weight to set the importance of the task with respect to the other tasks
                    - the default is 1
                - `percent_complete`
                    - float, optional
                    - percentage describing how much of the task is completed
                    - number between 0 and 1
                    - the default is 0
                - `whole_area`
                    - bool, optional
                    - whether to consider the whole area beneath the curves as 100% of the task
                        - this will especially in low percentages NOT line up with the actual GANTT-chart (second subplot) 
                    - otherwise will consider the area between `start` and `end` as 100% of the task
                        - will be exactly aligned with the GANTT-chart
                        - but it might be that 0% already has some area colored in
                    - the default is `True`
                - `testplot`
                    - bool, optional
                    - whether to show a testplot of the created task
                    - the default is `False`

            Raises
            ------
                - `ValueError`
                    - if `percent_complete` bigger than 1 or smaller than 0 is passed

            Returns
            -------
                - `task`
                    - np.ndarray
                    - an array of the percentages of workload over the whole task
            
            Comments
            --------


        """

        if percent_complete < 0 or percent_complete > 1:
            raise ValueError("'percent_complete has to be a float between 0 and 1!")

        #array of unit-timesteps relative to the starting date
        calc_time = np.arange(0, self.time.shape[0], 1)

        #update self.starts and self.ends
        
        if end > self.time.shape[0]-1:
            end_idx = self.time.shape[0]-1
        else:
            end_idx = end
        if start < 0:
            start_idx = 0
        else:
            start_idx = start
        self.starts = np.append(self.starts, self.time[int(start_idx)])
        self.ends   = np.append(self.ends,   self.time[int(end_idx)])

        #define task
        task = self.task_func(calc_time, start, end, start_slope, end_slope)

        if self.tasks.shape[0] == 0:
            self.tasks = np.append(self.tasks, task)
        else:
            self.tasks = np.vstack((self.tasks, task))
        
        #add weight
        self.weights = np.append(self.weights, weight)
        
        #add percentage that has been completed
        self.percent_complete = np.append(self.percent_complete, percent_complete)
        
        #add index of percentage that has been completed
        percent_idx = self.task_integral(calc_time, start, end, start_slope, end_slope, percent_complete, whole_area)
        self.percent_idxi = np.append(self.percent_idxi, percent_idx)
        
        #add task-name
        if taskname is None:
            taskname = f"Task {len(self.tasknames)+1:d}"
        self.tasknames.append(taskname)

        if testplot:
            fig = plt.figure(figsize=(16,9))
            ax = fig.add_subplot(111)
            fig.suptitle(f"Testplot for your {taskname}", fontsize=24)
            ax.plot(self.time, task, "-", label=taskname)
            ax.fill_between(self.time, task, where=(self.time < self.time[percent_idx]), alpha=.3, label=f"completion ({percent_complete*100}%)")
            ax.set_xlabel("Time [Your Unit]", fontsize=20)
            ax.set_ylabel("Relative Workload (Within Task) [-]", fontsize=20)
            ax.tick_params("both", labelsize=20)
            plt.tight_layout()
            plt.legend(fontsize=20)
            plt.show()

        return task


    def plot_result(self,
        today:float=None,
        colors:np.ndarray=None, enumerate_tasks:bool=True,
        show_totalwork:bool=True, show_completion:bool=True,
        plot_kwargs:dict=None, fill_between_kwargs:dict=None,
        ) -> Tuple[np.ndarray,Figure,plt.Axes]:
        """
            - method to visualize the workload of a project w.r.t. the time
            - will create a plot inlcuding a clssical GANTT-plot and a variation

            Parameters
            ----------
                - `today`
                    - float, optional
                    - current state
                    - will plot a vertical line at the current state in the plot created
                    - only relevant for the plot created
                    - the default is `None`
                        - will not plot a line
                - `colors`
                    - np.ndarray, optional
                    - list of matplotlib colors or rgb-tupels
                    - the default is `None`
                        - will generate colors automatically
                - `enumerate_tasks`
                    - bool, optional
                    - whether to enumerate the tasks contained in the GANTT-instance
                        - will enumerate them in the order they got added to the GANTT-instance
                    - the default is `True`
                - `show_totalwork`
                    - bool, optional
                    - whether to add a plot of the total work for each point in time
                    - the default is `True`
                - `show_completion`
                    - bool, optional
                    - whether to show the completion of individual tasks
                    - the default is `True`
                - `plot_kwargs`
                    - dict, optional
                    - kwargs passed to `ax.plot()`
                    - the default is `None`
                        - will be set to `{}`
                - `fill_between_kwargs`
                    - dict, optional
                    - kwargs passed to `ax.fill_between()`
                    - the default is `None`
                        - will be set to `{'alpha':0.3}`
                
                Raises
                ------
                    - `TypeError`
                        - if `colors` has wrong type
                    - `ValueError`
                        - if `colors` has other length than `self.tasks`

                Returns
                -------
                    - `tasks_combined`
                        - np.array
                        - combination of all tasks
                        - the maximum workload for a given point in time of all tasks combined is 100% 
                    - `fig`
                        - matplotlib Figure
                        - created figure
                    - `axs`
                        - plt.Axes
                        - axes corresponding to `fig`

                Comments
                --------
                    - `whole_area` (argument of `self.task`), will be adjusted for each graph separately
                        - make sure you set `whole_area` to the value you need, when passing a task to the class

        """

        #initialize
        if fill_between_kwargs is None: fill_between_kwargs = {'alpha':0.3}
        if plot_kwargs is None: plot_kwargs = {}

        if colors is None:
            colors = generate_colors(len(self.tasks)+2, cmap='nipy_spectral')[1:-1]
        elif not isinstance(colors, (np.ndarray, list)):
            raise TypeError("'colors' has to be a list or an np.array!")
        elif len(colors) != len(self.tasks):
            raise ValueError("'colors' has to have the same length as 'self.tasks'!")
        
        if today is None:
            today = self.time.min()
        
        tasks_zeromin = self.tasks-self.tasks.min()
        
        try:
            tasks_zeromin.shape[1]
            weights = np.sum(tasks_zeromin, axis=0)
        except:
            weights = tasks_zeromin
        
        tasks_combined = tasks_zeromin/weights.max()



        #create figure
        fig = plt.figure()
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)


        #################
        #GANTT variation#
        #################

        #all tasks
        try:
            tasks_combined.shape[1]
        except:
            tasks_combined = np.array([tasks_combined])
        tasks_combined = (tasks_combined.T*self.weights).T    #weight all tasks

        for idx, (task, percent_idx, percent_complete, c) in enumerate(zip(tasks_combined, self.percent_idxi, self.percent_complete, colors)):
            ax1.plot(self.time, task, zorder=2+(1/(idx+1)), color=c, **plot_kwargs)
            if show_completion:
                ax1.fill_between(self.time, task, where=(self.time < self.time[percent_idx]), zorder=1+(1/(idx+1)), color=c, **fill_between_kwargs)

        #total workload
        if show_totalwork:
            ax1.plot(self.time, np.sum(tasks_combined, axis=0), color="k", label="Total workload", zorder=0.9, **plot_kwargs)
            ax1.fill_between(self.time, np.sum(tasks_combined, axis=0), where=(self.time>today), color="tab:grey", label="TODO", zorder=0.9, **fill_between_kwargs)
            if today is not None:
                ax1.fill_between(self.time, np.sum(tasks_combined, axis=0), where=(self.time<today), color="tab:green", label="Finished", zorder=0.9, **fill_between_kwargs)


        #######
        #GANTT#
        #######

        ax2.plot(self.time, np.ones_like(self.time), alpha=0)   #just needed to get correct xaxis-labels

        text_shift = self.time[1]-self.time[0]
        for idx, (tn, start, end, percent_complete, c) in enumerate(zip(self.tasknames[::-1], self.starts[::-1], self.ends[::-1], self.percent_complete[::-1], colors[::-1])):
            if enumerate_tasks:
                label = f"{len(self.tasks)-idx:d}. {tn}"
            else:
                label = f"{tn}"
            duration = end - start
            completion = duration*percent_complete
            ax2.barh(tn, completion, left=start, color=c, alpha=1.0, zorder=2)
            ax2.barh(tn, duration, left=start, color=c, alpha=0.5, zorder=1)
            ax2.text(end+text_shift, idx, f"{percent_complete*100}%", va="center", alpha=0.8, fontsize=18)
            ax2.text(start-text_shift, idx, label, va="center", ha="right", alpha=0.8, fontsize=18)
        
        #current point in time
        ax1.vlines(today, ymin=0, ymax=1, color="k", zorder=20, label="Today", linestyle="--", linewidth=2.5)
        ax2.vlines(today, ymin=-1, ymax=self.tasks.shape[0], color="k", zorder=20, label="Today", linestyle="--", linewidth=2.5)

        
        #make visually appealing
        ax2.set_ylim(-1, self.tasks.shape[0])
        ax2.minorticks_on()
        ax2.tick_params(axis='y', which='minor', left=False)
        ax2.xaxis.grid(which="both", zorder=0)
        ax2.get_yaxis().set_visible(False)
        
        #create legend
        ax1.legend()

        #labelling
        ax1.set_xlabel("")
        ax1.set_ylabel("Relative Workload [-]", fontsize=20)
        ax1.set_xticklabels([])
        # ax1.set_xticks([])
        ax1.get_shared_x_axes().join(ax1, ax2)
        
        ax2.set_xlabel("Time")


        axs = fig.axes
        

        return tasks_combined, fig, axs

    def plot_classic_gantt(self,
        today:float=None,
        colors:np.ndarray=None, enumerate_tasks:bool=False,
        ) -> Tuple[Figure,plt.Axes]:
        """
            - method to create a GANTT-graph in classical style
            
            Parameters
            ----------
                - `today`
                    - float, optional
                    - current state
                    - will plot a vertical line at the current state in the plot created
                    - only relevant for the plot created
                    - the default is None (will not plot a line)
                - `colors`
                    - np.ndarray, optional
                    - list of matplotlib colors or rgb-tupels
                    - the default is `None`
                        - will generate colors automatically
                - `enumerate_tasks`
                    - bool, optional
                    - whether to enumerate the tasks contained in the GANTT-instance
                        - will enumerate them in the order they got added to the GANTT-instance
                    - only relevant for the plot created
                    - the default is `True`
                
                Raises
                ------
                    - `TypeError`
                        - if `colors` has wrong type
                    - `ValueError`
                        - if `colors` has other length than `self.tasks`

                Returns
                -------
                    - `fig`
                        - matplotlib Figure
                        - created figure
                    - `axs`
                        - plt.Axes
                        - axes corresponding to `fig`

                Comments
                --------

        """

        #check shapes
        if colors is None:
            colors = generate_colors(len(self.tasks)+2, cmap='nipy_spectral')[1:-1]
        elif type(colors) != (np.array and list):
            raise TypeError("'colors' has to be a list or an np.array!")
        elif len(colors) != len(self.task):
            raise ValueError("'colors' has to have the same length as 'self.tasks'!")

        #create plot
        fig = plt.figure()
        ax1 = fig.add_subplot(111)

        ax1.plot(self.time, np.ones_like(self.time), alpha=0)   #just needed to get correct xaxis-labels

        text_shift = self.time[1]-self.time[0]
        for idx, (tn, start, end, percent_complete, c) in enumerate(zip(self.tasknames[::-1], self.starts[::-1], self.ends[::-1], self.percent_complete[::-1], colors[::-1])):
            if enumerate_tasks:
                label = f"{len(self.tasks)-idx:d}. {tn}"
            else:
                label = f"{tn}"
            duration = end - start
            completion = duration*percent_complete
            ax1.barh(tn, completion, left=start, color=c, alpha=1.0, zorder=2)
            ax1.barh(tn, duration, left=start, color=c, alpha=0.5, zorder=1)
            ax1.text(end+text_shift, idx, f"{percent_complete*100}%", va="center", alpha=0.8, fontsize=18)
            ax1.text(start-text_shift, idx, label, va="center", ha="right", alpha=0.8, fontsize=18)
        
        #current point in time
        ax1.vlines(today, ymin=-1, ymax=self.tasks.shape[0], color="k", zorder=20, label="Today", linestyle="--", linewidth=2.5)

        #make visually appealing
        ax1.set_ylim(-1, self.tasks.shape[0])
        ax1.minorticks_on()
        ax1.tick_params(axis='y', which='minor', left=False)
        ax1.xaxis.grid(which="both", zorder=0)
        ax1.get_yaxis().set_visible(False)

        ax1.legend()

        axs = fig.axes


        return fig, axs

class GANTT:

    def  __init__(self,
        verbose:int=0,
        ) -> None:

        self.verbose = verbose

        return
    
    def __get_missing(
        self,
        df:pl.DataFrame,
        start_col:Union[str,int]=None, end_col:Union[str,int]=None, dur_col:Union[str,int]=None, comp_col:Union[str,int]=None,
        start_slope_col:Union[str,int]=None, end_slope_col:Union[str,int]=None,
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

        start_slope = df[start_slope_col]
        end_slope   = df[end_slope_col]


        if isinstance(start_col, int):          col_start = df.columns[start_col]
        else:                                   col_start = start_col
        if isinstance(end_col, int):            col_end   = df.columns[end_col]
        else:                                   col_end   = end_col
        if isinstance(dur_col, int):            col_dur   = df.columns[dur_col]
        else:                                   col_dur   = dur_col
        if isinstance(comp_col, int):           col_comp  = df.columns[comp_col]
        else:                                   col_comp  = comp_col
        if isinstance(start_slope_col, int):    col_start_slope  = df.columns[start_slope_col]
        else:                                   col_start_slope  = start_slope_col
        if isinstance(end_slope_col, int):      col_end_slope  = df.columns[end_slope_col]
        else:                                   col_end_slope  = end_slope_col


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

        return (
            df,
            start, end, duration, completed, start_slope, end_slope,
            col_start, col_end, col_dur, col_comp, col_start_slope, col_end_slope,
        )

    def __generate_cmap(self,
        df:pl.DataFrame,
        col_cmap:str, cmap:str,
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
                    - str
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
            - method to calculate a sigmoid function

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
                    - sigmoid evaluated on `x`

            Comments
            --------

        """

        Q1 = 1 + np.e**(slope*(-(x - shift)))
        sigma = 1/Q1

        return sigma
  
    def workload_curve(self,
        time:pl.Series,
        start:pl.Series, end:pl.Series, duration:pl.Series, completed:pl.Series=None,
        start_slope:pl.Series=None, end_slope:pl.Series=None,
        weight:Union[pl.Series,float]=1,
        time_scaling:float=1E-12,
        ) -> Tuple[np.ndarray,np.ndarray]:
        #TODO: Docstring
        """
            - method to calculate a specific tasks workload curve
            - combines two sigmoids with opposite signs in the exponent to do that
        
            Parameters
            ----------
                - `x`
                    - np.ndarray
                    - the times (relative to the starting time) used for the tasks
                        - i.e. an array starting with 0
                - `start`
                    - float
                    - time at which the tasks starts
                    - relative to the zero point in time (i.e. `time.min() = 0`)
                - `end`
                    - float
                    - time at which the task ends
                    - relative to the zero point in time (i.e. `time.min() = 0`)
                - `start_slope`
                    - float, optional
                    - how steep the starting phase should be
                    -  the default is 1
                - `end_slope`
                    - float
                    - how steep the ending phase (reflection phase) should be
                    - the default is 1

            Raises
            ------

            Returns
            -------
                - task
                    - np.ndarray
                    - workload curve for `time`

            Comments
            --------
        """

        #get workload-curves
        ##no offset by start, because start is the zeropoint
        time_eval = np.array(len(start)*[time.cast(int)])
        l_start = self.sigmoid(time_eval.T, slope= start_slope.cast(float).to_numpy()*time_scaling, shift=(start.cast(int).to_numpy()))
        l_end   = self.sigmoid(time_eval.T, slope=-end_slope.cast(float).to_numpy()  *time_scaling, shift=(start.cast(int).to_numpy()+duration.cast(int).to_numpy()))

        ##weight curves
        workload = (l_start+l_end)
        workload -= workload.min()
        workload /= np.nanmax(workload, axis=0)
        workload *= weight


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
        color_by:Union[str,int]=0, sort_by:Union[str,int]=0,
        cmap:str='nipy_spectral',
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
                    - str, optional
                    - colormap to use for coloring bars by `color_by`
                    - the default is `nipy_spectral`
                - `verbose`
                    - int, optional
                    - verbosity level
                    - will override `self.verbose`
                    - the default is `None`
                        - will fall back to `self.verbose`
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

            Raises
            ------

            Returns
            -------

            Comments
            --------
        """

        if verbose is None:        verbose = self.verbose
        if axvline_kwargs is None: axvline_kwargs = {'color':'k', 'linestyle':'--'}
        if text_kwargs is None:    text_kwargs    = {'ha':'left', 'va':'bottom', 'y':0}
        if grid_kwargs is None:    grid_kwargs    = {'visible':True, 'axis':'x'}

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
        start, end, duration, completed, start_slope, end_slope, \
        col_start, col_end, col_dur, col_comp, col_start_slope, col_end_slope \
            = self.__get_missing(
                df=df, start_col=start_col, end_col=end_col, dur_col=dur_col, comp_col=comp_col,
                start_slope_col=None, end_slope_col=None,
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
        color_by:Union[str,int]=0, sort_by:Union[str,int]=0,
        cmap:str='nipy_spectral',
        res:int=100,
        verbose:int=None,
        plot_kwargs:dict=None, fill_between_kwargs:dict=None, axvline_kwargs:dict=None, text_kwargs:dict=None, grid_kwargs:dict=None
        ) -> None:

        if verbose is None:             verbose                 = self.verbose
        if plot_kwargs is None:         plot_kwargs             = {}
        if fill_between_kwargs is None: fill_between_kwargs     = {'alpha':0.5}
        if axvline_kwargs is None:      axvline_kwargs          = {'color':'k', 'linestyle':'--'}
        if text_kwargs is None:         text_kwargs             = {'ha':'left', 'va':'bottom', 'y':0}
        if grid_kwargs is None:         grid_kwargs             = {'visible':True, 'axis':'x'}

        if isinstance(color_by, int):   col_cmap  = df.columns[color_by]
        else:                           col_cmap  = color_by
        if isinstance(sort_by, int):    col_sort  = df.columns[sort_by]
        else:                           col_sort  = sort_by


        df, \
        start, end, duration, completed, start_slope, end_slope, \
        col_start, col_end, col_dur, col_comp, col_start_slope, col_end_slope \
            = self.__get_missing(
                df=df, start_col=start_col, end_col=end_col, dur_col=dur_col, comp_col=comp_col,
                start_slope_col=start_slope_col, end_slope_col=end_slope_col,
                verbose=verbose,
        )

        #get increments in time to reach res datapoints
        ls_interval = (end.max()-start.min())/timedelta(res)

        #time only needed for plotting
        time      = pl.date_range(start.min(), end.max(), interval=timedelta(ls_interval))

        workload, total_workload = self.workload_curve(
            time=time,
            start=start, end=end, duration=duration, completed=completed,
            weight=1, start_slope=start_slope, end_slope=end_slope,
            time_scaling=1E-12,
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
        colors = np.append([[0,0,0,1]], colors, axis=0) #append black for total curve

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
            # ax.set_xticklabels([])
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