
#%%imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
from typing import Callable, Any, List

from astroLuSt.visualization.plotting import generate_colors

#%%definitions
class ExecTimer:
    """
        - class to monitor and estimate program runtimes

        Attributes
        ----------
            - `verbose`
                - int, optional
                - verbosity level
                - the higher the more information will be displayed
                - the default is 1
            - `print_kwargs`
                - dict, optional
                - kwargs to pass to `print()`
                - the default is `None`
                    - will be set to `dict()`

        Infered Attributes
        ------------------
            - `df_protocoll`
                - pd.DataFrame
                - dataframe storing all the tasks created with one instance of `ExecTimer()`

        Methods
        -------
            - `check_taskname()`
            - `checkpoint_start()`
            - `checkpoint_end()`
            - `estimate_runtime()`
            - `get_execstats()`

        Dependencies
        ------------
            - matplotlib
            - numpy
            - pandas
            - time
            - typing

        Comments
        --------

    """

    def __init__(self,
        verbose:int=1,
        print_kwargs:dict=None,
        ) -> None:

        self.verbose = verbose
        if print_kwargs is None:    self.print_kwargs = dict()
        else:                       self.print_kwargs = print_kwargs

        self.df_protocoll = pd.DataFrame(
            columns=['Task', 'Start', 'End', 'Duration', 'Start_Seconds', 'End_Seconds', 'Duration_Seconds', 'Comment_Start', 'Comment_End'],
        )
        
        self.df_protocoll['Start'] = pd.to_datetime(self.df_protocoll['Start'])
        self.df_protocoll['End'] = pd.to_datetime(self.df_protocoll['End'])
        self.df_protocoll['Duration'] = pd.to_timedelta(self.df_protocoll['Duration'])
        
        return
    
    def __repr__(self) -> str:
        
        return (
            f'ExecTimer(\n'
            f'    verbose={repr(self.verbose)},\n'
            f')'
        )

    def check_taskname(self,
        taskname:str,
        ) -> str:
        """
            - method to check if a passed `taskname` is already existing in `self.df_protocoll`
            - if that is the case `taskname` will be modified by appending an unique index to it

            Parameters
            ----------
                - `taskname`
                    - str
                    - unique name given to the task
            
            Raises
            ------

            Returns
            -------
                - `taskname`
                    - str
                    - modified input (`taskname`)
                        - if the input was unique, it will be returned as is
                        - otherwise an index will be appended to it
                            - i.e. `tasknamen` will be returned for the n-th duplicate of `taskname`

            Comments
            --------
        """

        #make sure to only have unique tasknames
        addon = 1
        while taskname in self.df_protocoll['Task'].values:
            taskname = taskname.replace(str(addon-1),'')
            taskname += str(addon)
            addon += 1
        
        return taskname

    def checkpoint_start(self,
        taskname:str, comment:str=None,
        ) -> None:
        """
            - method to start a new task

            Parameters
            ----------
                - `taskname`
                    - str
                    - unique name given to the task
                - `comment`
                    - str, optional
                    - some comment to starting the task
                    - the default is `None`
                        - will be set to `''`
            
            Raises
            ------

            Returns
            -------

            Comments
            --------
        """

        #initialize
        if comment is None: comment = ''

        start_time = time.time()
        start_timestamp = np.datetime64('now')
        taskname = self.check_taskname(taskname)

        self.df_protocoll.loc[self.df_protocoll.shape[0]] = [
            taskname,
            start_timestamp,
            np.empty(1, dtype='datetime64[s]')[0],
            np.empty(1, dtype='timedelta64[s]')[0],
            start_time,
            np.nan,
            np.nan,
            comment,
            ''
        ]

        if self.verbose > 0:
            print('\n'+'#'*70, **self.print_kwargs)
            print(f'INFO: Started {taskname} at {start_timestamp}', **self.print_kwargs)


        return

    def checkpoint_end(self,
        taskname:str, comment:str='',
        ) -> None:
        """
            - method to wrap up a task of name `taskname`

            Parameters
            ----------
                - `taskname`
                    - str
                    - unique name given to the task to finish
                - `comment`
                    - str, optional
                    - some comment to wrapping up the task
                    - the default is `None`
                        - will be set to `''`
            
                
            Raises
            ------

            Returns
            -------

            Comments
            --------
        """

        #initialize
        if comment is None: comment = ''

        end_time = time.time()
        end_timestamp = np.datetime64('now')

        try:
            start_time      = self.df_protocoll[(self.df_protocoll['Task']==taskname)]['Start_Seconds'].values[0]
            start_timestamp = self.df_protocoll[(self.df_protocoll['Task']==taskname)]['Start'].values[0]
        except IndexError as ie:
            msg = (
                f'IndexError occured. Probably your provided `taskname` has never been initialized.'
                f'Make sure to initialize a `taskname` before calling `checkpoint_end()`!\n'
                f'Original Error: {ie}.'
            )
            raise LookupError(msg)

        duration = end_time-start_time
        duration_timedelta = pd.to_timedelta(end_timestamp-start_timestamp)

        cur_task = np.where(self.df_protocoll['Task']==taskname)[0][0]


        self.df_protocoll.at[cur_task, 'End'] = pd.to_datetime(end_timestamp)
        self.df_protocoll.at[cur_task, 'Duration'] = duration_timedelta
        self.df_protocoll.at[cur_task, 'Duration_Seconds'] = duration
        self.df_protocoll.at[cur_task, 'End_Seconds'] = end_time
        self.df_protocoll.at[cur_task, 'Comment_End'] = comment


        if self.verbose > 0:
            print(
                f'\n'
                f'INFO: Finished {taskname} at {end_timestamp}\n'
                f'Required time: {pd.to_timedelta(self.df_protocoll.at[cur_task, "Duration"])}',
                **self.print_kwargs
            )
            print('#'*70, **self.print_kwargs)
        return

    def estimate_runtime(self,
        taskname_pat:str,
        nrepeats:int, ndone:int=1,
        ) -> None:
        """
            - method to estimate the total runtime in dependence of how many repetition have been made and will be made

            Parameters
            ----------
                - `taskname_pat`
                    - str
                    - regular expression to query the `self.df_protocoll['Task']`
                        - all tasks that contain `taskname_pat` will contribute to the runtime-estimate
                - `nrepeats`
                    - int
                    - how often the task will be repeated
                - `ndone`
                    - int, optional
                    - how often the task has been executed already
                    - the default is 1
            
            Raises
            ------

            Returns
            -------

            Comments
            --------
        """

        tasks_to_consider_bool = self.df_protocoll['Task'].str.contains(taskname_pat, regex=True)

        cur_runtime = np.nansum(self.df_protocoll[tasks_to_consider_bool]['Duration'])

        runtime_estimate = cur_runtime*nrepeats/ndone

        print(f'INFO: Total estimated runtime for {nrepeats} repeats: {runtime_estimate}', **self.print_kwargs)        


        return

    def time_exec(self,
        taskname:str='Decorator Task',
        start_kwargs:dict=None,
        end_kwargs:dict=None
        ) -> Any:
        """
            - decorator method to always time a function at execution

            Parameters
            ----------
                - `taskname`
                    - str, optional
                    - unique name given to the task
                    - the default is `'Decorator Task'`
                - `start_kwargs`
                    - dict, optional
                    - kwargs to pass to `self.checkpoint_start()`
                    - the default is `None`
                        - will default to `{}`
                - `end_kwargs`
                    - dict, optional
                    - kwargs to pass to `self.checkpoint_end()`
                    - the default is `None`
                        - will default to `{}`
            Raises
            ------

            Returns
            -------
                - `wrap`
                    - Any
                    - output of wrapped (decorated) function

            Comments
            --------
                - use like decorator, i.e. as follows
    	            
                    ```python
                    >>> @ExecTimer().time_exec(*args)
                    >>> def func(...):
                    >>>     ...
                    >>>     return ...
                    ```

        """

        #initialize accordingly
        if start_kwargs is None:
            start_kwargs = {}
        if end_kwargs is None:
            end_kwargs = {}
        
        #function to wrap decorated function (func)
        def wrap(func:Callable):

            #function to execute wrapped function
            def wrapped_func(*args, **kwargs):
                
                # #check if a unique taskname was passed and modify accordingly
                taskname_use = self.check_taskname(taskname)
                
                self.checkpoint_start(taskname=taskname_use, **start_kwargs)
                func_res = func(*args, **kwargs)
                self.checkpoint_end(taskname=taskname_use, **end_kwargs)
                
                #return function result
                return func_res
            
            #return wrapped function (ultimately returns func_res)
            return wrapped_func
        
        #return wrap (ultimately returns wrapped_func and thus func_res)
        return wrap
    
    def get_execstats(self,
        n:int=500,
        metrics:List[Callable]=None,
        drop_from_df_protocoll:bool=True,
        ) -> pd.DataFrame:
        """
            - decorator method to execute a function `n` times and return a plot of requested statistics

            Parameters
            ----------
                - `n`
                    - int, optional
                    - number of times the function shall be executed in order to get a statistics
                    - the default is 500
                - `metrics`
                    - list, optional
                    - contains callables
                    - callables calculating the requested statistics
                    - the default is `None`
                        - defaults to `[np.nanmean, np.nanmedian, np.nanmin, np.nanmax]`
                - `drop_from_df_protocoll`
                    - bool, optional
                    - whether to drop the related entries from `self.df_protocoll`
                    - the default is `True`
            
            Raises
            ------

            Returns
            -------
                - `df_execstats`
                    - pd.DataFrame
                    - dataframe entries of the executions within `self.get_execstats()`

            Comments
            --------
                - use like decorator, i.e. as follows
                    ```python
                    >>> @ExecTimer().get_execstats(*args)
                    >>> def func(...):
                    >>>     ...
                    >>>     return ...
                    ```
                            
        """

        #initialize
        if metrics is None:
            metrics = [np.nanmean, np.nanmedian, np.nanmin, np.nanmax]

        #actual calculation
        def wrap(func:Callable):

            def wrapped_func(*args, **kwargs):
                #execute func n times
                for ni in range(n):
                    taskname_use = self.check_taskname('get_execstats()')
                    comment_use = f'__get_execstats()__'
                    self.checkpoint_start(taskname=taskname_use, comment=comment_use)
                    func_res = func(*args, **kwargs)
                    self.checkpoint_end(taskname=taskname_use, comment=comment_use)

                execstats_bool = '((Comment_Start == @comment_use)&(Comment_End == @comment_use))'
                df_execstats = self.df_protocoll.query(execstats_bool)

                #visualize statistics
                fig = plt.figure()
                ax1 = fig.add_subplot(111)

                ax1.hist(df_execstats['Duration_Seconds'], bins='sqrt', alpha=0.5)
                colors = generate_colors(len(metrics))
                for m, c in zip(metrics, colors):
                    val = m(df_execstats['Duration_Seconds'])
                    ax1.axvline(val, color=c, label=f'{m.__name__} = {val:.3f} s')

                #std
                ax1.scatter(np.nan, np.nan, color='none', label=f'nanstd = {np.nanstd(df_execstats["Duration_Seconds"]):.3f} s')

                ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')

                ax1.legend()

                ax1.set_xlabel('Elapsed Time [s]')
                ax1.set_ylabel('Counts')

                plt.show()

                #drop get_execstats() executions from df_protocoll if requested
                if drop_from_df_protocoll:
                    self.df_protocoll = self.df_protocoll.query('~'+execstats_bool)


                #return dataframe of statistics
                return df_execstats
            
            #return wrapped function (ultimately returns func_res)
            return wrapped_func
        
        #return wrap (ultimately returns wrapped_func and thus func_res)
        return wrap