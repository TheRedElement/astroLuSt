
#%%imports
import pandas as pd
import numpy as np


#%%definitions
class ExecTimer:
    """
        - class to monitor and estimate program runtimes

        Attributes
        ----------
            - verbose
                - int, optional
                - verbosity level
                - the higher the more information will be displayed
            - df_protocoll
                - pandas DataFrame
                - dataframe storing all the tasks created with one instance of ExecTimer

        Methods
        -------
            - checkpoint_start()
            - checkpoint_end()
            - estimate_runtime()

        Dependencies
        ------------
            - pandas
            - numpy

        Comments
        --------

    """

    def __init__(self,
        verbose:int=1
        ) -> None:

        self.verbose = verbose
        self.df_protocoll = pd.DataFrame(
            columns=["Task", "Start", "End", "Duration", "Comment_Start", 'Comment_End'],
        )
        
        self.df_protocoll["Start"] = pd.to_datetime(self.df_protocoll["Start"])
        self.df_protocoll["End"] = pd.to_datetime(self.df_protocoll["End"])
        self.df_protocoll["Duration"] = pd.to_timedelta(self.df_protocoll["Duration"])
        
        return

    def checkpoint_start(self,
        taskname:str, comment:str=""
        ) -> None:
        """
            - method to start a new task

            Parameters
            ----------
                - taskname
                    - str
                    - unique name given to the task
                - comment
                    - str, optional
                    - some comment to the task
                    - the default is ""
            
            Raises
            ------

            Returns
            -------

            Comments
            --------
        """

        start_time = np.datetime64('now')

        #make sure to only have unique tasknames
        addon = 1
        while taskname in self.df_protocoll["Task"].values:
            taskname += str(addon)
            addon += 1

        self.df_protocoll.loc[self.df_protocoll.shape[0]] = [
            taskname,
            start_time,
            np.empty(1, dtype='datetime64[s]')[0],
            np.empty(1, dtype='timedelta64[s]')[0],
            comment,
            ''
        ]

        if self.verbose > 0:
            print("\n"+"#"*70)
            print(f"INFO: Started {taskname} at {start_time}")


        return

    def checkpoint_end(self,
        taskname:str, comment:str='',
        ) -> None:
        """
            - method to wrap up a task of name 'taskname'

            Parameters
            ----------
                - taskname
                    - str
                    - unique name given to the task to finish
                - comment
                    - str, optional
                    - comment to overwrite
                
            Raises
            ------

            Returns
            -------

            Comments
            --------
        """

        end_time = np.datetime64('now')
        try:
            start_time = self.df_protocoll[(self.df_protocoll["Task"]==taskname)]["Start"].values[0]
        except IndexError as ie:
            msg = (
                f"IndexError occured. Probably your provided 'taskname' has never been initialized."
                f"Make sure to initialize a 'taskname' before calling 'checkpoint_end()!\n"
                f"Original Error: {ie}."
            )
            raise LookupError(msg)

        duration = pd.to_timedelta(end_time-start_time)


        cur_task = np.where(self.df_protocoll["Task"]==taskname)[0][0]


        self.df_protocoll.at[cur_task, "End"] = end_time
        self.df_protocoll.at[cur_task, "Duration"] = duration
        self.df_protocoll.at[cur_task, "Comment_End"] = comment


        if self.verbose > 0:
            print(
                f"\n"
                f"INFO: Finished {taskname} at {end_time}\n"
                f"Required time: {pd.to_timedelta(self.df_protocoll.at[cur_task, 'Duration'])}"
            )
            print("#"*70)
        return

    def estimate_runtime(self,
        taskname_pat:str,
        nrepeats:int, ndone:int=1,
        ) -> None:
        """
            - method to estimate the total runtime in dependence of how many repetition have been made and will be made

            Parameters
            ----------
                - taskname_pat
                    - str
                    - regular expression to query the self.df_protocoll['Task']
                        - all tasks that contain 'taskname_pat' will contribute to the runtime-estimate
                - nrepeats
                    - int
                    - how often the task will be repeated
                - ndone
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

        print(f"INFO: Total estimated runtime for {nrepeats} repeats: {runtime_estimate}")        


        return
