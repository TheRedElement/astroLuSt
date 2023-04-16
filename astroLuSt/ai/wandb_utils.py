
#%%imports
import os
import wandb

from joblib import Parallel, delayed

from astroLuSt.monitoring.timers import ExecTimer

#%%definitions
class WandB_parallel_sweep:
    """
        - class to enable parallelized hyperparameter sweep with the Weights And Biases (wandb) API

        Attributes
        ----------
            - sweep_id
                - str
                - unique id each sweep gets assigned
                - either directly pass the string or use the following command to generate a sweep id and initialize a new sweep
                    >>> sweep_id = wandb.sweep(sweep_config, entity="<youruser>", project="<yourproject>")
            - function
                - callable
                - function to be executed during the sweep for each combination of hyperparameters
                - i.e. the training loop
                - contains
                    - definition of your model
                    - compilation of your model
                    - fitting of your model
                    - monitoring and logging of metrics ect.
            - n_jobs
                - int, optional
                - how many jobs to use in parallel when running the sweep
                - argument of joblib.Parallel
                - the default is -1
                    - will use all available workers
            - n_agents
                - int, optional
                - how many agents to use for the sweep
                - this is the amount of training loops for different parameters executed in parallel
                - the default is 1
            - wandb_mode
                - str, optional
                - if not None will set the environment variable 'WANDB_MODE' to 'offline
                    - i.e. no synching to the web-api will take place
                    - useful for i.e. SLURM execution when there is no internet connection
                    - you can always synch your results later with the following command
                        >>> wand sync <path to run directory>
                    - usually the path to the run directory is ./wandb/offline
                    - use wildcards to sync multiple runs at once
                - the default is None
                    - will sync during sweep
            - verbose
                - int, optional
                - verbosity level
                - the default is 0 

        Methods
        -------
            - sweep_one()
            - sweep_parallel()

        Dependencies
        -------------
            - joblib
            - os
            - wandb

        Comments
        --------
    """
    
    def __init__(self,
        sweep_id:str, function:callable,
        n_jobs:int=-1, n_agents:int=1,
        wandb_mode:str=None,
        verbose:int=0
        ) -> None:
        

        self.sweep_id = sweep_id
        self.function = function
        self.n_jobs = n_jobs
        self.n_agents = n_agents
        self.wandb_mode = wandb_mode
        self.verbose = verbose

        self.ET = ExecTimer(verbose=self.verbose)
        
        
        if isinstance(wandb_mode, str):
            os.environ['WANDB_MODE'] = 'offline'  #run wandb offline and sync lateron
       
        pass

    def __repr__(self) -> str:
        
        return (
            f'WandB_parallel_sweep(\n'
            f'    sweep_id={self.sweep_id}, function={self.function},\n'
            f'    n_jobs={self.n_jobs}, n_agents={self.n_agents},\n'
            f'    wandb_mode={self.wandb_mode},\n'
            f'    verbose={self.verbose},\n'
            f')'
        )

    def sweep_one(self,
        sweep_id:str=None, function:callable=None,
        idx:int=0, n_agents:int=1,
        verbose:int=None,
        ) -> None:
        """
            - method to initialize a sweep using one particular agent

            Parameters
            ----------
                - sweep_id
                    - str, optional
                    - unique id each sweep gets assigned
                    - either directly pass the string or use the following command to generate a sweep id and initialize a new sweep
                        >>> sweep_id = wandb.sweep(sweep_config, entity="<youruser>", project="<yourproject>")
                    - overwrites self.sweep_id if passed
                    - the default is None
                - function
                    - callable, optional
                    - function to be executed during the sweep for each combination of hyperparameters
                    - i.e. the training loop
                    - contains
                        - definition of your model
                        - compilation of your model
                        - fitting of your model
                        - monitoring and logging of metrics ect.
                    - overwrites self.function if passed
                    - the default is None
                - idx
                    - int, optional
                    - index of the sweep currently executed
                    - only needed for verbosity
                    - the default is 0
                - n_agents
                    - int, optional
                    - how many agents to use for the sweep
                    - this is the amount of training loops for different parameters executed in parallel
                    - in this method only needed for verbosity
                    - the default is 1
                - verbose
                    - int, optional
                    - verbosity level
                    - overwrites self.verbose if passed
                    - the default is None 
                    
            Raises
            ------

            Returns
            -------

            Comments
            --------
                - self.ET will only show its output when the script is NOT executed from an interactive window
                - self.ET will only be able to store the timings in self.ET.df_execprotocoll if n_jobs == 1
        """

        self.ET.checkpoint_start(f'sweep_one, agent {idx}')

        #fallback to default if nothing is provided
        if sweep_id is None: sweep_id = self.sweep_id
        if function is None: function = self.function
        if verbose is None: verbose = self.verbose
        
        if verbose > 0:
            print(f'INFO(sweep_one): Sweeping with agent {idx+1}/{n_agents}')

        wandb.agent(sweep_id=sweep_id, function=function)

        self.ET.checkpoint_end(f'sweep_one, agent {idx}')

        return

    def sweep_parallel(self,
        sweep_id:str=None, function:callable=None,
        n_jobs:int=None, n_agents:int=None,
        verbose:int=None
        ) -> None:
        """
            - method to instantiate multiple agents working on one particular sweep
            - allows parallel execution of sweep

            Parameters
            ----------
                - sweep_id
                    - str, optional
                    - unique id each sweep gets assigned
                    - either directly pass the string or use the following command to generate a sweep id and initialize a new sweep
                        >>> sweep_id = wandb.sweep(sweep_config, entity="<youruser>", project="<yourproject>")
                    - overwrites self.sweep_id if passed
                    - the default is None
                - function
                    - callable, optional
                    - function to be executed during the sweep for each combination of hyperparameters
                    - i.e. the training loop
                    - contains
                        - definition of your model
                        - compilation of your model
                        - fitting of your model
                        - monitoring and logging of metrics ect.
                    - overwrites self.function if passed
                    - the default is None
                - n_jobs
                    - int, optional
                    - how many jobs to use in parallel when running the sweep
                    - argument of joblib.Parallel
                    - overwrites self.n_jobs if passed
                    - the default is None
                - n_agents
                    - int, optional
                    - how many agents to use for the sweep
                    - this is the amount of training loops for different parameters executed in parallel
                    - overwrites self.n_agents if passed
                    - the default is None
                - verbose
                    - int, optional
                    - verbosity level
                    - overwrites self.verbose if passed
                    - the default is None                     
                

            Raises
            ------

            Returns
            -------

            Comments
            --------

        """

        #fallback to default if nothing is provided
        if sweep_id is None: sweep_id = self.sweep_id
        if function is None: function = self.function
        if n_jobs is None: n_jobs = self.n_jobs
        if n_agents is None: n_agents = self.n_agents
        if verbose is None: verbose = self.verbose


        self.ET.checkpoint_start('sweep_parallel')
        #execute paralellized hyperparameter sweep
        res = Parallel(n_jobs=n_jobs, verbose=verbose)(
            delayed(self.sweep_one)(
                sweep_id=sweep_id, function=function,
                idx=idx, n_agents=n_agents,
                verbose=verbose,
            ) for idx in range(n_agents)
        )

        self.ET.checkpoint_end('sweep_parallel')



        return

