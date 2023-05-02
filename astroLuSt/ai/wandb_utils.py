
#%%imports
import itertools
import os
from typing import Union, Tuple, Callable
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
                - if n_agents < 0
                    - n_jobs - n_agents agents
                - the default is None
                    - will use as many agents as jobs (i.e. self.n_agents = self.n_jobs)
                        - a maximum of 20 agents will be used since 20 is the maximum allowed number of wokring agents)
            - wandb_mode
                - str, optional
                - will set the environment variable 'WANDB_MODE' accordingly
                    - 'offline'
                        - i.e. no synching to the web-api will take place
                        - useful for i.e. SLURM execution when there is no internet connection
                        - you can always synch your results later with the following command
                            >>> wand sync <path to run directory>
                        - usually the path to the run directory is ./wandb/offline...
                        - use wildcards to sync multiple runs at once
                    - online
                        - synching during sweep enabled
                - the default is 'online'
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
        sweep_id:str, function:Callable,
        n_jobs:int=-1, n_agents:int=None,
        wandb_mode:str=None,
        verbose:int=0
        ) -> None:
        

        self.sweep_id = sweep_id
        self.function = function
        self.n_jobs = n_jobs
        if n_agents is None:
            self.n_agents = n_jobs
        self.wandb_mode = wandb_mode
        self.verbose = verbose

        self.ET = ExecTimer(verbose=self.verbose)
        
        
        os.environ['WANDB_MODE'] = str(self.wandb_mode)  #run wandb on/offline
       
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

    def get_upper_bound_agents(self,
        sweep_config:dict
        ) -> int:
        """
            - method to estimate an upper bound of the number of agents needed
            - will do so by determining the amount of model-instances that will be computed
            
            Parameters
            ----------
                - sweep_config
                    - dict
                    - nested dict
                    - sweep configuration that gets passed to wandb.sweep()

            Raises
            ------

            Returns
            -------
                - n_comb
                    - int
                    - number of hyperparameter combinations that will be computed
                    - equivalent to the upper bound estimate of n_agents
                    
            Comments
            --------
        """
        

        #grid search
        if sweep_config['method'] == 'grid':
            #extract hyperparameters from sweep_config
            params = sweep_config['parameters']

            values = [p['values'] for p in params.values() if 'values' in p.keys()]
            # value = [[p['value']] for p in params.values() if 'value' in p.keys()]
            # values += value

            #get number of combinations resulting from hyperparameters
            n_combs = len(list(itertools.product(*values)))


        #random search (sampling from distributions)
        elif sweep_config['method'] == 'random':
            n_combs = sweep_config['run_cap']
            # distributions = [p['distribution'] for p in params.values() if 'distribution' in p.keys()]
        
        #bayesian search (sampling from distribution)
        elif sweep_config['method'] == 'bayes':
            n_combs = sweep_config['run_cap']
            # distributions = [p['distribution'] for p in params.values() if 'distribution' in p.keys()]

        if n_combs > 20:
            if self.verbose > 1:
                print('INFO(get_upper_bound_agents): The maximum agents working in parallel allowed are 20. Thus returning n_combs=20.')
            n_combs = 20

        return n_combs

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

        self.ET.checkpoint_start(f'sweep_one, agent {idx+1}')

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
        sweep_id:str=None, function:Callable=None,
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

        if n_jobs > 20:
            if self.verbose > 1:
                print('INFO(sweep_parallel): The maximum agents working in parallel allowed are 20. Setting n_jobs = 20.')
            n_jobs = 20
        elif n_jobs < 0:
            n_jobs = os.cpu_count()+(n_jobs+1)


        if n_agents > 20:
            if self.verbose > 1:
                print('INFO(sweep_parallel): The maximum agents working in parallel allowed are 20. Setting n_agents = 20.')
            n_agents = 20
        elif n_agents < 0:
            n_agents = n_jobs+n_agents+1
            pass
        
        if verbose > 1:
            print(f'INFO(sweep_parallel()): Using n_jobs={n_jobs} jobs and n_agents={n_agents} agents to run sweep.')

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

