
#%%imports
from astroquery.gaia import GaiaClass
import numpy as np
import pandas as pd
import re
from typing import List, Union, Any, Dict, Callable

from astroLuSt.monitoring import formatting as almf

#%%definitions
#GAIA
class GaiaDatabaseInterface:
    """
        - class to interact with the Gaia-archive

        Attributes
        ----------
            - `gaia_class`
                - `astroquery.gaia.GaiaClass`
                - instance of `astroquery.gaia.Gaia`
                - instance to use for quering the Gaia database
            - `verbose`
                - `int`, optional
                - verbosity level
                - the default is `0`
                
        
        Methods
        -------
            - `remove_all_jobs()`
            - `get_datalink()`
            - `save()`

        Dependencies
        ------------
            - `astroquery`
            - `numpy`
            - `pandas`
            - `re`
            - `typing`

        Comments
        --------

    """

    def __init__(self,
        gaia_class:GaiaClass,
        verbose:int=0,
        ) -> None :

        self.Gaia = gaia_class
        self.verbose = verbose
        
        return

    def __repr__(self) -> str:
        return (
            f'{self.__class__.__name__}(\n'
            f"    gaia_class={self.Gaia},\n"
            f"    verbose={self.verbose},\n"
            f')'
        )

    def __dict__(self) -> dict:
        return eval(str(self).replace(self.__class__.__name__, 'dict'))

    def remove_all_jobs(self,
        pd_filter:str=None,
        verbose:int=0
        ) -> None:
        """
            - method to remove all jobs stored in the Gaia-Archive
            - `self.Gaia` has to be an instance where you are logged in for this method to work
            
            Parameters
            ----------
                - `pd_filter`
                    - `str`, optional
                    - string that will be evaluated as a boolean filter
                        - serves to filter all jobs for some specific ones
                    - must have the following structure
                        - `"(jobs['colname1'] == 'argument1')&(jobs['colname2'] != 'argument2')"` ect.
                        - `'colname1'` and `'colname2'` are hereby substituted by one of the following
                            - `'jobid'`
                            - `'failed'`
                            - `'creationDate'`
                            - `'creationTime'`
                            - `'endDate'`
                            - `'endTime'`
                            - `'startDate'`
                            - `'startTime'`
                            - `'responseStatus'`
                            - `'phase'`
                        - `'argument1'` and `'argument2'` are substituted by some value for the boolean expression
                        - some useful examples
                            - removing all jobs created at the present day
                                
                                ```python
                                >>> today = pd.to_datetime('today').date()
                                >>> pd_filter = f"(jobs['creationDate'] == '{today}')"
                                >>> DI.remove_all_jobs(pd_filter=pd_filter, verbose=0)
                                ```
                                
                    - the default is `None`
                        - will delete all jobs in the archive
                - `verbose`
                    - `int`, optional
                    - how much additional information to display
                    - the default is `0`

            Raises
            ------
                - `ValueError`
                    - if the user is not logged in correctly

            Returns
            -------

            Comments
            --------
                - `self.Gaia` has to be an instance where you are logged in for this method to work
        """

        
        #get all jobs
        all_jobs = self.Gaia.search_async_jobs(verbose=False)
        
        jobs = pd.DataFrame({
            "jobid":[job.jobid for job in all_jobs],
            "name":[job.name for job in all_jobs],
            "failed":[job.failed for job in all_jobs],
            "creationTime":[job.creationTime for job in all_jobs],
            "endTime":[job.endTime for job in all_jobs],
            "startTime":[job.startTime for job in all_jobs],
            "responseStatus":[job.responseStatus for job in all_jobs],
            "phase":[job.get_phase() for job in all_jobs],
        })
        
        jobs[["creationDate", "creationTime"]] = jobs["creationTime"].str.split("T", expand=True)
        jobs[["startDate", "startTime"]] = jobs["startTime"].str.split("T", expand=True)
        jobs[["endDate", "endTime"]] = jobs["endTime"].str.split("T", expand=True)
        jobs["creationTime"] = jobs["creationTime"].str.slice(0,-1)
        jobs["startTime"] = jobs["startTime"].str.slice(0,-1)
        jobs["endTime"] = jobs["endTime"].str.slice(0,-1)

        jobs.sort_index(axis=1, inplace=True)
        
        if verbose > 0:
            print("INFO: The following jobs have been found:")
            print(jobs)

        #apply filter
        if pd_filter is not None:
            to_remove = jobs[eval(pd_filter)]["jobid"].to_list()
        else:
            to_remove = jobs["jobid"].to_list()

        #remove jobs
        self.Gaia.remove_jobs(to_remove)
        
        return

    def get_datalink(self,
        ids:Union[str,List[str]],
        retrieval_type:List[str]=None,
        get_normalized_flux:bool=True, normfunc:Callable=None,
        n_chunks:int=1,
        verbose:int=None,
        load_data_kwargs:dict=None,
        save_kwargs:dict=None,
        ) -> Dict[str,pd.DataFrame]:
        """
            - method to obtain the datalink data from a gaia archive
            - i.e.
                - photometry
                - rvs
            - resource:
                - https://www.cosmos.esa.int/web/gaia-users/archive/datalink-products
                - last accessed 2024/02/26
            
            Parameters
            ----------
                - `ids`
                    - `str`, `List`
                    - gaia source ids to download the data for
                - `retrieval_type`
                    - `List[str]`, optional
                    - type of data to be downloaded
                    - will be passed to `Gaia.load_data()`
                    - the default is `None`
                        - will be set to `['ALL']`
                - `get_normalized_flux`
                    - bool, optional
                    - whether to also extract the (passband-wise) normalized versions of the extracted fluxes
                    - the default is `True`
                - `normfunc`
                    - `Callable`, optional
                    - function to execute the normalization
                    - has to take exactly one two arguments
                        - `flux`
                        - `df`
                            - `pandas.DataFrame()` object
                            - contains all extracted quantities
                            - can be used to acess quality flags etc.
                    - the default is `None`
                        - will be set to `lambda x, df: x/np.nanmedian(x)`
                - `n_chunks`
                    - `int`, optional
                    - number of chunks to split `ids` into
                    - use if too many targets to extract, i.e. request refuses to process
                    - the default is `1`
                        - all `ids` at once
                - `verbose`
                    - `int`, optional
                    - verbosity level
                    - overrides `self.verbose`
                    - the default is `None`
                        - falls back to `self.verbose`
                - `load_data_kwargs`
                    - `dict`, optional
                    - kwargs to pass to `Gaia.load_data()`
                    - the default is `None`
                        - will be set to `dict()`
                - `save_kwargs`
                    - `dict`, optional
                    - kwargs to pass to `self.save()`
                    - the default is `None`
                        - will be set to `dict(directory=None)`
                        - data will not be saved
            
            Raises
            ------

            Returns
            -------
                - `results`
                    - `Dict[str,pd.DataFrame]`
                    - dict of pandas DataFrames
                    - keys are the retrieved quantity alongside the gaia id
                    - values contain the extracted data
                        

            Comments
            --------
        """

        if retrieval_type is None:      retrieval_type      = ['ALL']
        if normfunc is None:            normfunc            = lambda x, df: x/np.nanmedian(x)
        if verbose is None:             verbose             = self.verbose
        if load_data_kwargs is None:    load_data_kwargs    = dict()
        if save_kwargs is None:         save_kwargs_use     = dict(directory=None)
        else:                           save_kwargs_use     = save_kwargs.copy()

        #split into n_chuncks chuncks, for huge number of targets
        chunks = np.array_split(ids, n_chunks)

        #init dict of results
        result = dict()
        
        #iterate over requested dataproducts
        for ridx, rt in enumerate(retrieval_type):
            
            almf.printf(
                msg=f'Extracting retrieval_type {ridx+1}/{len(retrieval_type)} ({rt})',
                context=f'{self.__class__.__name__}.{self.get_datalink.__name__}()',
                type='INFO',
                level=0,
                verbose=verbose
            )

            #iterate over chuncks
            extracted = 0
            for cidx, chunk in enumerate(chunks):
                #update number of extracted targets
                extracted += len(chunk)

                almf.printf(
                    msg=f'Extracting chunk {cidx+1}/{len(chunks)} ({extracted}/{len(ids)})',
                    context=f'{self.__class__.__name__}.{self.get_datalink.__name__}()',
                    type='INFO',
                    level=1,
                    verbose=verbose,
                )

                #obtain data from gaia archive
                datalink = self.Gaia.load_data(ids=chunk, retrieval_type=rt, **load_data_kwargs)

                #check what got extracted
                keys = datalink.keys()
                
                #store extracted quantities accordingly
                for k in keys:
                    dl_id = re.findall('(?<=Gaia DR3 )\d+', k)
                    dl_prod = re.findall('^\w+(?=-)', k)

                    for idx, dl in enumerate(datalink[k]):
                        df = dl.to_table().to_pandas()
                    

                    if 'EPOCH_PHOTOMETRY' in k and get_normalized_flux:
                        # df['flux_normalized'] = 0.
                        for b in np.unique(df['band']):
                            df.loc[(df['band']==b),'flux_normalized'] = normfunc(df.query('band==@b')['flux'], df)
                    result[k] = df
                    
                    #save if wished for
                    if isinstance(save_kwargs_use['directory'], str):
                        self.save(
                            df=df,
                            filename=f'gaia{dl_id[idx]}_{dl_prod[idx].lower()}',
                            **save_kwargs,
                        )
                    
        return result

    def save(self,
        df:pd.DataFrame,
        filename:str,
        directory:str=None,
        pd_savefunc:str=None,
        save_kwargs:dict=None,
        ) -> None:
        """
            - method to save the extracted data

            Parameters
            ----------
                - `df`
                    - `pd.DataFrame`
                    - dataframe of extracted lc-data (`lcs` from `self.extract_source()`)
                - `filename`
                    - `str`
                    - name of the file in which the data gets stored
                    - NO FILE EXTENSION!
                - `directory`
                    - `str`, optional
                    - directory of where the data will be stored
                    - the default is `None`
                        - will be set to `'./'`
                - `pd_savefunc`
                    - `str`, optional
                    - pandas saving function to use
                        - i.e., methods of pd.DataFrames
                        - examples
                            - `'to_csv'`
                            - `'to_parquet'`
                    - the default is `None`
                        - will be set to `'to_parquet'`
                - `save_kwargs`
                    - `dict`, optional
                    - kwargs to pass to `pd_savefunc`
                    - the default is `None`
                        - will be set to `dict()`
            Raises
            ------

            Returns
            -------

            Comments
            --------
                - metadata will be saved to a separate file with the same name except `'_meta'` inserted before the extension
        """

        if directory is None:   directory   = './'
        if pd_savefunc is None: pd_savefunc = 'to_parquet'
        if save_kwargs is None: save_kwargs = dict()

        #get correct extension
        if pd_savefunc == 'to_hdf': ext = 'h5'
        elif pd_savefunc == 'to_excel': ext = 'xlsx'
        elif pd_savefunc == 'to_stata': ext = 'dta'
        elif pd_savefunc == 'to_markdown': ext = 'md'
        else: ext = pd_savefunc[3:]

        #save
        eval(f'df.{pd_savefunc}("{directory}{filename}.{ext}", **{save_kwargs})')

        return
    
    