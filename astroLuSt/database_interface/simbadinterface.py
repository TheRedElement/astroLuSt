
#%%imports
from astropy.table import Table
from astroquery.simbad import Simbad, SimbadClass
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
import re
from typing import Union, Literal, List

from astroLuSt.monitoring import formatting as almofo

#%%classes
#SIMBAD
class SimbadDatabaseInterface:
    """
        - class to interact with the [SIMBAD](https://simbad.u-strasbg.fr/simbad/) database

        Attributes
        ----------
            - `npartitions`
                - `int`, optional
                - how many partitions to split the `input_ids` into
                - useful for large queries
                    - maximum number of rows to upload is 200000
                - some working examples are
                    - `len(input_ids)==160000`, `npartitions=10`
                    - `len(input_ids)==320000`, `npartitions=20`
                    - `len(input_ids)==440000`, `npartitions=25`                
                - the default is `1`
            - `nperpartition`
                - `int`, optional
                - number of samples (ids) to extract per partition
                - overrides `self.npartition`
                - the default is `None`
            - `simbad_timeout`
                - `int`, optional
                - the timeout to allow for the SIMBAD database before an error is raised
                - the default is `120` (seconds)
            - `n_jobs`
                - `int`, optional
                - number of jobs to use for the query
                - will be passed to `joblib.parallel.Parallel()`
                - the default is `1`
            - `verbose`
                - `int`, optional
                - verbosity level
                - the default is `0`

        Methods
        -------
            - `get_ids()`
            - `extract_ids()`
                - deprecated
            - `match_ids_()`
                - deprecated
            
        Dependencies
        ------------
            - `astropy`
            - `astroquery`
            - `joblib`
            - `numpy`
            - `pandas`
            - `re`
            - `typing`

        Comments
        --------
    """
       
    def __init__(self,
        npartitions:int=1,
        nperpartition:int=None,
        n_jobs:int=1,
        simbad_timeout:int=120,
        verbose:int=0,
        ) -> None:

        self.npartitions    = npartitions
        self.nperpartition  = nperpartition
        self.simbad_timeout = simbad_timeout
        self.n_jobs         = n_jobs
        self.verbose        = verbose

        return

    def __repr__(self) -> str:
        return (
            f'{self.__class__.__name__}(\n'
            f'    npartitions={self.npartitions},\n'
            f'    nperpartition={self.nperpartition},\n'
            f'    n_jobs={self.n_jobs},\n'
            f'    simbad_timeout={self.simbad_timeout},\n'
            f'    verbose={self.verbose},\n'
            f')'
        )
        
    def __dict__(self) -> dict:
        return eval(str(self).replace(self.__class__.__name__, 'dict'))

    def get_ids(self,
        input_ids:List[str],
        npartitions:int=None,
        nperpartition:int=None,
        n_jobs:int=None,
        simbad_timeout:int=None,
        verbose:int=None,
        parallel_kwargs:dict=None,
        query_tap_kwargs:dict=None,
        ) -> pd.DataFrame:
        """
            - method to extract all IDs listed in SIMBAD for all objects in `input_ids`

            Parameters
            ----------
                - `input_ids`
                    - `list`
                    - list containing strings
                        - all the ids to query for
                    - each entry has follow the syntax as defined by SIMBAD i.e.
                        - `'tic 114886928'`
                        - `'gaia dr3 6681944303315818624'`
                - `npartitions`
                    - `int`, optional
                    - how many partitions to split the `input_ids` into
                    - useful for large queries
                        - maximum number of rows to upload is 200000
                    - some working examples are
                        - `len(input_ids)==160000`, `npartitions=10`
                        - `len(input_ids)==320000`, `npartitions=20`
                        - `len(input_ids)==440000`, `npartitions=25`                
                    - overrides `self.npartitions`
                    - the default is `None`
                        - will fall back to `self.npartitions`
                - `nperpartition`
                    - `int`, optional
                    - number of samples (ids) to extract per partition
                    - overrides `self.nperpartition`
                    - if set
                        - will be used instead of `self.npartition` to generate partitions
                    - the default is `None`
                        - will fall back to `self.nperpartition`               
                - `n_jobs`
                    - `int`, optional
                    - number of cores to use for parallel execution of partitions
                    - overrides `self.n_jobs`
                    - the default is `None`
                        - will fall back to `self.n_jobs`
                - `simbad_timeout`
                    - `int`, optional
                    - the timeout to allow for the SIMBAD database before an error is raised
                    - overrides `self.simbad_timeout`
                    - the default is `None`
                        - will fall back to `self.simbad_timeout`
                - `verbose`
                    - `int`, optional
                    - verbosity level
                    - overrides `self.verbose`
                    - the default is `None`
                        - will fall back to `self.verbose`
                - `parallel_kwargs`
                    - `dict`, optional
                    - kwargs to pass to `joblib.Parallel`
                    - the default is `None`
                        - will be set to `dict(backend='threading')`
                - `quey_tap_kwargs`
                    - `dict`, optional
                    - kwargs to pass to `Simbad().query_tap()`
                    - the default is `None`
                        - will be set to `dict()`

            Raises
            ------

            Returns
            -------
                - `df_res`
                    - `pandas.DataFrame`
                    - dataframe containing various ids alongside coordinates

            Comments
            --------
                - it is important to tune `npartitions` in a way that
                    - the table that has to be uploaded is not too large
                    - the table that has to be uploaded contains at least one target that is cataloged in SIMBAD
                    - some working examples are
                        - `len(input_ids)==160000`, `npartitions=10`
                        - `len(input_ids)==320000`, `npartitions=20`
                        - `len(input_ids)==440000`, `npartitions=25`
                    
        """
        
        #default parameters
        if npartitions is None:             npartitions             = self.npartitions
        if nperpartition is None:           nperpartition           = self.nperpartition
        if n_jobs is None:                  n_jobs                  = self.n_jobs
        if simbad_timeout is None:          simbad_timeout          = self.simbad_timeout
        if verbose is None:                 verbose                 = self.verbose
        if parallel_kwargs is None:         parallel_kwargs         = dict(backend='threading')
        if 'backend' not in parallel_kwargs.keys():
            parallel_kwargs['backend'] = 'threading'
        if 'verbose' not in parallel_kwargs.keys():
            parallel_kwargs['verbose'] = verbose        
        if query_tap_kwargs is None:        query_tap_kwargs        = dict()

        #override `npartitions` is specified
        if nperpartition is not None:
            npartitions = int(np.ceil(len(input_ids)/nperpartition))

        def get4partition(
            ids:List[str],
            simbad_timeout:float=None,
            idx:int=0,
            verbose:int=None,
            query_tap_kwargs:dict=None,
            ) -> pd.DataFrame:
            """
                - subfunction to extract IDs for one partition
            """

            almofo.printf(
                msg=f'Working on parition {idx+1}/{npartitions} ({len(ids):.0f} samples)',
                context=self.get_ids.__name__,
                type='INFO',
                level=0,
                verbose=verbose,
            )

            #setup SIMBAD
            id_Simbad = Simbad()
            id_Simbad.TIMEOUT = simbad_timeout
            
            #define query (reference: https://astroquery.readthedocs.io/en/latest/simbad/simbad.html)
            q = f"""
                SELECT
                    intable.id AS input_id,
                    ident.id AS main_id,
                    ids.ids,
                    basic.ra, basic.dec
                FROM TAP_UPLOAD.intable
                    LEFT JOIN ident ON ident.id = intable.id
                    LEFT JOIN ids ON ids.oidref = ident.oidref
                    LEFT JOIN basic ON basic.oid = ident.oidref
            """

            #convert input ids to Table (for upload)
            tab = Table(data=dict(id=ids))
            
            #execute query
            res_p = id_Simbad.query_tap(query=q, intable=tab, **query_tap_kwargs).to_pandas()

            return res_p
        

        #partition list of ids in case of large queries
        ids_partitioned = np.array_split(input_ids, npartitions)

        #query partitions in parallel
        res = Parallel(n_jobs=n_jobs, **parallel_kwargs)(
            delayed(get4partition)(
                ids=list(ids_p),
                idx=idx,
                verbose=verbose,
                query_tap_kwargs=query_tap_kwargs,
            ) for idx, ids_p in enumerate(ids_partitioned)
        )

        #merge result into one dataframe
        df_res = pd.concat(res, axis=0).reset_index().drop('index', axis=1, inplace=False)

        return df_res
