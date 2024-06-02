
#TODO: rewrite using ADQL
#       https://astroquery.readthedocs.io/en/latest/simbad/simbad.html
#       see MasterThesis_ComputerScience (test.ipynb)

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
        - class to interact with the SIMBAD database

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
                - the default is 1      
            - `simbad_timeout`
                - `int`, optional
                - the timeout to allow for the SIMBAD database before an error is raised
                - the default is 120 (seconds)
            - `n_jobs`
                - `int`, optional
                - number of jobs to use for the query
                - will be passed to `joblib.parallel.Parallel()`
                - the default is -1
                    - will use all available cores
            - `verbose`
                - `int`, optional
                - verbosity level
                - the default is 0

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
        n_jobs:int=-1,
        simbad_timeout:int=120,
        verbose:int=0,
        ) -> None:

        self.npartitions    = npartitions
        self.simbad_timeout = simbad_timeout
        self.n_jobs         = n_jobs
        self.verbose        = verbose

        return

    def __repr__(self) -> str:
        return (
            f'{self.__class__.__name__}(\n'
            f'    npartitions={self.npartitions},\n'
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
        n_jobs:int=None,
        simbad_timeout:int=None,
        verbose:int=None,
        parallel_kwargs:dict=None,
        query_tap_kwargs:dict=None,
        ) -> pd.DataFrame:
        """
            - method to extract all IDs listed in SIMBAD for all targets in `input_ids`

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
                    - the table that has to be uploaded contains at least target that is cataloged in SIMBAD
                    - some working examples are
                        - `len(input_ids)==160000`, `npartitions=10`
                        - `len(input_ids)==320000`, `npartitions=20`
                        - `len(input_ids)==440000`, `npartitions=25`
                    
        """
        
        #default parameters
        if npartitions is None:             npartitions             = self.npartitions
        if n_jobs is None:                  n_jobs                  = self.n_jobs
        if simbad_timeout is None:          simbad_timeout          = self.simbad_timeout
        if verbose is None:                 verbose                 = self.verbose
        if parallel_kwargs is None:         parallel_kwargs         = dict(backend='threading')
        if 'backend' not in parallel_kwargs.keys():
            parallel_kwargs['backend'] = 'threading'
        if 'verbose' not in parallel_kwargs.keys():
            parallel_kwargs['verbose'] = verbose        
        if query_tap_kwargs is None:        query_tap_kwargs        = dict()

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
                msg=f'Working on parition {idx+1}/{npartitions}',
                context=self.get_ids.__name__,
                type='INFO',
                level=0,
                verbose=verbose,
            )

            #setup SIMBAD
            id_Simbad = Simbad()
            id_Simbad.TIMEOUT = simbad_timeout
            
            #define query
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

    def match_ids_(self,
        input_id:str,
        simbad_ids:str, main_id:str,
        ra:str, dec:str,
        show_scanned_string:bool=False,
        verbose:int=None
        ) -> dict:
        """
            - method

            Parameters
            ----------
             - `input_id`
                - str
                - id queried in SIMBAD
                    - i.e., id to get other ids of
             - `simbda_ids`
                - str
                - `'IDS'` column in a response of a SIMBAD query
                    - contains ids separated by `'|'`
             - `main_id`
                - str
                - main id returned in the response of a SIMBAD query
                    - in column `'MAIN_ID'`
             - `ra`
                - str
                - `'RA'` column in a response of a SIMBAD query
                - contains right ascension
             - `dec`
                - str
                - `'DEC'` column in a response of a SIMBAD query
                - contains declination
            - `show_scanned_string`
                - bool, optional
                - whether to display the string that got scanned with a regular expression to extract the different identifiers and catalogues
                - the default is False
             - `verbose`
                - int, optional
                - verbosity level
                - overrides `self.verbose`
                - the default is `None`
                    - will fall back to `self.verbose`

            Raises
            ------

            Returns
            -------
                - `res`
                    - dict
                    - processed response of the SIMBAD query
                    - keys are the
                        - input id
                        - main id
                        - ra
                        - dec
                        - catalogs
                    - values are the corresponding ids

            Comments
            --------
        """

        if verbose is None: verbose = self.verbose

        #some verbosity
        if show_scanned_string:
            almofo.printf(
                msg=f'Scanned Target: {input_id}',
                context=self.match_ids_.__name__,
                type='INFO',
                level=1,
                verbose=verbose
            )
            almofo.printf(
                msg=f'Query Result: {simbad_ids}',
                context=self.match_ids_.__name__,
                type='INFO',
                level=1,
                verbose=verbose
            )

        #extract individual ids from string returned by SIMBAD
        simbad_ids = re.findall(pattern=r'[^|]+', string=simbad_ids)

        #init results dict
        res = dict(
            input_id = input_id,
            main_id = main_id,
            ra=ra, dec=dec,
        )

        for id in simbad_ids:
            catalogue = re.match(r'^.+[^\ ](?=\ )|NPM\d|CSI', id)
            if catalogue is None:
                almofo.printf(
                    msg=f'Catalog is `None`. Corresponding id: `"{id}"`',
                    context=f'{self.match_ids_.__name__}',
                    type='INFO',
                    level=0,
                    verbose=verbose,
                )
            else:
                id_in_cat = id.replace(catalogue[0], '')
                res[catalogue[0]] = id_in_cat.strip()

        return res

    def extract_ids(self,
        input_ids:list,
        npartitions:int=None, simbad_timeout:int=None,
        show_scanned_strings_at:Union[list,Literal['all']]=None,
        verbose:int=None,
        parallel_kwargs:dict=None,
        ) -> pd.DataFrame:
        """
            - DEPRECATED. Use `self.get_ids()` instead
            - method to query the SIMBAD database for additional identifiers

            Parameters
            ----------
                - `input_ids`
                    - list
                    - list containing strings
                        - all the ids to query for
                    - each entry has to be of the same syntax as the SIMBAD query i.e.
                        - `'tic 114886928'`
                        - `'gaia dr3 6681944303315818624'`
                - `npartitions`
                    - int, optional
                    - into how many partitions to split the `input_ids` into
                    - useful for large queries
                    - overrides `self.npartitions`
                    - the default is `None`
                        - will fall back to `self.npartitions`
                - `simbad_timeout`
                    - int, optional
                    - the timeout to allow for the SIMBAD database before an error is raised
                    - overrides `self.simbad_timeout`
                    - the default is `None`
                        - will fall back to `self.simbad_timeout`
                - `show_scanned_strings_at`
                    - list, Literal['all'], optional
                    - list of indices to display the strings that get scanned with a regular expression to extract the different identifiers and catalogues
                    - if `'all'` will display scanned strings for all `input_ids`
                    - the default is `None`
                        - will be set to `[]`
                - `verbose`
                    - int, optional
                    - verbosity level
                    - overrides `self.verbose`
                    - the default is `None`
                        - will fall back to `self.verbose`
                - `parallel_kwargs`
                    - dict, optional
                    - kwargs to pass to `joblib.Parallel`
                    - the default is `None`
                        - will be set to `dict(backend='threading')`
            
            Raises
            ------

            Returns
            -------

            Comments
            --------
                - DEPRECATED. Use `self.get_ids()` instead

        """

        #default parameters
        if npartitions is None:             npartitions             = self.npartitions
        if simbad_timeout is None:          simbad_timeout          = self.simbad_timeout
        if show_scanned_strings_at is None: show_scanned_strings_at = []
        if verbose is None:                 verbose                 = self.verbose
        if parallel_kwargs is None:         parallel_kwargs         = dict(backend='threading')
        if 'backend' not in parallel_kwargs.keys():
            parallel_kwargs['backend'] = 'threading'
        if 'verbose' not in parallel_kwargs.keys():
            parallel_kwargs['verbose'] = verbose
        
        almofo.printf(
            msg=f'DEPRECATED! Use `SimbadDatabaseInterface().get_ids()` instead\n\n',
            context=self.extract_ids.__name__,
            type='WARNING',
            level=0,
            verbose=verbose,
        )



        unique_ids = np.unique(input_ids)

        #setup SIMBAD
        my_Simbad = Simbad()
        my_Simbad.TIMEOUT = simbad_timeout
        my_Simbad.add_votable_fields('ids')
        my_Simbad.add_votable_fields('typed_id')
        # print(my_Simbad.list_votable_fields())

        #create boolean to decide which scanned strings to display
        if show_scanned_strings_at == 'all':
            show_scanned_strings_bool = np.ones(len(unique_ids))
        else:
            show_scanned_strings_bool = np.zeros(len(unique_ids))
            show_scanned_strings_bool[show_scanned_strings_at] = 1
        
        #split the query into chuncks for big queries
        ids_partitioned = np.array_split(unique_ids, npartitions)
        show_scanned_strings_bool = np.array_split(show_scanned_strings_bool, npartitions)
        
        result = []
        for idx, (ids, show_scanned_strings) in enumerate(zip(ids_partitioned, show_scanned_strings_bool)):
            
            if verbose > 0:
                almofo.printf(
                    msg=f'Working on partition {idx+1}/{len(ids_partitioned)}',
                    context=self.extract_ids.__name__,
                    level=0,
                    verbose=verbose,
                )

            #query SIMBAD
            id_table = my_Simbad.query_objects(ids)

            # print(id_table.keys())
            main_ids    = id_table['MAIN_ID']
            ras         = id_table['RA']
            decs        = id_table['DEC']


            res = Parallel(**parallel_kwargs)(
                delayed(self.match_ids_)(
                    input_id=input_id,
                    simbad_ids=simbad_ids, main_id=main_id,
                    ra=ra, dec=dec,
                    show_scanned_string=show_scanned_string,
                    verbose=verbose,
                ) for input_id, simbad_ids, main_id, ra, dec, show_scanned_string in zip(ids, id_table['IDS'], main_ids, ras, decs, show_scanned_strings)
            )

            #append to output result
            result += res

        df_ids = pd.DataFrame.from_dict(result)
            
        return df_ids

    # def get_ids_coords(self,
    #     input_coords=[],
    #     nparallelrequests=1000, simbad_timeout=120,
    #     show_scanned_strings_at=[],
    #     verbose=0):

    #     import numpy as np
    #     import pandas as pd
    #     import re

    #     from astroquery.simbad import Simbad
    #     from astropy.coordinates import SkyCoord
    #     from joblib import Parallel, delayed

    #     import warnings

    #     warnings.filterwarnings("ignore", message=r"Coordinate string is being interpreted")

    #     #setup SIMBAD
    #     my_Simbad = Simbad()
    #     my_Simbad.TIMEOUT = simbad_timeout
    #     my_Simbad.add_votable_fields("ids")
    #     my_Simbad.add_votable_fields("typed_id")
    #     # print(my_Simbad.list_votable_fields())

    #     #split the query into chuncks for big queries
    #     intervals = np.arange(0, input_coords.shape[0], nparallelrequests)
    #     if intervals[-1] < input_coords.shape[0]: intervals = np.append(intervals, input_coords.shape[0])
        
    #     for idx, (start, end) in enumerate(zip(intervals[:-1], intervals[1:])):
            
    #         if verbose > 0:
    #             print(f"Working on partition {idx+1}/{len(intervals)-1}")

    #         cur_coords = input_coords[start:end]

    #         for cuc in cur_coords[:]:
    #             id_table = my_Simbad.query_region(cuc, radius=".1s")


    #             #extract IDs out of the query result
    #             if id_table is not None:
                    
    #                 # print(id_table.keys())
    #                 main_ids = id_table["MAIN_ID"]
    #                 ras = id_table["RA"]
    #                 decs = id_table["DEC"]
                    
    #                 simbad_result = Parallel(n_jobs=3)(delayed(re.findall)(pattern=r"[^|]+", string=str(id)) for id in id_table["IDS"])

    #                 print(main_ids, simbad_result)




    #     return

    #VIZIER
    # def query_objects_vizier(self,
    #     ra_list, dec_list,
    #     catalog=["*"], columns=["*"], timeout=60, row_limit=-1
    #     ):
    #     """
        
    #     """
    #     from astroquery.vizier import Vizier
    #     from astropy.coordinates import SkyCoord
    #     from astropy.table import Table

    #     v = Vizier(
    #         catalog=catalog, columns=columns,
    #         timeout=timeout,
    #         row_limit=row_limit
    #     )

    #     for ra, dec in zip(ra_list, dec_list):
    #         print(ra, dec)
    #         result = v.query_object(f"{ra} {dec}")
    #         print(result)

    #     return

