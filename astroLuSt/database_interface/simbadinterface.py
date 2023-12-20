
#%%imports
from astroquery.simbad import Simbad
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
import re

#SIMBAD
class SimbadDatabaseInterface:
    """
        - class to interact with the SIMBAD database

        Attributes
        ----------

        Infered Attributes
        ------------------
            - `df_ids`
                - pd.DataFrame
                - contains all IDs listed in SIMBAD for the queried objects (`input_ids`)

        Methods
        -------
            - `get_ids()`

        Dependencies
        ------------
            - numpy
            - pandas
            - re
            - astroquery
            - joblib

        Comments
        --------
    """
    

    
    def __init__(self,
        ) -> None:

        self.df_ids = pd.DataFrame()

        return

    def __repr__(self) -> str:
        return (
            f'SimbadDatabaseInterface(\n'
            f')'
        )
        

    def get_ids(self,
        input_ids:list,
        nparallelrequests:int=1000, simbad_timeout:int=120,
        show_scanned_strings_at:list=[],
        verbose:int=0
        ) -> None:
        """
            - method to query the SIMBAD database for additional identifiers

            Parameters
            ----------
                - `input_ids`
                    - list
                    - list containing all the ids to query for
                    - each entry has to be of the same syntax as the SIMBAD query i.e.
                        - `'tic 114886928'`
                        - `'gaia dr3 6681944303315818624'`
                - `nparallelrequests`
                    - int, optional
                    - how many requests to run in parallel
                        - i.e. if your DataFrame has N columns the query will be subdivided into n partitions such that `nparallelrequest` queries will be executed in parallel
                    - useful for large queries
                    - the default is 1000
                - `simbad_timeout`
                    - int, optional
                    - the timeout to allow for the SIMBAD database before an error is raised
                    - the default is 120 (seconds)
                - `show_scanned_strings_at`
                    - list
                    - list of indices to display the strings that get scanned with a regular expression to extract the different identifiers and catalogues
                    - the default is `None`
                        - will be set to `[]`
                - `verbose`
                    - int, optional
                    - verbosity level
                    - how much additional information to display while executing the method
                    - the default is 0
            
            Raises
            ------

            Returns
            -------

            Comments
            --------

        """

        if show_scanned_strings_at is None: show_scanned_strings_at = []

        unique_ids = pd.unique(input_ids)

        #setup SIMBAD
        my_Simbad = Simbad()
        my_Simbad.TIMEOUT = simbad_timeout
        my_Simbad.add_votable_fields('ids')
        my_Simbad.add_votable_fields('typed_id')
        # print(my_Simbad.list_votable_fields())

        #split the query into chuncks for big queries
        intervals = np.arange(0, unique_ids.shape[0], nparallelrequests)
        if intervals[-1] < unique_ids.shape[0]: intervals = np.append(intervals, unique_ids.shape[0])
        
        for idx, (start, end) in enumerate(zip(intervals[:-1], intervals[1:])):
            
            if verbose > 0:
                print(f'Working on partition {idx+1}/{len(intervals)-1}')

            cur_unique_ids = [uid for uid in unique_ids[start:end]]

            #query SIMBAD
            id_table = my_Simbad.query_objects(cur_unique_ids)

            # print(id_table.keys())
            main_ids = id_table['MAIN_ID']
            ras = id_table['RA']
            decs = id_table['DEC']

            #extract IDs out of the query result
            simbad_result = Parallel(n_jobs=3)(delayed(re.findall)(pattern=r'[^|]+', string=str(id)) for id in id_table['IDS'])
            
            # print(simbad_result)

            for iid, ids, mid, ra, dec in zip(cur_unique_ids, simbad_result, main_ids, ras, decs):
                df_temp = pd.DataFrame()
                df_temp['input_id'] = [iid]
                df_temp['main_id'] = [mid]
                df_temp['ra'] = [ra]
                df_temp['dec'] = [dec]
                for id in ids:
                    
                    if id != mid:
                        catalogue = re.match(r'^.+[^\ ](?=\ )|NPM\d|CSI', id)
                        if catalogue is None:
                            print(f'INFO: catalog is None. Corresponding id: {id}')
                            pass
                        else:
                            id_in_cat = id.replace(catalogue[0], '')
                            df_temp[catalogue[0]] = [id_in_cat]
                            df_temp[catalogue[0]] = df_temp[catalogue[0]].str.strip()

                #update main dataframe
                self.df_ids = pd.concat([self.df_ids, df_temp], ignore_index=True)
            
            
            
            #some verbosity
            if len(show_scanned_strings_at) != 0 and verbose > 1:
                print('Scanned strings:')
                print('----------------')
                for idx in show_scanned_strings_at:
                    print(f'    Scanned Target: {id_table["TYPED_ID"][idx]}')
                    print(f'        Query Result: {id_table["IDS"][idx]}\n')
                print()            
        
        #sort self.df_id alphabetically
        self.df_ids.sort_index(axis=1, inplace=True)
        self.df_ids.insert(0, 'input_id', self.df_ids.pop('input_id'))
        
        #some verbosity
        if verbose > 2:
            print(self.df_ids)

        return 

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

