
#%%imports
from astropy.table import Table
from astroquery.utils.tap.core import TapPlus
from joblib.parallel import Parallel, delayed
import numpy as np
import os
import pandas as pd


#%%classes


#%%functions
def query_upload_table(
    tap:TapPlus,
    query:str,
    df_upload:pd.DataFrame,
    upload_table_name:str,
    nsplits:int=1,
    parallel_kwargs:dict=None,
    launch_job_async_kwargs:dict=None,
    ) -> pd.DataFrame:
    """
        - function to execute a query requiring a large table to be uploaded
        - will
            - split the table to be uploaded into smaller chunks
            - execute the query for each chunk
            - combine the results into one table
        
        Parameters
        ----------
            - `tap`
                - `astroquery.utils.tap.core.TapPlus`
                - table acess protocoll instance to execute the query with
            - `query`
                - `str`
                - query to be executed
                - will use `upload_table_name` to refer to the uploaded resource
            - `df_upload`
                - `pandas.DataFrame`
                - table to be uploaded to the service
                - will temporarily be split and stored as small VO-Tables
            - `upload_table_name`
                - `str`
                - name of the uploaded resource to use for referencing the table in the ADQL query
            . `nsplits`
                - `int`, optional
                - number of splits to generate of the input table
                - will be passed as `indices_or_sections` to `np.array_split()`
                - the default is 1
                    - upload the whole `df_upload`
            - `launch_job_async_kwargs`
                - `dict`, optional
                - kwargs to pass to `tap.launch_job_async()`
                - the default is `None`
                    - will be set to `dict()`

        Raises
        ------

        Returns
        -------
            - `df_res`
                - `pandas.DataFrame`
                - combined query result

        Dependencies
        ------------
            - astroquery
            - astropy
            - joblib
            - numpy
            - pandas
            - os
        
        Comments
        --------
    """

    def query_one(
        s:pd.DataFrame,
        query:str,
        temp_filename:str,
        upload_table_name:str,
        launch_job_async_kwargs:dict,
        ) -> pd.DataFrame:
        """
            - subroutine for parallelization
        """

        #temporarily store in votable for upload
        Table().from_pandas(s).write(temp_filename, format='votable')

        #execute query
        job = tap.launch_job_async(
            query=query,
            upload_resource=temp_filename,
            upload_table_name=upload_table_name,
            **launch_job_async_kwargs,
        )
        
        df_res = job.get_results().to_pandas()

        #clean up temporary files
        os.remove(temp_filename)

        return df_res
    
    #default parameters
    if parallel_kwargs is None:         parallel_kwargs = dict()
    if launch_job_async_kwargs is None: launch_job_async_kwargs = dict()


    
    #execute queries
    splits = np.array_split(df_upload, nsplits)
    res = Parallel(**parallel_kwargs)(
        delayed(query_one)(
            s=s,
            query=query,
            temp_filename=f'_query_upload_table_{idx:.0f}.vot',
            upload_table_name=upload_table_name,
            launch_job_async_kwargs=launch_job_async_kwargs,
        ) for idx, s in enumerate(splits)
    )


    df_res = pd.concat(res)


    return df_res


