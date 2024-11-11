
#%%imports
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.table import Table
from astroquery.utils.tap.core import TapPlus
from astroquery.simbad import Simbad
from joblib.parallel import Parallel, delayed
import numpy as np
import os
import pandas as pd
from requests import HTTPError
import time

from astroLuSt.monitoring import formatting as almofo
#%%classes


#%%functions
def get_reference_objects(
    coords:SkyCoord,
    radius:float,
    verbose:int=0,
    ) -> Table:
    """
        - function to extract the coordinates of reference targets given coordinates of a science target
        - reference targets are constant stars that are close to the science target

        Parameters
        ----------
            - `coords`
                - `SkyCoord`
                - coordinates of the science target
                - will be used as center point around which to search for reference targets
            - `radius`
                - `float`
                - radius in degrees that defines the size of the (circular) search region around `coords`
                    - this region is considered to find reference targets
            - `verbose`
                - `int`, optional
                - verbosity level
                - the default is `0`

        Raises
        ------

        Returns
        -------
            - `res_tab`
                - `astropy.tables.Table()`
                - table resulting from a Simbad query
                - contains potentially constant stars within `radius` around `coords`

        Dependencies
        ------------
            - `astropy`
            - `astroquery`
            - `typing`

        Comments
        --------
            - assumes that
                - reference targets close to the science target experience the same (local) effects
                - the camera and CCD for science and reference target are identical
                    - therefore, also the timestamps will be identical
    """

    ra = coords.ra.to(u.deg).value
    dec = coords.dec.to(u.deg).value

    q = f"""
        SELECT
            subquery.*,
            mesvar.vartyp AS vartyp_mesvar
        FROM (
            SELECT TOP 100
                basic.oid,
                basic.ra, basic.dec,
                basic.main_id, basic.otype, ids.ids,
                DISTANCE(
                    POINT('ICRS', basic.ra, basic.dec),
                    POINT('ICRS', {ra}, {dec})
                ) AS "distance"
            FROM basic
                INNER JOIN ids
                    ON basic.oid = ids.oidref
            WHERE
                (otype != 'V*..') AND (otype = '*')
                AND 1 = CONTAINS(
                        POINT('ICRS', basic.ra, basic.dec),
                        CIRCLE('ICRS', {ra}, {dec}, {radius:g})
                )
            ORDER BY
                "distance" ASC
        ) AS subquery
            LEFT JOIN mesVar
                on subquery.oid = mesVar.oidref
    """

    #query for potential reference targets
    res_tab = Simbad.query_tap(query=q, maxrec=1000)

    almofo.printf(
        msg=f'Found {len(res_tab)} reference star candidate(s).',
        context=f'{get_reference_objects.__name__}()',
        type='INFO',
        level=0,
        verbose=verbose,
    )
            
    if len(res_tab) == 0:
        almofo.printf(
            msg=(
                f'WARNING: did not find any nearby constant stars for {ra=} deg, {dec=} deg, {radius=} deg.'
                f' Consider increasing `radius` or choose a different method for background-correction.'
                f' Returning `None` for that matter.'
            ),
            context=f'{get_reference_objects.__name__}()',
            type='WARNING',
            level=0,
            verbose=verbose,
        )

        res_tab = None

    return res_tab

def query_upload_table(
    tap:TapPlus,
    query:str,
    df_upload:pd.DataFrame,
    upload_table_name:str,
    nsplits:int=1,
    query_async:bool=True,
    sleep:int=0,
    verbose:int=0,
    parallel_kwargs:dict=None,
    launch_job_kwargs:dict=None,
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
                - the default is `1`
                    - upload the whole `df_upload`
            - `query_async`
                - `bool`, optional
                - whether to query in an
                    - asynchronosous manner
                        - will call `tap.launch_job_async()`
                    - synchronosous manner
                        - will call `tap.launch_job()`
                - for small queries `tap.launch_job()` is much faster, as it utilizes the local memory
                - the default is `True`
                    - will use `tap.launch_job_async()`
            - `sleep`
                - `float`, optional
                - time to sleep after each iteration
                - will be passed to `time.sleep()`
                - the deafault is `0`
            - `verbose`
                - `int`, optional
                - verbosity level
                - the default is `0`
            - `parallel_kwargs`
                - `dict`, optional
                - kwargs to pass to `joblib.parallel.Parallel()`
                - the default is `None`
                    - will be set to `dict()`
            - `launch_job_kwargs`
                - `dict`, optional
                - kwargs to pass to `tap.launch_job_async()` or `tap.launch_job()`
                    - depending on which one got used
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
            - `astroquery`
            - `astropy`
            - `joblib`
            - `numpy`
            - `pandas`
            - `os`
        
        Comments
        --------
    """

    def query_one(
        s:pd.DataFrame,
        query:str,
        temp_filename:str,
        query_async:bool,
        sleep:int,
        upload_table_name:str,
        nsplits:int, idx:int=0,
        verbose:int=0,
        launch_job_kwargs:dict=None,
        ) -> pd.DataFrame:
        """
            - subroutine for parallelization
        """

        if launch_job_kwargs is None: launch_job_kwargs = dict()

        almofo.printf(
            msg=f'Extracting split {idx+1}/{nsplits} (len(split): {len(s)})',
            context=query_upload_table.__name__,
            type='INFO',
            level=0,
            verbose=verbose
        )

        #temporarily store in votable for upload
        Table().from_pandas(s).write(temp_filename, format='votable', overwrite=True)

        #execute query (ignore if query failed due to HTTPError, but log just in case)
        try:
            if query_async:
                job = tap.launch_job_async(
                    query=query,
                    upload_resource=temp_filename,
                    upload_table_name=upload_table_name,
                    **launch_job_kwargs,
                )
            else:
                job = tap.launch_job(
                    query=query,
                    upload_resource=temp_filename,
                    upload_table_name=upload_table_name,
                    **launch_job_kwargs,
                )
            df_res = job.get_results().to_pandas()
        except HTTPError as e:
            almofo.printf(
                msg=(
                    f'HTTPError when extracting split #{idx+1}/{nsplits}, hence ignoring. '
                    f'{e}'
                ),
                context=query_upload_table.__name__,
                type='WARNING', level=1,
                verbose=verbose
            )
            df_res = None

        #clean up temporary files
        os.remove(temp_filename)
        time.sleep(sleep)

        return df_res
    
    #default parameters
    if parallel_kwargs is None:         parallel_kwargs = dict()
    if launch_job_kwargs is None: launch_job_kwargs = dict()


    
    #execute queries
    splits = np.array_split(df_upload, nsplits)
    res = Parallel(**parallel_kwargs)(
        delayed(query_one)(
            s=s,
            query=query,
            temp_filename=f'_query_upload_table_{idx:.0f}.vot',
            query_async=query_async,
            sleep=sleep,
            upload_table_name=upload_table_name,
            nsplits=nsplits, idx=idx,
            verbose=verbose,
            launch_job_kwargs=launch_job_kwargs,
        ) for idx, s in enumerate(splits)
    )


    df_res = pd.concat([r for r in res if r is not None])


    return df_res
