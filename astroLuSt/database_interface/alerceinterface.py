
#%%imports
from alerce.core import Alerce
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.figure import Figure
import numpy as np
import os
import pandas as pd
import time
from typing import Tuple, List

from astroLuSt.monitoring import (formatting as almofo, errorlogging as almoer)


#ALeRCE
class AlerceDatabaseInterface:
    """
        - class to interact with the ZTF database via the Alerce Python API

        Attributes
        ----------
            - `sleep`
                - `float`, optional
                - number of seconds to sleep after downloading each object
                - the default is `0`
            - `n_jobs`
                - `int`, optional
                - number of jobs to be used by `joblib.Parallel()`
                - the default is `-1`
            -  `verbose`
                - `int`, optional
                - verbosity level
                - the default is `0`

        Infered Attributes
        ------------------
            - `self.alerce`
                - `alerce.core.ALERCE`
                - api to interact with the ALERCE database
            - `self.LE`
                - `astroLuSt.monitoring.errorlogging.LogErrors` instance
                - used to log and display caught errors

        Methods
        -------
            - `crossmatch_by_coordinates()`
            - `download_one()`
            - `download_lightcurves()`
            - `plot_result()`

        Dependencies
        ------------
            - `alerce`
            - `joblib`
            - `matplotlib`
            - `numpy`
            - `os`
            - `pandas`
            - `time`
            - `typing`

        Comments
        --------
    """

    def __init__(self,
        sleep:float=0,
        n_jobs:int=-1,
        verbose:int=0,
        ) -> None:

        self.sleep      = sleep
        self.n_jobs     = n_jobs
        self.verbose    = verbose

        #infered attributes
        self.alerce = Alerce()
        self.LE = almoer.LogErrors()
        
        return
    
    def __repr__(self) -> str:
        return (
            f'{self.__class__.__name__}(\n'
            f'    n_jobs={self.n_jobs},\n'
            f'    sleep={self.sleep},\n'
            f'    verbose={self.verbose},\n'
            f')'
        )

    def __dict__(self) -> dict:
        return eval(str(self).replace(self.__class__.__name__, 'dict'))

    def crossmerge_by_coordinates(self,
        df_coords:pd.DataFrame,
        ra_colname:str, dec_colname:str, radius:float,
        sleep:float=None,
        n_jobs:int=None,
        verbose:int=None,
        parallel_kwargs:dict=None,
        ) -> pd.DataFrame:
        """
            - method to crossmerge `df_coords` with the ZTF catalog by coordinates (cone search)
                - will take each row in `df_coords` and find the corresponding target in the ZTF catalog via coordinates
                - will then proceed to append to each matched entry in the ZTF catalog the input row from `df_coords`
                - will combine everything into one huge table
                - will ignore anything that is not extractable and track failed extractions in `self.LE.df_errorlog`

            Parameters
            ----------
                - `df_coords`
                    - `pd.DataFrame`
                    - table to be crossmerged with ZTF
                    - must contain ra and dec as columns
                - `ra_colname`
                    - `str`
                    - name of the column to be considered as Right Ascension
                - `dec_colname`
                    - `str`
                    - name of the column to be considered as Declination
                - `radius`
                    - `float`
                    - radius to use for the cone search
                - `sleep`
                    - `float`, optional
                    - number of seconds to sleep after downloading each target
                    - the default is `None`
                        - will fall back to `self.sleep`                   
                - `n_jobs`
                    - `int`, optional
                    - number of jobs to be used by `joblib.Parallel()`
                    - the default is `None`
                        - will fall back to `self.n_jobs`
                -  `verbose`
                    - `int`, optional
                    - verbosity level
                    - the default is `None`
                        - will fall back to `self.verbose`
                - `parallel_kwargs`
                    - `dict`, optional
                    - kwargs to pass to `joblib.Parallel`
                    - the default is `None`
                        - will be set to `dict(backend='threading')`

            Raises
            ------

            Returns
            -------
                - `df`
                    - `pd.DataFrame`
                    - ZTF catalog crossmerged with `df_coords` via coordinates

            Comments
            --------

        """
        def query_one(
            idx:int, inrow:pd.DataFrame,
            ra_colname:str, dec_colname:str, radius:float,
            total_targets:int,
            sleep:float,
            verbose:int,
            ) -> pd.DataFrame:

            almofo.printf(
                msg=f"Extracting #{idx+1}/{total_targets}",
                context=self.crossmerge_by_coordinates.__name__,
                type='INFO',
                level=0,
                verbose=verbose
            )

            try:
                df = self.alerce.query_objects(
                    format='pandas',
                    ra=inrow[ra_colname], dec=inrow[dec_colname], radius=radius
                )

                df.rename(columns={'meanra':'raj2000', 'meandec':'dej2000'}, inplace=True)
                df = df.add_suffix('_ztf')

                #append all entries of inrow to each of the extracted rows in df
                df.loc[:, inrow.index] = inrow.values

                #correct dtypes
                regex = "^g_r.+$"
                df[df.filter(regex=regex).columns] = df.filter(regex=regex).astype(np.float64)


            except Exception as e:
                df = pd.DataFrame() #empty DataFrame
                self.LE.print_exc(
                    e,
                    prefix=f"{inrow['id']}",
                    suffix=self.crossmerge_by_coordinates.__name__,
                    verbose=verbose
                )
                self.LE.exc2df(
                    e,
                    prefix=f"{inrow['id']}",
                    suffix=self.crossmerge_by_coordinates.__name__,
                    verbose=verbose
                )

            #sleep after each target
            time.sleep(sleep)

            return df

        #default parameters
        if n_jobs is None:                  n_jobs                  = self.n_jobs
        if sleep is None:                   sleep                   = self.sleep
        if verbose is None:                 verbose                 = self.verbose
        if parallel_kwargs is None:         parallel_kwargs         = dict(backend='threading')

        if parallel_kwargs is None:         parallel_kwargs         = dict(backend='threading')
        if 'backend' not in parallel_kwargs.keys():
            parallel_kwargs['backend'] = 'threading'
        if 'verbose' not in parallel_kwargs.keys():
            parallel_kwargs['verbose'] = verbose

        result = Parallel(n_jobs, **parallel_kwargs)(
            delayed(query_one)(
                idx=idx, inrow=inrow,
                ra_colname=ra_colname, dec_colname=dec_colname, radius=radius,
                total_targets=df_coords.shape[0],
                sleep=sleep,
                verbose=verbose,
            ) for idx, inrow in df_coords.iterrows()
        )

        df = pd.concat([r for r in result if not r.empty], ignore_index=True)

        return df

    def download_one(self,
        ztf_id:str,
        redownload:bool=False,
        save:str=False,
        sleep:float=None,
        idx:int=0,
        total_targets:int=1,
        verbose:int=None,
        ) -> pd.DataFrame:
        """
            - method to download the lightcurve of one particular `ztf_id`
            - will be called during `self.download_lightcurves()` in parallel
            - will ignore anything that is not extractable and track failed extractions in `self.LE.df_errorlog`
            
            Parameters
            ----------
                - `ztf_id`
                    - `str`
                    - id of the desired target
                - `redownload`
                    - `bool`, optional
                    - whether to redownload lightcurves that already have been donwloaded at some point in the past
                        - i.e. are found in the save-directory
                    - the default is False
                - `save`
                    - str, `bool`, optional
                    - directory of where to store the downloaded lightcurves
                    - if set to `False`, will not save the data
                    - `save` has to end with a slash (`'/'`)
                    - the default is `'./'`
                - `sleep`
                    - `float`, optional
                    - number of seconds to sleep after downloading each target
                    - the default is `None`
                        - will fall back to `self.sleep`
                - `idx`
                    - `int`, optional
                    - index of the currently downloaded target
                    - only necessary to print in the protocoll when called in `self.download_lightcurves()`
                    - the default is `0`
                - `total_targets`
                    - `int`, optional
                    - total number of targets that get extracted
                    - only necessary to print in the protocoll when called in `self.download_lightcurves()`
                    - the default is `1`
                -  `verbose`
                    - `int`, optional
                    - verbosity level
                    - the default is `None`
                        - will fall back to `self.verbose`

            Raises
            ------

            Returns
            -------
                - `df`
                    - `pd.DataFrame`
                    - containing the downloaded lightcurve data

            Comments
            --------  

        """

        #default parameters
        if sleep is None:   sleep   = self.sleep
        if verbose is None: verbose = self.verbose


        #current filename
        savefile = f'{save}{ztf_id}.parquet'

        #get files that already have been extracted
        try:
            already_extracted = os.listdir(str(save))
        except:
            already_extracted = []

        almofo.printf(
            msg=f"Extracting {ztf_id} (#{idx+1}/{total_targets})",
            context=self.download_one.__name__,
            type='INFO',
            level=0,
            verbose=verbose
        )

        if savefile.replace(str(save),'') not in already_extracted or redownload:

            try:    #try extraction
                df = self.alerce.query_detections(
                    ztf_id,
                    format='pandas'
                )

                if isinstance(save, str): df.to_parquet(savefile, index=False)

            except Exception as e:  #skip if extraction fails
                #empty placeholder DataFrame
                df = pd.DataFrame()
                self.LE.print_exc(
                    e,
                    prefix=f"{ztf_id}",
                    suffix=self.download_one.__name__,
                    verbose=verbose
                )
                self.LE.exc2df(
                    e,
                    prefix=f"{ztf_id}",
                    suffix=self.download_one.__name__,
                    verbose=verbose
                )

            #sleep after downloading one target
            time.sleep(sleep)
        else:   #load data if already extracted
            df = pd.read_parquet(savefile) #empty placeholder
            almofo.printf(
                msg=f"{ztf_id} has already been extracted and `redownload==False`... ignoring",
                context=self.download_one.__name__,
                type='INFO',
                level=1,
                verbose=verbose
            )

        return df

    def download_lightcurves(self,
        ztf_ids:List[str],
        save:str=False,
        redownload:bool=False,
        sleep:float=0,
        n_jobs:int=-1,
        verbose:int=None,
        parallel_kwargs:dict=None,
        ) -> List[pd.DataFrame]:
        """
            - function to download all lightcurves corresponding to the ZTF ids in `ztf_ids`
            - will remove failed extractions from the output
                - failed extractions are tracked in `self.LE.df_errorlog`

            Parameters
            ----------
                - `ztf_ids`
                    - `List[str]`
                    - ids to extract the lcs for
                - `save`
                    - `str`, `bool`, optional
                    - directory of where to store the downloaded lightcurves to
                    - if set to False, will not save the data
                    - save has to end with a slash (`'/'`)
                    - the default is `False`
                - `redownload`
                    - `bool`, optional
                    - whether to redownload lightcurves that already have been donwloaded at some point in the past
                        - i.e. are found in the save-directory
                    - the default is `False`                   
                - `sleep`
                    - `float`, optional
                    - number of seconds to sleep after downloading each target
                    - the default is `None`
                        - will fall back to `self.sleep`                   
                - `n_jobs`
                    - `int`, optional
                    - number of jobs to be used by `joblib.Parallel()`
                    - the default is `None`
                        - will fall back to `self.n_jobs`
                -  `verbose`
                    - `int`, optional
                    - verbosity level
                    - the default is `None`
                        - will fall back to `self.verbose`
                - `parallel_kwargs`
                    - `dict`, optional
                    - kwargs to pass to `joblib.Parallel`
                    - the default is `None`
                        - will be set to `dict(backend='threading')`

            Raises
            ------
                - assertion error
                    - if 'save' or 'save_plots' are not formatted correctly
                
            Returns
            -------
                - `dfs_lc`
                    - `List[pd.DataFrame]`
                    - list containing the dataframes for all extracted objects

            Comments
            --------

        """

        #default parameters
        if n_jobs is None:                  n_jobs                  = self.n_jobs
        if sleep is None:                   sleep                   = self.sleep
        if verbose is None:                 verbose                 = self.verbose
        if parallel_kwargs is None:         parallel_kwargs         = dict(backend='threading')
        if 'backend' not in parallel_kwargs.keys():
            parallel_kwargs['backend'] = 'threading'
        if 'verbose' not in parallel_kwargs.keys():
            parallel_kwargs['verbose'] = verbose

        if isinstance(save, str):
            assert save[-1] == '/' or save[-1] == '\\', \
                '"save" has to end either with a slash ("/") or backslash ("\\")'

        dfs_lc = Parallel(n_jobs, **parallel_kwargs)(
            delayed(self.download_one)(
                ztf_id=ztf_id,
                save=save,
                redownload=redownload,
                sleep=sleep,
                total_targets=len(ztf_ids),
                idx=idx,
                verbose=verbose,                
            ) for idx, ztf_id in enumerate(ztf_ids)
        )

        #remove failed extractions
        dfs_lc = [df for df in dfs_lc if len(df) > 0]

        return dfs_lc

    def plot_result(self,
        df:pd.DataFrame,
        ) -> Tuple[Figure, plt.Axes]:
        """
            - method to plot the result of the extraction

            Parameters
            ----------
                - `df`
                    - `pd.DataFrame`
                    - dataframe containing the downloaded data for one extracted object
            
            Returns
            -------
                - `fig`
                    - `Figure`
                    - created figure
                - `axs`
                    - `plt.Axes`
                    - axes corresponding to `fig`
            
            Comments
            --------

        """

        cm = plt.cm.get_cmap('viridis')
        newcolors = cm(np.linspace(0, 1, 256))
        newcolors[        :1*256//3, :] = mcolors.to_rgba('tab:green')
        newcolors[1*256//3:2*256//3, :] = mcolors.to_rgba('tab:red')
        newcolors[2*256//3:3*256//3, :] = mcolors.to_rgba('tab:purple')
        newcmap = mcolors.ListedColormap(newcolors)

        fig = plt.figure()
        ax1 = fig.add_subplot(111)

        try:
            sctr = ax1.scatter(df['mjd'], df['magpsf_corr'], c=df['fid'], cmap=newcmap, vmin=1, vmax=3, marker='^', label='magpsf_corr')
        except:
            pass
        sctr = ax1.scatter(df['mjd'], df['magpsf'],   c=df['fid'], cmap=newcmap, vmin=1, vmax=3, marker='.', label='magpsf')
        sctr = ax1.scatter(df['mjd'], df['magapbig'], c=df['fid'], cmap=newcmap, vmin=1, vmax=3, marker='s', label='magapbig')
        
        cbar = fig.colorbar(sctr, ax=ax1)
        cbar.set_label('Filter ID')
        cbar.set_ticks([1+(3-1)*1/6, 1+(3-1)*3/6, 1+(3-1)*5/6])
        cbar.set_ticklabels([1, 2, 3])

        ax1.invert_yaxis()

        ax1.set_xlabel('Modified JD')
        ax1.set_ylabel(r'm [mag]')

        ax1.legend()

        axs = fig.axes

        return fig, axs
