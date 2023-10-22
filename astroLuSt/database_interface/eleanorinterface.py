
#%%imports
import eleanor
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.figure import Figure
import numpy as np
import os
import pandas as pd
import shutil
import time
from typing import Union, Tuple, Callable, List
import warnings

from astroLuSt.monitoring import timers as almt
from astroLuSt.monitoring import errorlogging as alme
from astroLuSt.monitoring import formatting as almf

#%%catch problematic warnings and convert them to exceptions
w2e1 = r".*The following header keyword is invalid or follows an unrecognized non-standard convention:.*"
warnings.filterwarnings("error", message=w2e1)

class EleanorDatabaseInterface:

    def __init__(self,
        sleep:float=0,
        n_jobs:int=-1,
        clear_metadata:bool=False,
        verbose:int=0,
        ) -> None:
        
        self.sleep          = sleep
        self.n_jobs         = n_jobs
        self.clear_metadata = clear_metadata
        self.verbose        = verbose

        #infered attributes
        self.LE = alme.LogErrors()


        pass

    def __repr__(self) -> str:
        return (
            f'{self.__class__.__name__}(\n'
            f'\n'
            f')'
        )
    
    def __dict__(self) -> dict:
        return eval(str(self).replace(self.__class__.__name__, 'dict'))

    def extract_source(self,
        sectors:Union[str,List]=None,
        source_id:dict=None,
        multi_sectors_kwargs:dict=None,
        targetdata_kwargs:dict=None,
        save_kwargs:dict=None,
        ) -> np.ndarray:

        if sectors is None:                 sectors                 = 'all'
        if source_id is None:               source_id               = dict()
        if multi_sectors_kwargs is None:    multi_sectors_kwargs    = dict()
        if targetdata_kwargs is None:       targetdata_kwargs       = dict()
        
        save_kwargs_use = save_kwargs.copy()
        if save_kwargs_use is not None:
            if 'filename' not in save_kwargs_use.keys():
                save_kwargs_use['filename'] = '_'.join([''.join(item) for item in source_id.items()])
        
        
        #define storage structures
        lcs = []
        headers = ['time', 'raw_flux', 'flux_err', 'corr_flux', 'quality', 'sector', 'tess_mag', 'aperture_size']
        if 'do_pca' in targetdata_kwargs.keys(): headers += ["pca_flux"]*targetdata_kwargs['do_pca']
        if 'do_psf' in targetdata_kwargs.keys(): headers += ["psf_flux"]*targetdata_kwargs['do_psf']
        
        #extract data
        try:
            #obtain sources
            star = eleanor.multi_sectors(
                sectors=sectors,
                **source_id,
                **multi_sectors_kwargs,
            )

            for idx, s in enumerate(star):
                datum = eleanor.TargetData(source=s, **targetdata_kwargs)

                lc = np.array([
                    datum.time,
                    datum.raw_flux, datum.flux_err,
                    datum.corr_flux,
                    datum.quality,
                    [s.sector]*datum.time.shape[0],
                    [s.tess_mag]*datum.time.shape[0],
                    [datum.aperture_size]*datum.time.shape[0],
                ]).T

                if datum.pca_flux is not None:
                    lc = np.append(lc, np.expand_dims(datum.pca_flux,1), axis=1)
                if datum.psf_flux is not None:
                    lc = np.append(lc, np.expand_dims(datum.psf_flux,1), axis=1)

                lcs.append(lc)

                if idx > 0: 
                    _ = 1/0

            
            lcs = np.concatenate(lcs, axis=0)

            if save_kwargs_use is not None:
                self.save(
                    df=pd.DataFrame(data=lcs, columns=headers),
                    **save_kwargs_use,
                )

            #sleep to prevent server timeout
            time.sleep(self.sleep)
        
        except Exception as e:
            lcs = np.empty((0,len(headers)))

            self.LE.print_exc(
                e,
                prefix=f'{source_id}',
                suffix=None,
            )

        print(self.LE.df_errorlog)

        return lcs
    
    def download(self,
        sectors:Union[str,list]=None,
        source_ids:List[dict]=None,
        n_chunks:int=1,
        parallel_kwargs:dict=None,
        multi_sectors_kwargs:dict=None,
        targetdata_kwargs:dict=None,
        save_kwargs:dict=None,
        verbose:int=None,
        ):

        #default parameters
        if verbose is None: verbose = self.verbose
        if parallel_kwargs is None:
            parallel_kwargs = dict(n_jobs=self.n_jobs, backend='threading', verbose=verbose)

        #split into chunks
        chunks = np.array_split(source_ids, n_chunks)


        #iterate over chunks
        for cidx, chunk in enumerate(chunks):
            almf.printf(
                msg=f'Extracting chunk {cidx+1}/{len(chunks)}',
                context=f'{self.__class__.__name__}.{self.download.__name__}()',
                type='INFO',
                verbose=verbose,
            )

            #extract targets (in parallel)
            res = Parallel(**parallel_kwargs)(
                delayed(self.extract_source)(
                    sectors=sectors,
                    source_id=source_id,
                    multi_sectors_kwargs=multi_sectors_kwargs,
                    targetdata_kwargs=targetdata_kwargs,
                    save_kwargs=save_kwargs,
                ) for idx, source_id in enumerate(chunk)
            )

            print(res)


        return
    
    def save(self,
        df:pd.DataFrame,
        filename:str,
        directory:str=None,
        pd_savefunc:str=None,
        save_kwargs:dict=None,
        ) -> None:

        if directory is None:   directory   = './'
        if pd_savefunc is None: pd_savefunc = 'to_csv'
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

    def plot_result(self,
        ) -> Tuple[Figure,plt.Axes]:
        
        return

# #ELEANOR
# class EleanorDatabaseInterface:
#     """
#         - class to interact with the eleanor code for downloading lightcurves from the MAST archive

        
#         Attributes
#         ----------
#             - `tic`
#                 - list[int], optional
#                 - list of TIC identifiers
#                 - the default is `None`
#                     - will be initialized with `[]`
#             - `sectors`
#                 - list[int], str, optional
#                 - list of sectors to extract
#                 - allowed strings
#                     - `'all'`
#                         - if all available sectors shall be considered
#                 - the default is `'all'`
#             - `do_psf`
#                 - bool, optional
#                 - whether to execute a psf
#                 - argument of `eleanor.multi_sectors()`
#                 - the default is `False`
#             - `do_pca`
#                 - bool, optional
#                 - whether to execute a pca
#                 - argument of `eleanor.multi_sectors()`
#                 - the default is `False`
#             - `aperture_mode`
#                 - str, optional
#                 - specify which aperture to use
#                     - one of
#                         - `'small'`
#                         - `'normal'`
#                         - `'large'`
#                 - argument of `eleanor.TargetData()`
#                 - the default is `'normal'`
#             - `regressors`
#                 - str, optional
#                 - which methods to use to estimate the background
#                 - argument of `eleanot.TargetData()`
#                 - the default is `'corners'`
#             - `try_load`
#                 - bool, optional
#                 - whether to search hidden `~/.eleanor` dictionary for alrady existing TPF
#                 - argument of `eleanor.TargetData()`
#                 - the default is `True`
#             - `height`
#                 - int, optional
#                 - pixel height of the cutout
#                 - argument of `eleanor.multi_sectors()`
#                 - the default is 15
#             - `width`
#                 - int, optional
#                 - pixel width of the cutout
#                 - argument of `eleanor.multi_sectors()`
#                 - the default is 15
#             - `bkg_size`
#                 - int, optional
#                 - argument of `eleanor.multi_sectors()`
#                     - see documentation for more info
#                 - the default is 31
#             - `clear_metadata_after_extract`
#                 - bool, optional
#                 - whether to delete all downloaded metadata after the extraction of any target
#                 - if `False`
#                     - will result in faster download, if the target is downloaded again
#                 - the default is `False`


        
#         Methods
#         -------
#             - `data_from_eleanor_alltics()`
#             - `data_from_eleanor()`
#             - `plot_result_eleanor()`
#             - `result2pandas()`
#             - `save_npy_eleanor()`
#                 - not maintained
#                 - might have compatibility issues

#         Dependencies
#         ------------
#             - eleanor
#             - joblib
#             - matplotlib
#             - numpy
#             - os
#             - pandas
#             - shutil
#             - time
#             - warnings

#     """

#     def __init__(self,
#         tics:List[int]=None, sectors:Union[List[int],str]='all',
#         do_psf:bool=False, do_pca:bool=False,
#         aperture_mode:str='normal', regressors:str='corner', try_load:bool=True,
#         height:int=15, width:int=15, bkg_size:int=31,
#         clear_metadata_after_extract:bool=False,
#         ) -> None:


#         if tics is None:
#             self.tics = []
#         else:
#             self.tics = tics
#         self.sectors = sectors
#         self.do_psf = do_psf
#         self.do_pca = do_pca
#         self.aperture_mode = aperture_mode
#         self.regressors = regressors
#         self.try_load = try_load
#         self.height = height
#         self.width = width
#         self.bkg_size = bkg_size
#         self.clear_metadata_after_extract = clear_metadata_after_extract

#         self.metadata_path = "./mastDownload/HLSP"

#         self.ET = ExecTimer()

#         return

#     def __repr__(self) -> str:
#         return (
#             f'EleanorDatabaseInterface(\n'
#             f'    tics={repr(self.tics)}, sectors={repr(self.sectors)},\n'
#             f'    do_psf={repr(self.do_psf)}, do_pca={repr(self.do_pca)},\n'
#             f'    aperture_mode={repr(self.aperture_mode)}, regressors={repr(self.regressors)}, try_load={repr(self.try_load)},\n'
#             f'    height={repr(self.height)}, width={repr(self.width)}, bkg_size={repr(self.bkg_size)},\n'
#             f'    clear_metadata_after_extract={repr(self.clear_metadata_after_extract)},\n'
#             f')'
#         )

#     def data_from_eleanor(self,
#         tic:int, sectors:Union[List[int],str]="all", 
#         do_psf:bool=False, do_pca:bool=False, 
#         aperture_mode:str="normal", regressors:str="corner", try_load:bool=True,
#         height:int=15, width:int=15, bkg_size:int=31,
#         verbose:int=2
#         ) -> Tuple[list,list,list,str]:
#         """
#             - function to download data using eleonor

#             Parameters
#             ----------
#                 - `tic`
#                     - int
#                     - TIC identifier number of the target
#                 - `sectors`
#                     - list of int, str, optional
#                     - sectors to search in
#                     - argument of `eleanor.multi_sectors()`
#                     - the default is `'all'`
#                         - will consider all sectors
#                 - `do_psf`
#                     - bool, optional
#                     - whether to execute a psf
#                     - argument of `eleanor.multi_sectors()`
#                     - the default is `False`
#                 - `do_pca`
#                     - bool, optional
#                     - whether to execute a pca
#                     - argument of `eleanor.multi_sectors()`
#                     - the default is `False`
#                 - `aperture_mode`
#                     - str, optional
#                     - specify which aperture to use
#                         - one of
#                             - `'small'`
#                             - `'normal'`
#                             - `'large'`
#                     - argument of `eleanor.TargetData()`
#                     - the default is `'normal'`
#                 - `regressors`
#                     - str, optional
#                     - which methods to use to estimate the background
#                     - argument of `eleanot.TargetData()`
#                     - the default is `'corners'`
#                 - `try_load`
#                     - bool, optional
#                     - whether to search hidden `~/.eleanor` dictionary for alrady existing TPF
#                     - argument of `eleanor.TargetData()`
#                     - the default is `True`
#                 - `height`
#                     - int, optional
#                     - pixel height of the cutout
#                     - argument of `eleanor.multi_sectors()`
#                     - the default is 15
#                 - `width`
#                     - int, optional
#                     - pixel width of the cutout
#                     - argument of `eleanor.multi_sectors()`
#                     - the default is 15
#                 - `bkg_size`
#                     - int, optional
#                     - argument of `eleanor.multi_sectors()`
#                         - see documentation for more info
#                     - the default is 31
#                 - `verbose`
#                     -int, optional
#                     - verbosity level
#                     - the higher the more information will be shown
#                     - the default is 2

#             Raises
#             ------
            
#             Returns
#             -------
#                 - `data`
#                     - list
#                     - list of the data extracted
#                         - entries are the different sectors
#                 - `sectors`
#                     - list
#                     - list of the observed sectors
#                 - `tess_mags`
#                     - list
#                     - contains the tess-magnitudes for each sector
#                 - `error_msg`
#                     - str
#                     - error message if the extraction fails

#             Comments
#             --------

#         """

#         try:
#             error_msg = None

#             star = eleanor.multi_sectors(tic=tic, sectors=sectors)
            
#             data = []
#             sectors = []
#             tess_mags = []
#             for s in star:
#                 #try if sector works
#                 try:
#                     datum = eleanor.TargetData(
#                         s, height=height, width=width, bkg_size=bkg_size, 
#                         do_psf=do_psf, 
#                         do_pca=do_pca,
#                         aperture_mode=aperture_mode,
#                         try_load=try_load,
#                         regressors=regressors,
#                     )

#                     data.append(datum)

#                     #Get sector
#                     sectors.append(s.sector)

#                     #Get TESS-magnitude
#                     tess_mags.append(s.tess_mag)
                
#                 #if sector fails return row of np.nan
#                 except Exception as e:
#                     if verbose > 1:
#                         print(f"WARNING: Error in eleanor.TargetData(sector={s.sector})")
#                         print(f"ORIGINAL ERROR: {e}")
#                     data.append(None)
#                     sectors.append(s.sector)
#                     tess_mags.append(s.tess_mag)

#                     if len(str(e)) > 80: e = str(e)[:79] + '...'
#                     td_text = f"eleanor.TargetData(sector={s.sector})"
#                     error_msg = f"{td_text:35s}: {e}"


#         except Exception as e:
#             if verbose > 1:
#                 print("WARNING: Error in eleanor.multi_sectors()")
#                 print(f"\t Maybe TIC {tic} was not observed with TESS?")
#                 print(f"\t If you run extract_data() in a loop, try to run it separately on TIC {tic}.")
#                 print(f"ORIGINAL ERROR: {e}")
            
#             data = None
#             sectors = None
#             tess_mags = None
#             if len(str(e)) > 80: e = str(e)[:79] + '...'
#             error_msg = f"{'eleanor.multi_sectors()':25s}: {e}"
        
#         return data, sectors, tess_mags, error_msg

#     def extraction_onetic(self,
#         tic:int,
#         #saving data
#         save:Union[str,bool]='./',
#         redownload:bool=True,
#         #plotting
#         plot_result:bool=True,
#         save_plot:str=False,
#         #computing
#         sleep:float=0,
#         cidx:int=0, idx:int=0, chunk_len:int=1, n_chunks:int=1,
#         verbose:int=0,
#         save_kwargs:dict={},
#         plot_kwargs:dict={},
#         ) -> dict:
#         """
#             - method to download and store the data for one particular TIC

#             Parameters
#             ----------
#                 - `tic`
#                     - int, optional
#                     - TIC number of the target to extract
#                 - `save`
#                     - str, bool, optional
#                     - path to the directory of where to store the extracted files
#                     - the default is `"./"`
#                         - will save to the current working directory
#                 - `redownload`
#                     - bool, optional
#                     - whether to download a target again, if a file with the same name already exists in `save`
#                     - the default is `True`
#                 - `plot_result`
#                     - bool, optional
#                     - whether to create a plot of the extracted file
#                     - the default is `True`
#                 - `save_plot`
#                     - str, optional
#                     - location of where to save the created plot
#                     - the default is `False`
#                         - will not save the plot
#                 - `sleep`
#                     - float, optional
#                     - number of seconds to sleep after downloading each target
#                     - the default is 0
#                         - no sleep at all                
#                 - `n_jobs`
#                     - int, optional
#                     - number of workers to use for parallel execution of tasks
#                         - `1` is essentially sequential computaion (useful for debugging)
#                         - `2` uses 2 CPUs
#                         - `-1` will use all available CPUs
#                         - `-2` will use all available CPUs-1
#                     - for more info see https://joblib.readthedocs.io/en/latest/generated/joblib.Parallel.html
#                     - the default is `-1`
#                 - `cidx`
#                     - int, optional
#                     - index of the chunk that currently gets extracted
#                     - only used for monitoring download process in `self.data_from_eleanor_alltics()`
#                     - the default is 0
#                 - `idx`
#                     - int, optional
#                     - index of the target within the current chunk
#                     - only used for monitoring download process in `self.data_from_eleanor_alltics()`
#                     - the default is 0
#                 - `chunk_len`
#                     - int, optional
#                     - length of the current chunk
#                     - only used for monitoring download process in `self.data_from_eleanor_alltics()`
#                     - the default is 1
#                 - `n_chunks`
#                     - int, optional
#                     - total number of chunks to be extracted via `self.data_from_eleanor_alltics()`
#                     - only used for monitoring download process in `self.data_from_eleanor_alltics()`
#                     - the default is 1
#                 - `verbose`
#                     - int, optional
#                     - verbosity level
#                         - also for `joblib.Parallel()`
#                     - higher numbers will show more information
#                     - the default is 2
#                 - `save_kwargs`
#                     - dict, optional
#                     - kwargs passed to `self.result2pandas()`
#                 - `plot_kwargs`
#                     - dict, optional
#                     - kwargs passed to `self.plot_result_eleanor()`
            
#             Raises
#             ------

#             Returns
#             -------
#                 - `extraction_summary`
#                     - dict
#                     - contains
#                         - tic
#                             - target id
#                         - success
#                             - whether the extraction was successful
#                         - original error message
#                             - the error that occured, if an error occured

#             Comments
#             --------


            
#         """
        
#         savefile = f"{save}tic{tic}.csv"
        
#         #get files that already have been extracted
#         try:
#             already_extracted = os.listdir(str(save))
#         except:
#             already_extracted = []

#         save_kwargs['save'] = savefile
#         plot_kwargs['save'] = save_plot

#         #print protocoll of download
#         print(f"\nExtracting tic{tic} (chunk {cidx+1}/{n_chunks}, {idx+1}/{chunk_len} = {chunk_len*(cidx)+idx+1}/{len(self.tics)} Total)")

#         #redownload target if specified, otherwise ignore if it already has been downloaded before
#         if savefile.replace(str(save),'') not in already_extracted or redownload:

#             #download data
#             data, sectors, tess_mags, error_msg = self.data_from_eleanor(
#                 tic, sectors=self.sectors,
#                 do_psf=self.do_psf, do_pca=self.do_pca,
#                 aperture_mode=self.aperture_mode, regressors=self.regressors, try_load=self.try_load,
#                 height=self.height, width=self.width, bkg_size=self.bkg_size,
#                 verbose=verbose
#             )

#             #if the download succeeded
#             if data is not None:

#                 #save data
#                 if isinstance(save, str):
#                     df_res = self.result2pandas(
#                         # save=savefile,
#                         data=data, sectors=sectors, tess_mags=tess_mags, tic=tic,
#                         **save_kwargs
#                     )

#                 #plot data
#                 if plot_result:
#                     if isinstance(save_plot, str): save_plot = f"{save_plot}tic{tic}.png"
#                     else: save_plot = False
#                     fig, axs = self.plot_result_eleanor(
#                         data=data, tic=tic, sectors=sectors, tess_mags=tess_mags,
#                         **plot_kwargs              
#                     )
                
#                 extraction_summary = {"TIC":tic, "success":True, "original error message":error_msg}
#             #if the download fails report in extraction summary
#             else:
#                 #append failed extractions
#                 extraction_summary = {"TIC":tic, "success":False, "original error message":error_msg}         

#             #sleep after each target
#             time.sleep(sleep)
        
#         #report in extraction summary if target already existed before the download
#         else:
#             error_msg = f'WARNING: {savefile} already exists in {save}. Ignoring target because "redownload" == False '
#             print(error_msg)
#             extraction_summary = {'TIC':tic, 'success':False, 'original error message':error_msg}

#         return extraction_summary

#     def data_from_eleanor_alltics(self,
#         #saving data
#         save:Union[str,bool]="./",
#         redownload:bool=True,
#         #plotting
#         plot_result:bool=True,
#         save_plot:str=False,
#         #computing
#         sleep:float=0,
#         n_jobs:int=-1, n_chunks:int=1,
#         verbose:int=2,
#         #kwargs
#         extraction_onetic_kwargs:dict={},
#         ) -> pd.DataFrame:
#         """
#             - method to extract data for all ids in `self.tics`

#             Parameters
#             ----------
#                 - `save`
#                     - str, bool, optional
#                     - path to the directory of where to store the extracted files
#                     - the default is `"./"`
#                         - will save to the current working directory
#                 - `redownload`
#                     - bool, optional
#                     - whether to download a target again, if a file with the same name already exists in `save`
#                     - the default is `True`
#                 - `plot_result`
#                     - bool, optional
#                     - whether to create a plot of the extracted file
#                     - the default is `True`
#                 - `save_plot`
#                     - str, optional
#                     - location of where to save the created plot
#                     - the default is `False`
#                         - will not save the plot
#                 - `sleep`
#                     - float, optional
#                     - number of seconds to sleep after downloading each target
#                     - the default is 0
#                         - no sleep at all                
#                 - `n_jobs`
#                     - int, optional
#                     - number of workers to use for parallel execution of tasks
#                         - `1` is essentially sequential computaion (useful for debugging)
#                         - `2` uses 2 CPUs
#                         - `-1` will use all available CPUs
#                         - `-2` will use all available CPUs-1
#                     - for more info see https://joblib.readthedocs.io/en/latest/generated/joblib.Parallel.html
#                     - the default is `-1`
#                 - `n_chunks`
#                     - int, optional
#                     - number of chuncks to split the input data into
#                         - will take `self.tics` and split it into `n_chunks` more or less equal subarrays
#                     - after processing of each chunk, the metadata will get deleted, if `self.clear_metadata_after_extract` is set to `True`
#                     - it is advisable, though not necessary, to choose `n_chunks` such that each chunck contains an amount of elements that can be evenly distributed across `n_jobs`
#                     - the default is 1
#                         - i.e. process all the data in one go
#                 - `verbose`
#                     - int, optional
#                     - verbosity level
#                         - also for `joblib.Parallel()`
#                     - higher numbers will show more information
#                     - the default is 2
#                 - `extraction_onetic_kwargs`
#                     - dict, optional
#                     - kwargs passed to `self.extraction_onetic()`
            
#             Raises
#             ------

#             Returns
#             -------
#                 - `df_extraction_summary`
#                     - pd.DataFrame
#                     - dataframe containing a summary of the extraction success for each tic

#             Comments
#             --------


#         """
#         #TODO: include exect timer + runtime estimation for verbose > 2

#         if verbose > 2:
#             self.ET.checkpoint_start('EleanorDatabaseInterface().data_from_eleanor_alltics()')

#         #split array into n_chuncks chuncks to divide the load
#         chunks = np.array_split(self.tics, n_chunks)
#         if verbose > 1:
#             print(f"INFO: Extracting {len(chunks)} chuncks with shapes {[c.shape for c in chunks]}")

#         #sucess-monitoring
#         df_extraction_summary = pd.DataFrame()

#         #iterate over created chunks
#         for cidx, chunk in enumerate(chunks):

#             print(f"INFO: Working on chunk {cidx+1}/{len(chunks)} with size {chunk.shape[0]}")
           
#             #extract data
#             extraction_summary = Parallel(
#                 n_jobs=n_jobs, verbose=verbose
#             )(
#                 delayed(self.extraction_onetic)(
#                         tic=tic,
#                         save=save,
#                         save_plot=save_plot,
#                         redownload=redownload,
#                         plot_result=plot_result,
#                         sleep=sleep,
#                         verbose=verbose,
#                         cidx=cidx, idx=idx, chunk_len=len(chunk), n_chunks=n_chunks,
#                         **extraction_onetic_kwargs,
#                     ) for idx, tic in enumerate(chunk)
#             )

#             #delete metadata after extraction of each chunck
#             if self.clear_metadata_after_extract:
                
#                 try:
#                     print('INFO: Removing Metadata...')
#                     shutil.rmtree(self.metadata_path)
#                 except FileNotFoundError as e:
#                     print(f'INFO: No Metadata to clear.')
#                     print(f'    Original ERROR: {e}')

#             df_extraction_summary_chunck = pd.DataFrame(data=extraction_summary)
#             df_extraction_summary = pd.concat([df_extraction_summary, df_extraction_summary_chunck], ignore_index=True)


#         #short protocoll about extraction
#         if verbose > 1:
#             print(f"\nExtraction summary:")
#             print(df_extraction_summary)

#         if verbose > 2:
#             self.ET.checkpoint_end('EleanorDatabaseInterface().data_from_eleanor_alltics()')


#         return df_extraction_summary

#     def plot_result_eleanor(self,
#         data:list, tic:int, sectors:list, tess_mags:list,
#         quality_expression:str="(datum.quality == 0)",
#         aperture_detail:int=50, ylims:list=None,
#         fontsize:int=16, figsize:tuple=(16,9), save:Union[str,bool]=False
#         ) -> Tuple[Figure, plt.Axes]:
#         """
#             - function to autogenerate a summary-plot of the data downloaded using eleonor
        
#             Parameters
#             ----------
#                 - `data`
#                     - list
#                     - contains the data for each sector
#                         - extracted with eleanor
#                 - `tic`
#                     - str
#                     - TIC identifier of the target
#                 - `sectors`
#                     - list
#                     - containing the sectors in which the target has been observed
#                 - `tess_mags`
#                     - list
#                     - contains the tess-magnitudes of the target for each sector           
#                 - `quality_expression`
#                     - str, optional
#                     - string containing some boolean condition w.r.t. `datum.quality`
#                     - `eval()` function will be applied to construct a boolean array
#                     - the default is `"(datum.quality == 0)"`
#                 - `aperture_detail`
#                     - int, optional
#                     - how highly resolved the outline of the aperture should be depicted
#                     - the default is 20
#                 - `ylims`
#                     - list, optional
#                     - list of the same length as `sectors`
#                         - contains list of ylims for the respective sector
#                     - the default is `None`
#                         - uses matplotlib default ylims
#                 - `fontsize`
#                     - int, optional
#                     - a measure for the fontsize used in the plot
#                         - fontsize of title, ticklabels etc. are scaled with respect to this value as well
#                     - the default is 16
#                 - `figsize`
#                     - tuple, optional
#                     - dimensions of the created figure
#                     - the default is `(16,9)`
#                 - `save`
#                     - str, bool, optional
#                     - whether to save the created image
#                     - when a location is given (i.e. a string), the image will be saved to that location
#                     - the default is `False`

#             Raises
#             ------

#             Returns
#             -------
#                 - `fig`
#                     - Figure
#                     - created figure
#                 - `axs`
#                     - plt.Axes
#                     - axes corresponding to `fig`

#             Comments
#             --------
#         """

#         if ylims is None:
#             ylims = [False]*len(sectors)
#         assert len(ylims) == len(sectors), f"'ylims' has to be 'None' or of the same shape as 'sectors'!"

#         fig = plt.figure(figsize=figsize)
#         fig.suptitle(f"TIC {tic}", fontsize=fontsize+6)

        
#         for idx, (datum, sector, tess_mag, ylim) in enumerate(zip(data, sectors, tess_mags, ylims)):
            
#             #add subplots
#             row = len(sectors)
#             column = 2
#             pos = (idx+1)*2
#             ax2 = fig.add_subplot(row, column, pos)
#             ax1 = fig.add_subplot(row, column, pos-1)

#             #TPFs
#             aperture_mask = datum.aperture.copy()
#             aperture_mask[(aperture_mask > 0)] = 1
#             aperture_plot = np.kron(aperture_mask, np.ones((aperture_detail, aperture_detail)))
#             extent = [0-0.5, datum.aperture.shape[1]-0.5, 0-0.5, datum.aperture.shape[0]-0.5]

#             ##plot TPF
#             ax1.imshow(datum.tpf[0])
            
#             ##plot aperture
#             ax1.contour(aperture_plot, colors='r', corner_mask=False, origin='lower', aspect='auto', extent=extent, zorder=2)

#             #TODO: Maybe check here if extraction worked (only one sector not working?)
#             q = eval(quality_expression)
#             ax2.plot(datum.time[q], datum.corr_flux[q]/np.nanmedian(datum.corr_flux[q]), marker=".", linestyle="", color="tab:blue", label=f"Corrected Flux")
#             try:
#                 ax2.plot(datum.time[q], datum.pca_flux[q]/np.nanmedian(datum.pca_flux[q]), marker=".", linestyle="", color="tab:orange", label=f"PCA")
#             except:
#                 pass
#             try:
#                 ax2.plot(datum.time[q], datum.psf_flux[q]/np.nanmedian(datum.psf_flux[q]), marker=".", linestyle="", color="tab:green", label=f"PSF")
#             except:
#                 pass
            
            
#             #add legends
#             leg = ax2.legend(
#                 fontsize=fontsize-2,
#                 title=r"$\mathbf{Sector~%i}$, TESS_mag: %.2f"%(sector, tess_mag), title_fontsize=fontsize-2,
#                 loc="upper right",
#             )
#             leg._legend_box.align = "left"
            
#             aperture_patch = Line2D([0], [0], color="r", label="Chosen Aperture", linewidth=1)
#             imlegend = ax1.legend(handles=[aperture_patch])

#             #label

#             if ylim != False:
#                 ax2.set_ylim(*ylim)

#             if idx == len(data)-1:
#                 #only label bottom axis
#                 ax2.set_xlabel("Time [BJD - 2457000]", fontsize=fontsize)
#                 ax1.set_xlabel("Pixel", fontsize=fontsize)
#             if pos == 0:
#                 ax1.set_title("Target Pixel Files", fontsize=fontsize+4)
#             ax1.set_ylabel("Pixel", fontsize=fontsize)

#             ax1.tick_params("both", labelsize=fontsize)
#             ax2.tick_params("both", labelsize=fontsize)
#             ax2.yaxis.set_major_formatter(plt.FormatStrFormatter("%.1f"))    #only use 1 decimal place for tick-labels


#         #add common label to second column
#         ax01 = fig.add_subplot(122, frameon=False)   #big hidden axis
#         ax01.tick_params(labelcolor='none', which='both', labelsize=fontsize, top=False, bottom=False, left=False, right=False) #hide ticks and ticklabels
#         ax01.set_ylabel("Normalized Flux", fontsize=fontsize)    #set axis-label
        
#         plt.tight_layout()

#         #save if specified
#         if type(save) == str:
#             plt.savefig(f'{save}tic{tic}.csv', dpi=180)
#         plt.show()
        
#         axs = fig.axes

#         return fig, axs
   
#     def result2pandas(self,
#         data:list, sectors:list, tess_mags:list, tic:str,
#         quality_expression:str="(datum.quality == 0)",
#         sep=";",
#         include_aperture=False, include_tpf=False,
#         save:Union[str,bool]=False,
#         ) -> pd.DataFrame:
#         """
#             - method to convert the result returned by eleanor to a pandas DataFrame
#             - also allows saving of the created DataFrame as .csv

#             Parameters
#             ----------
#                 - `data`
#                     - list
#                     - contains the data for each sector
#                         - extracted with eleanor
#                 - `sectors`
#                     - list
#                     - containing the sectors in which the target has been observed
#                 - `tess_mags`
#                     - list
#                     - contains the tess-magnitudes of the target for each sector           
#                 - `tic`
#                     - str
#                     - TIC identifier of the target
#                 - `quality_expression`
#                     - str, optional
#                     - string containing some boolean condition w.r.t. `datum.quality`
#                     - `eval()` function will be applied to construct a boolean array
#                     - the default is `"(datum.quality == 0)"`
#                 - `sep`
#                     - str, optional
#                     - separator to use when creating the .csv file
#                     - the default is `";"`
#                         - reason for choosing `";"` over `","` is that aperture and tpf will be stored as nested list, which contain `","` as separators
#                 - `include_aperture`
#                     - bool, optional
#                     - whether to include the used aperture in the extracted file
#                     - will store the aperture for every single frame
#                         - thus, can lead to quite large files
#                     - the default is `False`
#                 - `include_tpf`
#                     - bool, optional
#                     - whether to include the target-pixel-files in the extracted file
#                     - will store the tpf for every single frame
#                         - thus, can lead to quite large files
#                 - `save`
#                     - str, bool, optional
#                     - path to the directory of where to store the created csv-file
#                     - the default is `False`
#                         - will not save results to .csv file
                
#             Raises
#             ------

#             Returns
#             -------
#                 - `df_lc`
#                     - pd.DataFrame
#                     - contains all the information extracted for the target
#                     - i.e. various fluxes, times, sectors, tess magnitude, ...

#             Comments
#             --------

#         """

#         df_lc = pd.DataFrame(
#             columns=["time", "raw_flux", "corr_flux", "pca_flux", "psf_flux", "sector", "q_eleanor", "tess_mag"]+[ "aperture"]*include_aperture+["tpf"]*include_tpf
#             )
#         if include_aperture:
#             df_lc["aperture"].astype(object)
#         if include_tpf:
#             df_lc["tpf"].astype(object)

#         for idx, (datum, sector, tess_mag) in enumerate(zip(data, sectors, tess_mags)):

#             #if TargetData() did not fail (i.e. if the sector was successfully extracted)
#             if datum is not None:
#                 q = eval(quality_expression)

#                 if self.do_pca:
#                     pca_flux = datum.pca_flux[q]
#                 else:
#                     pca_flux = [None]*len(datum.time[q])
#                 if self.do_psf:
#                     psf_flux = datum.pca_flux[q]
#                 else:
#                     psf_flux = [None]*len(datum.time[q])

#                 df_datum = pd.DataFrame({
#                     "time":datum.time[q],
#                     "raw_flux":datum.raw_flux[q],
#                     "corr_flux":datum.corr_flux[q],
#                     "pca_flux":pca_flux,
#                     "psf_flux":psf_flux,
#                     "sector":[sector]*len(datum.time[q]),
#                     "q_eleanor":datum.quality[q],
#                     "tess_mag":[tess_mag]*len(datum.time[q]),
#                 })

#                 if include_aperture:
#                     df_datum["aperture"] = [datum.aperture]*len(datum.time[q])
#                 if include_tpf:
#                     df_datum["tpf"] = datum.tpf[q].tolist()
            
#             #insert row of np.nan if the sector failed
#             else:
#                 df_datum = pd.DataFrame({
#                     "time":[np.nan],
#                     "raw_flux":[np.nan],
#                     "corr_flux":[np.nan],
#                     "pca_flux":[np.nan],
#                     "psf_flux":[np.nan],
#                     "sector":[sector],
#                     "q_eleanor":[np.nan],
#                     "tess_mag":[tess_mag],
#                 })                


#             df_lc = pd.concat((df_lc, df_datum), ignore_index=True)

#         if isinstance(save, str):
#             df_lc.to_csv(save, index=False, sep=sep)
            

#         return df_lc


#     #DEPRECATED
#     def save_npy_eleanor(self,
#         data:list, sectors:list, tess_mag:float, save:str,
#         target:str, TIC:str=None, GCVS_class:str=None, GCVS_period:float=None, GCVS_RA:str=None, GCVS_DEC:str=None):
#         """
#             - method to save the extracted data into an 0-dimensional np.array

#             Parameters
#             ----------
#                 - data
#                     - list
#                     - contains the data for each sector
#                         - extracted with eleanor
#                 - sectors
#                     - list
#                     - containing the sectors in which the target has been observed
#                 - tess_mags
#                     - list
#                     - contains the tess-magnitudes for each sector            
#                 - save
#                     - str
#                     - location to save the array to
#                     - the default is False
#                 - target
#                     - str
#                     - you want to give your target
#                         - a good idea is to use the common name or some standard identiier
#                 - TIC
#                     - str, optional
#                     - TIC identifier of your target
#                     - the default is None
#                 - GCVS_class
#                     - str, optional
#                     - class assigned to the target in the literature (i.e. GCVS)
#                     - the default is None
#                 - GCVS_period
#                     - float, optional
#                     - period noted in the GCVS for the target
#                     - the default is None
#                 - GCVS_RA
#                     - str, optional
#                     - right ascension noted in the GCVS for the target
#                     - the default is None
#                 - GCVS_DEC
#                     - str, optional
#                     - declination noted in the GCVS for the target
#                     - the default is None

#             Raises
#             ------

#             Returns
#             -------

#             Comments
#             --------
#                 - NOT MAINTAINED ANYMOER
        
#         """

#         warnings.warn("WARNING: save_npy_eleanor is deprecated. It might not be compatible with newer versions of the other methods.")

#         savedict = {
#             "target":target,
#             "TIC":TIC,
#             "tess_mags":tess_mag,
#             "times":np.array([]),
#             "raw_flux":np.array([]),
#             "corr_flux":np.array([]),
#             "pca_flux":np.array([]),
#             "psf_flux":np.array([]),
#             "sectors":np.array([]),
#             "aperture":[],
#             "tpf":[],
#             "GCVS_class":GCVS_class,
#             "GCVS_period":GCVS_period,
#             "GCVS_RA":GCVS_RA,
#             "GCVS_DEC":GCVS_DEC
#         }
#         for idx, (datum, sector) in enumerate(zip(data, sectors)):

#             q = datum.quality == 0
#             savedict["times"] = np.append(savedict["times"], datum.time[q])
#             savedict["raw_flux"] = np.append(savedict["raw_flux"], datum.raw_flux[q])
#             savedict["corr_flux"] = np.append(savedict["corr_flux"], datum.corr_flux[q])
#             try:
#                 savedict["pca_flux"] = np.append(savedict["pca_flux"], datum.pca_flux[q])
#             except:
#                 pass
#             try:
#                 savedict["psf_flux"] = np.append(savedict["psf_flux"], datum.psf_flux[q])
#             except:
#                 pass
#             savedict["sectors"] = np.append(savedict["sectors"], [sector]*(len(datum.raw_flux[q])))
#             savedict["aperture"].append(datum.aperture)
#             savedict["tpf"].append(datum.tpf)
            
#             # print(sector, datum.aperture.shape)

#         # print(len(savedict["aperture"]))
#         # print(savedict.keys())

#         # print(len(savedict["times"]))
#         # print(len(savedict["raw_flux"]))
#         # print(len(savedict["corr_flux"]))
#         # print(len(savedict["pca_flux"]))
#         # print(len(savedict["psf_flux"]))
#         # print(len(savedict["sectors"]))
#         # print(savedict["GCVS_class"])
#         # print(savedict["GCVS_period"])
#         np.save(save, savedict)

#         return

