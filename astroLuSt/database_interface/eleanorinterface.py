
#TODO: document custom_aperture_kwargs

#%%imports
import eleanor
import glob
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
import shutil
import time
from typing import Union, Tuple, Callable, List
import warnings

from astroLuSt.monitoring import errorlogging as alme
from astroLuSt.monitoring import formatting as almf

#%%catch problematic warnings and convert them to exceptions
w2e1 = r".*The following header keyword is invalid or follows an unrecognized non-standard convention:.*"
warnings.filterwarnings("error", message=w2e1)

class EleanorDatabaseInterface:
    """
        - class functioning as a parallelized wrapper around the `eleanor` package

        Attributes
        ----------
            - `sleep`
                - float, optional
                - time to sleep inbetween downloads
                - necessary to avoid server timeout
                - the default is 0
            - `n_jobs`
                - int, optional
                - number of jobs to use for the parallel extraction
                - will be passed to `joblib.Parallel`
                - the default is -1
                    - will use all avaliable cpus
            - `metadata_path`
                - str, optional
                - path where the metadata is stored
                - the default is `None`
                    - will be set to `'./mastDownload/HLSP'`
            - `clear_metadata`
                - bool, optional
                - whether to clear the metadata after completing the download of one chunk (in `self.download()`)
                - the default is False
            - `redownload`
                - bool, optional
                - whether to redownload already stored targets or ignore them
                - the default is False
            - `verbose`
                - int, optional
                - verbosity level
                - the default is 0

        Infered Attributes
        ------------------
            - `self.LE`
                - astroLuSt.monitoring.errorloggin.LogErrors instance
                - used to log and display caught errors

        Methods
        -------
            - `self.extract_source()`
            - `self.download()`
            - `self.save()`
            - `self.plot_result()`

        Dependencies
        ------------
            - eleanor
            - glob
            - joblib
            - matplotlib
            - numpy
            - pandas
            - shutil
            - time
            - typing
            - warnings

        Comments
        --------

    """

    def __init__(self,
        sleep:float=0,
        n_jobs:int=-1,
        metadata_path:str=None,
        clear_metadata:bool=False,
        redownload:bool=False,
        verbose:int=0,
        ) -> None:
        
        self.sleep          = sleep
        self.n_jobs         = n_jobs
        self.clear_metadata = clear_metadata
        self.redownload     = redownload
        self.verbose        = verbose

        if metadata_path is None: self.metadata_path = './mastDownload/HLSP'

        #infered attributes
        self.LE = alme.LogErrors()


        pass

    def __repr__(self) -> str:
        return (
            f'{self.__class__.__name__}(\n'
            f'    sleep={repr(self.sleep)},\n'
            f'    n_jobs={repr(self.n_jobs)},\n'
            f'    metadata_path={repr(self.metadata_path)},\n'
            f'    clear_metadata={repr(self.clear_metadata)},\n'
            f'    redownload={repr(self.redownload)},\n'
            f'    verbose={repr(self.verbose)},\n'
            f')'
        )
    
    def __dict__(self) -> dict:
        return eval(str(self).replace(self.__class__.__name__, 'dict'))

    def extract_source(self,
        sectors:Union[str,List]=None,
        source_id:dict=None,
        tpfs2store:slice=None, store_aperture_masks:bool=True,
        verbose:int=None,
        multi_sectors_kwargs:dict=None,
        targetdata_kwargs:dict=None,
        custom_aperture_kwargs:dict=None,
        save_kwargs:dict=None,
        ) -> Tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray]:
        """
            - method to extract data for a single source

            Parameters
            ----------
                - `sectors`
                    - str, list, optional
                    - TESS sectors to extract data from
                    - the default is `None`
                        - will be set to `'all'`
                        - i.e. all available sectors will be extracted
                - `source_id`
                    - dict, optional
                    - contains mission and keys available in `eleanor.multi_sectors()`
                    - keys of the dict have to be one of
                        - `'tic'`
                        - `'gaia'`
                        - `'coords'`
                        - `'name'`
                    - values of the dict has to be the corresponding identifier
                    - the default is `None`
                        - will be set to `dict()`
                - `tpfs2store`
                    - slice, optional
                    - which target-pixel-files to store
                    - if you want to store all pass `slice(None)`
                    - the default is `None`
                        - will be set to `slice(0)`
                        - no tpf extracted
                - `store_aperture_masks`
                    - bool, optional
                    - whether to also store aperture-masks
                    - the default is True
                - `verbose`
                    - int, optional
                    - verbosity level
                    - overrides `self.verbose`
                    - the default is `None`
                        - will fall back to `self.verbose`
                - `multi_sectors_kwargs`
                    - dict, optional
                    - kwargs to pass to `eleanor.multi_sectors()`
                    - the default is `None`
                        - will be set to `dict()`
                - `targetdata_kwargs`
                    - dict, optional
                    - kwargs to pass to `eleanor.TargetData()`
                    - the default is `None`
                        - will be set to `dict()`
                - `custom_aperture_kwargs`
                    - dict, optional
                    - kwargs if one wants to use a custom aperture
                    - will override the automatically determined aperture
                    - for the source of the `custom_aperture` method within eleanor see
                        - https://github.com/afeinstein20/eleanor/blob/4a5eceb71cffb9a3d1b44aa24db0c1aa0524df41/eleanor/targetdata.py#L1090
                        - last accessed: 2023/11/28
                    - the default is `None`
                        - will not use a custom aperture
                - `save_kwargs`
                    - dict, optional
                    - kwargs to pass to `self.save()`
                    - the default is `None`
                        - will be set to `dict(filename='_'.join([''.join(item) for item in source_id.items()], directory='./')`

            Raises
            ------

            Returns
            -------
                - `lcs`
                    - np.ndarray
                    - contains lightcurve related quantities that got extracted
                    - will be of shape `(nobservations,nparameters)`
                        - axis 1: contains extracted parameters
                            - `'time'`
                            - `'raw_flux'`
                            - `'flux_err'`
                            - `'corr_flux'`
                            - `'quality'`
                            - `'sector'`
                            - `'tess_mag'`
                            - `'aperture_size'`
                            - `'do_pca'`
                                - only if specified within `targetdata_kwargs`
                            - `'do_psf'`
                                - only if specified within `targetdata_kwargs`
                - `headers`
                    - np.ndarray
                    - contains headers (column names) corresponding to `lcs`
                - `tpfs`
                    - np.ndarray
                    - tpfs that got extracted
                    - will have shape `(ntpfs,xpix,ypix,1)`
                        - `ntpfs` denotes the number of tpfs that got extracted
                        - `xpix` is the pixel coordinate in x direction
                        - `ypix` is the pixel coordinate in y direction
                        - last dimension contains flux values
                - `aperture_masks`
                    - np.ndarray
                    - boolean arrays
                    - contains aperture-masks for every sector
                    - will have shape `(nsectors,xpix,ypix,1)`
                        - `nsectors` denotes the number of sectors that got extracted
                        - `xpix` is the pixel coordinate in x direction
                        - `ypix` is the pixel coordinate in y direction
                        - last dimension contains boolean values

            Comments
            --------
        """

        if sectors is None:                 sectors                 = 'all'
        if source_id is None:               source_id               = dict()
        if tpfs2store is None:              tpfs2store              = slice(0) 
        if verbose is None:                 verbose                 = self.verbose
        if multi_sectors_kwargs is None:    multi_sectors_kwargs    = dict()
        if targetdata_kwargs is None:       targetdata_kwargs       = dict()
        
        if save_kwargs is None: save_kwargs_use = None
        else:
            save_kwargs_use = save_kwargs.copy()
        if save_kwargs_use is not None:
            if 'filename' not in save_kwargs_use.keys():
                save_kwargs_use['filename'] = '_'.join([''.join(item) for item in source_id.items()])
            if 'directory' not in save_kwargs_use.keys():
                save_kwargs_use['directory'] = './'
        

        
        #define storage structures
        lcs = []
        tpfs = []
        aperture_masks = []
        headers = np.array(['time', 'raw_flux', 'flux_err', 'corr_flux', 'quality', 'sector', 'tess_mag', 'aperture_size'])
        if 'do_pca' in targetdata_kwargs.keys(): headers = np.append(headers, ["pca_flux"]*targetdata_kwargs['do_pca'])
        if 'do_psf' in targetdata_kwargs.keys(): headers = np.append(headers, ["psf_flux"]*targetdata_kwargs['do_psf'])

        #check if redownload is wished and target alread got extracted in the past
        if not self.redownload and len(glob.glob(f"{save_kwargs_use['directory']}{save_kwargs_use['filename']}.*")) > 0:
            almf.printf(
                msg=f'Ignoring {source_id} because found in {save_kwargs_use["directory"]} and `self.redownload==False`!',
                context=f'{self.__class__.__name__}.{self.extract_source.__name__}()',
                type='INFO',
                verbose=verbose
            )

            #return nothing
            lcs = np.empty((0,len(headers)))
            return lcs, headers, tpfs, aperture_masks


        #extract data
        ##test if overall failure
        try:
            #obtain sources
            star = eleanor.multi_sectors(
                sectors=sectors,
                **source_id,
                **multi_sectors_kwargs,
            )

            for idx, s in enumerate(star):
                
                #test if failure in sector
                try:
                    datum = eleanor.TargetData(source=s, **targetdata_kwargs)

                    #use custom aperture
                    if custom_aperture_kwargs is not None:
                        eleanor.TargetData.custom_aperture(datum, **custom_aperture_kwargs)

                    aperture_size = datum.aperture.sum()

                    lc = np.array([
                        datum.time,
                        datum.raw_flux, datum.flux_err,
                        datum.corr_flux,
                        datum.quality,
                        [s.sector]*datum.time.shape[0],
                        [s.tess_mag]*datum.time.shape[0],
                        # [datum.aperture_size]*datum.time.shape[0],
                        [aperture_size]*datum.time.shape[0],
                    ]).T

                    if datum.pca_flux is not None:
                        lc = np.append(lc, np.expand_dims(datum.pca_flux,1), axis=1)
                    if datum.psf_flux is not None:
                        lc = np.append(lc, np.expand_dims(datum.psf_flux,1), axis=1)

                    lcs.append(lc)
                    # if tpfs2store is not None: tpfs.append(datum.tpf[tpfs2store])
                    tpfs.append(datum.tpf[tpfs2store])
                    if store_aperture_masks:    aperture_masks.append(datum.aperture)

                except Exception as e:
                    #log and try next sector
                    self.LE.print_exc(e, prefix=f'{source_id}', suffix=f'sector {s.sector}')
                    self.LE.exc2df(e, prefix=f'{source_id}', suffix=f'sector {s.sector}')

            #concatenate and transform to arrays
            ##expand dimensions to get (nframes,xpix,ypix,flux)
            lcs             = np.concatenate(lcs, axis=0)
            tpfs            = np.concatenate(np.expand_dims(tpfs, axis=-1), axis=0)
            aperture_masks  = np.array(np.expand_dims(aperture_masks, axis=-1))

            if save_kwargs_use is not None:
                self.save(
                    df=pd.DataFrame(data=lcs, columns=headers),
                    **save_kwargs_use,
                )

            #sleep to prevent server timeout
            time.sleep(self.sleep)
        
        except Exception as e:
            #log and return empty result
            lcs = np.empty((0,len(headers)))

            self.LE.print_exc(e, prefix=f'{source_id}', suffix=None,)
            self.LE.exc2df(e, prefix=f'{source_id}', suffix=None,)

        return lcs, headers, tpfs, aperture_masks
    
    def download(self,
        sectors:Union[str,list]=None,
        source_ids:List[dict]=None,
        tpfs2store:slice=None, store_aperture_masks:bool=True,
        n_chunks:int=1,
        verbose:int=None,
        parallel_kwargs:dict=None,
        multi_sectors_kwargs:dict=None,
        targetdata_kwargs:dict=None,
        custom_aperture_kwargs:dict=None,
        save_kwargs:dict=None,
        ) -> Tuple[list,list,list,list]:
        """
            - method to execute a parallel download of a list of targets

            Parameters
            ----------
                - `sectors`
                    - str, list, optional
                    - TESS sectors to extract data from
                    - the default is `None`
                        - will be set to `'all'`
                        - i.e. all available sectors will be extracted
                - `source_ids`
                    - list, optional
                    - contains dicts
                        - dicts contain mission and keys available in `eleanor.multi_sectors()`
                        - keys of the dict have to be one of
                            - `'tic'`
                            - `'gaia'`
                            - `'coords'`
                            - `'name'`
                        - values of the dict has to be the corresponding identifier
                    - the default is `None`
                        - will be set to `[]`
                - `tpfs2store`
                    - slice, optional
                    - which target-pixel-files to store
                    - if you want to store all pass `slice(None)`
                    - the default is `None`
                        - will be set to `slice(0)`
                        - no tpf extracted
                - `store_aperture_masks`
                    - bool, optional
                    - whether to also store aperture-masks
                    - the default is True
                - `n_chunks`
                    - int, optional
                    - number of chuks to divide the data into
                    - similar to batch-size in machine learning
                    - the default is 1
                        - all data processed in one go
                - `verbose`
                    - int, optional
                    - verbosity level
                    - overrides `self.verbose`
                    - the default is `None`
                        - will fall back to `self.verbose`
                - `parallel_kwargs`
                    - dict, optional
                    - kwargs to pass to `joblib.Parallel()`
                    - the default is `None`
                        - will be set to `dict(n_jobs=self.n_jobs, backend='threading', verbose=verbose)`
                - `multi_sectors_kwargs`
                    - dict, optional
                    - kwargs to pass to `eleanor.multi_sectors()`
                    - the default is `None`
                        - will be set to `dict()`
                            - done within `self.extract_source()`
                - `targetdata_kwargs`
                    - dict, optional
                    - kwargs to pass to `eleanor.TargetData()`
                    - the default is `None`
                        - will be set to `dict()`
                            - done within `self.extract_source()`
                - `custom_aperture_kwargs`
                    - dict, optional
                    - kwargs if one wants to use a custom aperture
                    - will override the automatically determined aperture
                    - for the source of the `custom_aperture` method within eleanor see
                        - https://github.com/afeinstein20/eleanor/blob/4a5eceb71cffb9a3d1b44aa24db0c1aa0524df41/eleanor/targetdata.py#L1090
                        - last accessed: 2023/11/28
                    - the default is `None`
                        - will not use a custom aperture                            
                - `save_kwargs`
                    - dict, optional
                    - kwargs to pass to `self.save()`
                    - the default is `None`
                        - will be set to `dict(filename='_'.join([''.join(item) for item in source_id.items()], directory='./')`
                            - done within `self.extract_source()`

            Raises
            ------

            Returns
            -------
                - `lcs`
                    - list
                    - contains np.ndarray for every extracted source
                        - contain lightcurve related quantities that got extracted
                        - will be of shape `(nobservations,nparameters)`
                            - axis 1: contains extracted parameters
                                - `'time'`
                                - `'raw_flux'`
                                - `'flux_err'`
                                - `'corr_flux'`
                                - `'quality'`
                                - `'sector'`
                                - `'tess_mag'`
                                - `'aperture_size'`
                                - `'do_pca'`
                                    - only if specified within `targetdata_kwargs`
                                - `'do_psf'`
                                    - only if specified within `targetdata_kwargs`
                - `headers`
                    - list
                    - contains np.ndarray for every extracted source
                        - contain headers (column names) corresponding to entries of `lcs`
                - `tpfs`
                    - list
                        - contains np.ndarray for every extracted source
                        - contain tpfs that got extracted
                        - will have shape `(ntpfs,xpix,ypix,1)`
                            - `ntpfs` denotes the number of tpfs that got extracted
                            - `xpix` is the pixel coordinate in x direction
                            - `ypix` is the pixel coordinate in y direction
                            - last dimension contains flux values
                - `aperture_masks`
                    - list
                        - contains np.ndarray for every extracted source
                        - boolean arrays
                        - contain aperture-masks for every sector
                        - will have shape `(nsectors,xpix,ypix,1)`
                            - `nsectors` denotes the number of sectors that got extracted
                            - `xpix` is the pixel coordinate in x direction
                            - `ypix` is the pixel coordinate in y direction
                            - last dimension contains boolean values
            
            Comments
            --------
        """

        #default parameters
        if source_ids is None:  source_ids = []
        if verbose is None:     verbose = self.verbose
        if parallel_kwargs is None:
            parallel_kwargs = dict(n_jobs=self.n_jobs, backend='threading', verbose=verbose)

        #split into chunks
        chunks = np.array_split(source_ids, n_chunks)

        #init output lists
        lcs             = []
        headers         = []
        tpfs            = []
        aperture_masks  = []

        #iterate over chunks
        extracted = 0
        for cidx, chunk in enumerate(chunks):
            almf.printf(
                msg=f'Extracting chunk {cidx+1}/{len(chunks)} ({extracted}/{len(source_ids)})',
                context=f'{self.__class__.__name__}.{self.download.__name__}()',
                type='INFO',
                verbose=verbose,
            )

            #extract targets (in parallel)
            res = Parallel(**parallel_kwargs)(
                delayed(self.extract_source)(
                    sectors=sectors,
                    source_id=source_id,
                    tpfs2store=tpfs2store, store_aperture_masks=store_aperture_masks,
                    multi_sectors_kwargs=multi_sectors_kwargs,
                    targetdata_kwargs=targetdata_kwargs,
                    custom_aperture_kwargs=custom_aperture_kwargs,
                    save_kwargs=save_kwargs,
                ) for idx, source_id in enumerate(chunk)
            )

            #delete metadata after extraction of each chunck
            if self.clear_metadata:
                try:
                    almf.printf(
                        msg='Removing Metadata...',
                        context=f'{self.__class__.__name__}.{self.download.__name__}()',
                        type='INFO',
                        verbose=verbose,
                    )
                    shutil.rmtree(self.metadata_path)
                except FileNotFoundError as e:                    
                    self.LE.print_exc(e=e, prefix='No Metadata to clear!')
                    self.LE.exc2df(e=e, prefix='No Metadata to clear!')

            #append to output lists
            lcs             += [r[0] for r in res]
            headers         += [r[1] for r in res]
            tpfs            += [r[2] for r in res]
            aperture_masks  += [r[3] for r in res]

        #update number of extracted targets
        extracted += len(chunk)

        return lcs, headers, tpfs, aperture_masks
    
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
                    - pd.DataFrame
                    - dataframe of extracted lc-data (`lcs` from `self.extract_source()`)
                - `filename`
                    - str
                    - name of the file in which the data gets stored
                    - NO FILE EXTENSION!
                - `directory`
                    - str, optional
                    - directory of where the data will be stored
                    - the default is `None`
                        - will be set to `'./'`
                - `pd_savefunc`
                    - str, optional
                    - pandas saving function to use
                        - i.e., methods of pd.DataFrames
                        - examples
                            - `'to_csv'`
                            - `'to_parquet'`
                    - the default is `None`
                        - will be set to `'to_parquet'`
                - `save_kwargs`
                    - dict, optional
                    - kwargs to pass to `pd_savefunc`
                    - the default is `None`
                        - will be set to `dict()`
            Raises
            ------

            Returns
            -------

            Comments
            --------
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

    def plot_result(self,
        lcs:np.ndarray,
        headers:np.ndarray,
        tpfs:np.ndarray=None,
        aperture_masks:np.ndarray=None,
        fig:Figure=None,
        sctr_kwargs:dict=None,
        ) -> Tuple[Figure,plt.Axes]:
        """
            - method to plot the result for one particular target
            
            Parameters
            ----------
                - `lcs`
                    - np.ndarray
                    - extracted data corresponding to the lightcurve
                    - has to be of shape `(nobservations,nquantities)`
                    - output from `self.extract_source`
                - `headers`
                    - np.ndarray
                    - headers/column names corresponding to `lcs` as output by `self.extract_source`
                - `tpfs`
                    - np.ndarray, optional
                    - exemplary target pixel files for each sector
                    - has to be of shape `(nsectors,xpix,ypix,1)`
                    - the default is `None`
                        - will be ignored in the plot
                - `aperture_masks`
                    - np.ndarray, optional
                    - aperture masks for each sector
                    - has to be of shape `(nsectors,xpix,ypix,1)`
                    - the default is `None`
                        - will be ignored in the plot
                - `fig`
                    - Figure, optional
                    - figure to plot into
                    - the default is `None`
                        - will create a new figure
                - `sctr_kwargs`
                    - dict, optional
                    - kwargs to pass to `ax.scatter()`
                    - the default is `None`
                        - will be set to `dict(cmap='nipy_spectral')`

            Raises
            ------

            Returns
            -------
                - `fig`
                    - Figure
                    - created figure
                - `axs`
                    - plt.Axes
                    - axes corresponding to `fig`

            Comments
            --------
        """

        #default parameters
        sectors = np.unique(lcs[:,(headers=='sector')])
        if tpfs is None:            tpfs            = [None]*len(sectors)
        if aperture_masks is None:  aperture_masks  = [None]*len(sectors)
        if sctr_kwargs is None:     sctr_kwargs     = dict(cmap='nipy_spectral')


        if fig is None: fig = plt.figure(figsize=(16,16))

        for idx, (s, tpf, ap) in enumerate(zip(sectors, tpfs, aperture_masks)):
            
            #boolean of current sector
            s_bool = (lcs[:,(headers=='sector')] == s).flatten()
            

            #plot tpf and aperture (only if one of them is provided)
            ##add ax
            if tpf is not None or ap is not None:
                ax1 = fig.add_subplot(len(sectors)+1, 2, 2*idx+1)
                if tpf is not None: 
                    mesh = ax1.pcolormesh(tpf[:,:,0])
                    #add colorbar
                    cbar = fig.colorbar(mesh, ax=ax1)
                    cbar.set_label(r'Flux $\left[\frac{e^-}{s}\right]$')
                #plot aperture
                if ap is not None:
                    mesh_ap = ax1.pcolormesh(ap[:,:,0], zorder=2, edgecolor='r', facecolors='none')
                    mesh_ap.set_alpha(ap)
                    ax1.plot(np.nan, np.nan, '-r', label='Aperture')
                
                #force square plot for tpf
                ax1.set_aspect('equal', adjustable='box')

                if idx == 0:
                    ax1.legend()
                ax1.set_ylabel('Pixel')
                if idx == len(sectors)-1:
                    ax1.set_xlabel('Pixel')
            else:
                pass

            
            #plot lc
            ##add ax
            if tpf is None and ap is None:
                ax2 = fig.add_subplot(len(sectors)+1, 1, idx+1)
            else:
                ax2 = fig.add_subplot(len(sectors)+1, 2, 2*idx+2)
            sctr = ax2.plot(lcs[s_bool,(headers=='time')], lcs[s_bool,(headers=='raw_flux')]/np.nanmedian(lcs[s_bool,(headers=='raw_flux')]), label='Raw Flux'*(idx==0))
            try: sctr = ax2.plot(lcs[s_bool,(headers=='time')], lcs[s_bool,(headers=='corr_flux')]/np.nanmedian(lcs[s_bool,(headers=='corr_flux')]), label='Corr Flux'*(idx==0))
            except: pass
            try: sctr = ax2.plot(lcs[s_bool,(headers=='time')], lcs[s_bool,(headers=='pca_flux')]/np.nanmedian(lcs[s_bool,(headers=='pca_flux')]), label='PCA Flux'*(idx==0))
            except: pass
            try: sctr = ax2.plot(lcs[s_bool,(headers=='time')], lcs[s_bool,(headers=='psf_flux')]/np.nanmedian(lcs[s_bool,(headers=='psf_flux')]), label='PSF Flux'*(idx==0))
            except: pass
        
            

            #labelling
            ax2.legend(title=f'Sector {s:.0f}')

            if idx == len(sectors)-1:
                ax2.set_xlabel('Time [BJD - 2457000]')
            ax2.set_ylabel('Normalized Flux')
            

        #add scatter of all sectors
        ax0 = fig.add_subplot(len(sectors)+1, 1, len(sectors)+1)
        try: 
            sctr = ax0.scatter(lcs[:,(headers=='time')], lcs[:,(headers=='corr_flux')]/np.nanmedian(lcs[:,(headers=='corr_flux')]), c=lcs[:,(headers=='sector')], **sctr_kwargs)
        except:
            sctr = ax0.scatter(lcs[:,(headers=='time')], lcs[:,(headers=='raw_flux')]/np.nanmedian(lcs[:,(headers=='raw_flux')]),   c=lcs[:,(headers=='sector')], **sctr_kwargs)
        cbar = fig.colorbar(sctr, ax=ax0)
        
        cbar.set_label('Sector')
        ax0.set_xlabel('Time [BJD - 2457000]')
        ax0.set_ylabel('Normalized Flux')


        fig.tight_layout()
            

        axs = fig.axes

        return fig, axs

    # def animate(self,
    #     ):
    #     #TODO

    #     return
