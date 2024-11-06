
#%%imports
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table, vstack
from astropy import units as u
from astroquery.mast import Tesscut
import glob
from joblib.parallel import Parallel, delayed
import matplotlib.pyplot as plt
from matplotlib import animation as manimation
from matplotlib.figure import Figure
import numpy as np
import requests
import time
from typing import List, Callable, Union, Tuple

from astroLuSt.monitoring import (formatting as almofo, errorlogging as almoer)

#%%classes
class TESScut_Interface:
    """
        - class to provide a parallelized interface to `astroquery.mast.Tesscut`
        - used to download tess TPFs based on coordinates of targets

        Attributes
        ----------
            - `n_jobs`  
                - `int`, optional
                - how many parallel jobs to use for the extraction
                - the default is 1
            - `redownload`
                - `bool`, optional
                - whether to redownload already extracted targets
                - the default is `False`
            - `sleep`
                - `float`, optional
                - time to sleep after extraction of  each target-sector combination
                - necessary to not overload the server with requests
                    - https://mast.stsci.edu/tesscut/docs/getting_started.html:
                        - "TESScut limits each user to no more than 5 requests per second. After a user has reached this limit, TESScut will return a 503 Service Temporarily Unavailable Error."
                - the default is `0`
            - `verbose`
                - `int`, optional
                - verbosity level
                - the default is `0`
            - `get_cutouts_kwargs`
                - `dict`, optional
                - kwargs to pass to `astroquery.mast.Tesscut.get_cutouts()`
                - the default is `None`
                    - will be set to `dict()`                

        Infered Attributes
        ------------------
            - `self.LE`
                - `astroLuSt.monitoring.errorloggin.LogErrors` instance
                - used to log and display caught errors        
        
        Methods
        -------
            - `_check_if_extracted()`
            - `_merge_sectors()`
            - `_get_normalized_frames()`
            - `extract_target()`
            - `download()`
            - `save()`
            - `plot_result()`

        Dependencies
        ------------
            - `astropy`
            - `astroquery`
            - `glob`
            - `joblib`
            - `matplotlib`
            - `requests`
            - `typing`

        Comments
        --------

    """

    def __init__(self,
        n_jobs:int=1,
        redownload:bool=False,
        sleep:float=0,
        verbose:int=0,
        get_cutouts_kwargs:dict=None,
        ) -> None:
        
        self.n_jobs     = n_jobs
        self.redownload = redownload
        self.sleep      = sleep
        self.verbose    = verbose

        if get_cutouts_kwargs is None:  self.get_cutouts_kwargs  = dict()
        else: self.get_cutouts_kwargs = get_cutouts_kwargs

        #infered attributes
        self.LE = almoer.LogErrors(verbose=verbose-2)

        pass

    def __repr__(self) -> str:
        return (
            f'{self.__class__.__name__}(\n'
            f'    n_jobs={repr(self.n_jobs)},\n'
            f'    redownload={repr(self.redownload)},\n'
            f'    sleep={repr(self.sleep)},\n'
            f'    verbose={repr(self.verbose)},\n'
            f')'
        )

    def __dict__(self) -> dict:
        return eval(str(self).replace(self.__class__.__name__, 'dict'))

    def _check_if_extracted(self,
        filename:str,
        savedir:str,
        sectors:List[int],
        combine_sectors:bool,
        verbose:int=None,
        ) -> np.ndarray[bool]:
        """
            - method to check if a target has been extracted already

            Parameters
            ----------
                - `filename`
                    - `str`, optional
                    - name to use for the saved fits-file
                    - the default is `None`
                        - will autogenerate name based on
                            - `targ_id` and `sector`, if available
                            - `coords` and `sector`, otherwise
                - `savedir`
                    - `str`, optional
                    - directory to store the extracted data into
                    - the default is `None`
                        - files will NOT be saved
                - `sectors`
                    - `List[str]`
                    - sectors to be extracted
                - `combine_sectors`
                    - `bool`
                    - whether sectors will be combined into one large file
                - `verbose`
                    - `int`, optional
                    - verbosity level
                    - overrides `self.verbose`
                    - the default is `None`
                        - will fall back to `self.verbose`
            
            Raises
            ------

            Returns
            -------
                - `ext_bool`
                    - `np.ndarray[bool]`
                    - boolean array flagging if a target has not yet been extracted

            Comments
            --------

        """
        #default parameters
        if verbose is None: verbose = self.verbose

        if combine_sectors:
            #still need a boolean flag for any sector
            ext_bool = np.array([len(glob.glob(f'{savedir}{filename}_cutouts.*'))==0 for s in sectors])
        else:
            ext_bool = np.array([len(glob.glob(f'{savedir}{filename}_{s}_cutouts.*'))==0 for s in sectors])

        return ext_bool

    def _merge_sectors(self,
        hdulist_sectors:List[fits.HDUList],
        target_id:str=None,
        ffi_header_keys:List[str]=None,
        prim_header_keys:List[str]=None,
        col_keys:List[str]=None,
        verbose:int=None,
        ) -> fits.HDUList:
        """
            - method to merge a list of `HDUList` instances (i.e. fits files) extracted with this class into one big HDUList

            Parameters
            ----------
                - `hdulist_sectors`
                    - `List[fits.HDUList]`
                    - list containing the extracted `HDUList`s to be merged
                - `target_id`
                    - `str`, optional
                    - some user-defined id to give the target that is extracted
                    - only needed to populate the fits-headers
                    - the default is `None`
                - `ffi_header_keys`
                    - `List[str]`, optional
                    - keys from the header of the original files to include in the header of the merged file (`astropy.io.fits.BinaryHDU()`)
                    - will be extracted from first entry in `hdulist_sectors`
                    - the default is `None`
                        - will be set to `['TELESCOP', 'INSTRUME', 'RADESYS', 'EQUINOX', 'FFI_TYPE', 'RA_OBJ', 'DEC_OBJ', 'BJDREFI', 'BJDREFF', 'TIMEUNIT']`
                - `prim_header_keys`
                    - `List[str]`, optional
                    - keys from the header of the original files to include in the header of the merged files `astropy.io.fits.PrimaryHDU()`
                    - will be extracted from first entry in `hdulist_sectors`
                    - the default is `None`
                        - will be set to `ffi_header_keys`
                - `col_keys`
                    - `List[str]`, optional
                    - additional information of the individual `HDUList()` instances in `hdulist_sectors` to include as columns in the `astropy.io.fits.BinaryHDU()`
                        - i.e. the tpf-unit
                    - the default is `None`
                        - will be set to `['SECTOR']`
                - `verbose`
                    - `int`, optional
                    - verbosity level
                    - overrides `self.verbose`
                    - the default is `None`
                        - will fall back to `self.verbose`

            Raises
            ------

            Returns
            -------
                - `hdul`
                    - `astropy.io.fits.HDUList()`
                    - contains 3 units
                        - `'PRIMARY'`
                            - `PrimaryHDU()`
                            - global information extracted from primary header of first `HDUList()` in `hdulist_sectors`
                        - `'PIXELS'`
                            - `BinTableHDU()`
                            - timeseries of extracted cutouts
                        - `'SECTOR_INFO'`
                            - `BinTableHDU()`
                            - additional information related to the sectors
                            - i.e., exposure used for each sector, camera used for each sector, ccd used for each sectors

            Comments
            --------

        """

        almofo.printf(
            msg='Merging sectors...',
            context=f'{self.__class__.__name__}.{self._merge_sectors.__name__}',
            type='INFO',
            level=1,
            verbose=verbose,
        )

        #default parameters
        if verbose is None: verbose = self.verbose
        ##headers to include as global header to the BinaryTableHDU - extracted from first entry in hdulist_sectors
        if ffi_header_keys is None:
            ffi_header_keys = [
                'TELESCOP', 'INSTRUME',
                'RADESYS', 'EQUINOX', 'FFI_TYPE',
                'RA_OBJ', 'DEC_OBJ',
                'BJDREFI', 'BJDREFF', 'TIMEUNIT',
            ]
        ##headers to include as global header to the PrimaryHDU
        if prim_header_keys is None: prim_header_keys = ffi_header_keys
        ##additional information from the header to include as column/data
        if col_keys is None:
            col_keys = ['SECTOR']
        

        #create BinTableHDU for FFIs
        ##merge all sectors to one fits file    
        hdu1_base = hdulist_sectors[0][1]       #first fits file as template for header
        hdu1_tables = []
        col_attrs = hdu1_base.columns.info(attrib='all', output=False)
        for hs in hdulist_sectors:
            #convert data to tables
            tab_sector = Table(hs[1].data)
            
            #add additional columns based on header
            tab_sector.add_columns(
                [[hs[1].header[ck]]*len(tab_sector) for ck in col_keys],
                names=col_keys,
            )
            hdu1_tables.append(tab_sector)

        ##stack all sectors
        hdu1 = vstack(hdu1_tables)
        
        ##convert to BinTableHDU for output as HDUList()
        hdu1 = fits.BinTableHDU(
            data=hdu1,
            name=hdu1_base.name
        )

        ##make sure all the column attributes are copied to combined table
        for ca in list(col_attrs.keys())[1:]:   #ignore first attribute (name)
            for cn, cav in zip(col_attrs['name'], col_attrs[ca]):
                try:
                    hdu1.columns.change_attrib(col_name=cn, attrib=ca, new_value=cav)
                except Exception as e:
                    self.LE.print_exc(  e, prefix=f'{hdu1_base.header["RA_OBJ"]}, {hdu1_base.header["DEC_OBJ"]} ({target_id})')
                    self.LE.exc2df(     e, prefix=f'{hdu1_base.header["RA_OBJ"]}, {hdu1_base.header["DEC_OBJ"]} ({target_id})')
                    pass

        ##modify header to specifications
        ###update infered headers to contain descriptions
        for hk in hdu1.header.keys():  
            if hk in hdu1_base.header.keys(): #make sure to ignore any columns that were added after the fact
                hdu1.header.set(hk, hdu1.header[hk], hdu1_base.header.comments[hk])
        ###add specified keys from base header
        for hk in ffi_header_keys:          hdu1.header.set(hk, hdu1_base.header[hk], hdu1_base.header.comments[hk])
        ###additional columns
        hdu1.header.append(('TARGNAME', target_id, 'custom target name passed by the user'))
        

        #add PrimaryHDU
        # tmin = Time(hdu1.data.field('TIME').min()+hdu1.header['BJDREFI'], format='jd', scale='tdb')
        # tmax = Time(hdu1.data.field('TIME').max()+hdu1.header['BJDREFI'], format='jd', scale='tdb')
        # print(hdulist_sectors[0][1].header['DATE-OBS'])
        # print(hdulist_sectors[-1][1].header['DATE-END'])
        # print(tmin.fits)
        # print(tmax.fits)

        hdu0 = fits.PrimaryHDU(
            header=fits.Header([
                ('TSTART',  hdu1.data.field('TIME').min(),                                  'observation start time in TJD of first FFI'),
                ('TSTOP',   hdu1.data.field('TIME').max(),                                  'observation stop time in TJD of last FFI'),
                # ('DATE-OBS', tmin.fits, 'TSTART as UTC calendar date of first FFI), #NOTE: Does not match original entry for some reason
                # ('DATE-END', tmin.fits, 'TSTOP as UTC calendar date of last FFI), #NOTE: Does not match original entry for some reason
                ('TELAPSE', hdu1.data.field('TIME').max()-hdu1.data.field('TIME').min(),    '[d] time elapsed by the combined observations'),
                ('NSECTORS',len(hdulist_sectors),                                           'number sectors stored in the file'),
                ('TARGNAME', target_id,                                                     'custom target name passed by the user')
                # ('CREATOR',),
            ]),
        )
        for hk in prim_header_keys: hdu0.header.set(hk, hdu1_base.header[hk], hdu1_base.header.comments[hk])


        #add supplement BinTableHDUs
        hdu2 = fits.BinTableHDU(
            data=Table(
                data=np.array([
                    [hdu[1].header['EXPOSURE'] for hdu in hdulist_sectors],
                    [hdu[1].header['SECTOR'] for hdu in hdulist_sectors],
                    [hdu[1].header['CAMERA'] for hdu in hdulist_sectors],
                    [hdu[1].header['CCD'] for hdu in hdulist_sectors],
                ]).T,
                names=['EXPOSURE', 'SECTOR', 'CAMERA', 'CCD'],
                units=[u.day, '', '', ''],
            ),
            name='SECTOR_INFO'
        )

        #combine to HDUList
        hdul = fits.HDUList([hdu0, hdu1, hdu2])
        almofo.printf(
            msg=hdul.info(output=False),    #show HDUList info as list of strings
            context=f'{self.__class__.__name__}.{self._merge_sectors.__name__}()',
            type='INFO',
            level=1,
            verbose=verbose-1,
        )

        return hdul

    def extract_target(self,
        coords:SkyCoord,
        sectors:List[int]=None,
        target_id:str=None,
        combine_sectors:bool=True,
        nretries:int=3,
        ffi_header_keys:List[str]=None,
        prim_header_keys:List[str]=None,
        col_keys:List[str]=None,        
        return_hduls:bool=True,
        filename:str=None,
        savedir:str=None,
        verbose:int=None,
        get_cutouts_kwargs:dict=None,
        ) -> Union[fits.HDUList, List[fits.HDUList]]:
        """
            - method to extract a single target using `astroquery.mast.Tesscut` based on its coordinates
            - the method will only return the *FIRST* extracted tpf-series in the found results for each element in `sectors`!

            Parameters
            ----------
                - `coords`
                    - `astropy.coordinates.SkyCoord`
                    - coordinates of the target to extract
                - `sectors`
                    - `List[int]`, optional
                    - which sectors to extract
                    - will be passed to `astroquery.mast.Tesscut.get_cutouts()`
                    - the default is `None`
                        - will extract all sectors
                        - sectors extracted using `astroquery.mast.Tesscut.get_sectors()`
                - `target_id`
                    - `str`, optional
                    - some id of the target to extract
                    - will be used instead of `filename`, if `filename` is `None`
                    - the default is `None`
                        - will fall back to `coords` for the filename
                - `combine_sectors`
                    - `bool`, optional
                    - whether to combine the sectors into one large `astropy.io.fits.HDUList()`
                    - if `True`
                        - will merge all sectors into one large `astropy.io.fits.HDUList()` and return that
                    - if `False`
                        - will return the result of each sector as separate `astropy.io.fits.HDUList()`
                        - i.e., the returned parameter will be a list of `astropy.io.fits.HDUList()`
                    - the default is `True`
                - `nretries`
                    - `int`, optional
                    - number of times to try executing `astroquery.mast.Tesscut.get_cutouts()` despite a `requests.exceptions.ConnectionError`
                        - could appear if request is send faster than wifi/network can react
                    - if all retries fail
                        - will return empty result
                    - the default is `3`
                - `ffi_header_keys`
                    - only relevant if `combine_sectors==True`
                    - `List[str]`, optional
                    - keys from the header of the original files to include in the header of the merged file (`astropy.io.fits.BinaryHDU()`)
                    - will be extracted from first entry in `hdulist_sectors`
                    - the default is `None`
                        - will be set to `['TELESCOP', 'INSTRUME', 'RADESYS', 'EQUINOX', 'FFI_TYPE', 'RA_OBJ', 'DEC_OBJ', 'BJDREFI', 'BJDREFF', 'TIMEUNIT']`
                - `prim_header_keys`
                    - only relevant if `combine_sectors==True`
                    - `List[str]`, optional
                    - keys from the header of the original files to include in the header of the merged files `astropy.io.fits.PrimaryHDU()`
                    - will be extracted from first entry in `hdulist_sectors`
                    - the default is `None`
                        - will be set to `ffi_header_keys`
                - `col_keys`
                    - only relevant if `combine_sectors==True`
                    - `List[str]`, optional
                    - additional information of the individual `HDUList()` instances in `hdulist_sectors` to include as columns in the `astropy.io.fits.BinaryHDU()`
                        - i.e. the tpf-unit
                    - the default is `None`
                        - will be set to `['SECTOR']`                    
                - `return_hduls`
                    - `bool`, optional
                    - whether to return the extracted data or not
                    - if set to `False` method will return `None` instead of the data
                    - useful to free some memory in case of large extractions
                    - the default is `True`
                        - will return the extracted data
                - `filename`
                    - `str`, optional
                    - name to use for the saved fits-file
                    - the default is `None`
                        - will autogenerate name based on
                            - `targ_id` and `sector`, if available
                            - `coords` and `sector`, otherwise
                - `savedir`
                    - `str`, optional
                    - directory to store the extracted data into
                    - the default is `None`
                        - files will NOT be saved
                - `verbose`
                    - `int`, optional
                    - verbosity level
                    - overrides `self.verbose`
                    - the default is `None`
                        - will fall back to `self.verbose`
                - `get_cutouts_kwargs`
                    - `dict`, optional
                    - kwargs to pass to `astroquery.mast.Tesscut.get_cutouts()`
                    - the default is `None`
                        - will be set to `dict()`

            Raises
            ------

            Returns
            -------
                - `hdulist`
                    - `astropy.io.fits.HDUList`, `List[astropy.io.fits.HDUList]`
                    - extracted target tpfs in form of a header-data-unit list
                    - `None` in case `self.redownload == False` and target has been extracted already
                    - if `combine_sectors == True`
                        - is a 1-d list with lengths
                            - `len(hdulist)`
                                - number of units in the merged `astropy.io.fits.HDUList()`
                    - if `combine_sectors == False`
                        - is a 2-d list with dimensions
                            - `len(hdulist)`
                                - number of sectors for each target that were extractable
                            - `len(hdulist[i])`
                                - number of units in the origianl `astropy.io.fits.HDUList()`
                    - if `return_hduls == False`
                        - set to `None` to save (a little bit of memory)
            
            Comments
            --------
        """

        #default parameters
        if filename is None:
            if target_id is not None:
                filename = target_id
            else:
                filename = f'{coords.ra.value:+f} {coords.dec.value:+f}'
        if verbose is None:             verbose             = self.verbose
        if get_cutouts_kwargs is None:  get_cutouts_kwargs  = dict()

        #logging
        almofo.printf(
            msg=f'Working on `{coords.ra.value} {coords.dec.value}` ({target_id=}, {self.idx+1:.0f}/{self.n2extract:.0f})',
            context=f'{self.__class__.__name__}.{self.extract_target.__name__}()',
            type='INFO',
            level=0,
            verbose=verbose,
        )
        self.idx += 1   #update counter


        #get sectors
        if sectors is None:
            sectors = Tesscut.get_sectors(
                coordinates=coords,
            )['sector'].data
        
        sectors = np.array(sectors) #convert to np.ndarray
        
        #get boolean flag of what has not been extracted yet
        ext_bool = self._check_if_extracted(
            savedir=savedir,
            filename=filename,
            sectors=sectors,
            combine_sectors=combine_sectors,
            verbose=verbose,
        )
        
        #check if already extracted - ignore if specified
        if not self.redownload:
            almofo.printf(
                msg=f'Ignoring sectors `{sectors[~ext_bool]}` for `{coords.ra.value} {coords.dec.value}` because found in {savedir} and `self.redownload==False`!',
                context=f'{self.__class__.__name__}.{self.extract_target.__name__}()',
                type='INFO',
                level=1,
                verbose=verbose,
            )
            sectors = sectors[ext_bool]

        hdulist_sectors = []        #list to store all extracted hdulists

        #extract for each sector
        for sector in sectors:
            almofo.printf(
                msg=f'Extracting `{sector=}` for `{coords.ra.value} {coords.dec.value}` ({target_id=})',
                context=f'{self.__class__.__name__}.{self.extract_target.__name__}()',
                type='INFO',
                level=1,
                verbose=verbose,
            )

            #get relevant tpfs (cutouts)
            ##try `nretries` times in case of a ConnectionError (usually HTTPSConnectionPoolError)
            for rt in range(nretries):
                almofo.printf(
                    msg=f'Try {rt+1}/{nretries}',
                    context=f'{self.__class__.__name__}.{self.extract_target.__name__}()',
                    type='INFO',
                    level=2,
                    verbose=verbose,
                )                
                try:
                    cutouts = Tesscut.get_cutouts(
                        coordinates=coords,
                        sector=sector,
                        **get_cutouts_kwargs,
                    )
                    if len(cutouts) > 0:
                        break
                except requests.exceptions.ConnectionError as e:
                    if rt+1 == nretries:
                        almofo.printf(
                            msg=f'Failed after {nretries} retries due to {e}',
                            context=f'{self.__class__.__name__}.{self.extract_target.__name__}()',
                            type='WARNING',
                            level=2,
                            verbose=verbose,
                        )                
                    cutouts = []    #return empty sectors
            
            
            #check if sector was extractable
            if len(cutouts) > 0:            
                hdulist = cutouts[0]
                almofo.printf(
                    msg=hdulist,
                    context=f'{self.__class__.__name__}.{self.extract_target.__name__}()',
                    type='INFO',
                    level=1,
                    verbose=verbose-1,
                )                
            
                #add new information to header
                if target_id is not None:
                    hdulist[0].header['TARGNAME'] = target_id
                else:
                    hdulist[0].header['TARGNAME'] = f'{coords.ra.value:+f} {coords.dec.value:+f}'

                #store all hdulists
                hdulist_sectors.append(hdulist)
                
                #save individual sectors as separate files
                if savedir is not None and not combine_sectors:
                    self.save(
                        hdulist=hdulist,
                        filename=filename+f'_{sector}',
                        directory=savedir,
                    )

            #set to empty list if not extractable
            else:
                almofo.printf(
                    msg=f'WARNING: `{sector=}` not found for `{coords.ra.value} {coords.dec.value}` ({target_id=})',
                    context=f'{self.__class__.__name__}.{self.extract_target.__name__}()',
                    type='WARNING',
                    level=1,
                    verbose=verbose,                
                )
                pass

            #sleep to ensure not being kicked off the server
            time.sleep(self.sleep)

        #combine fits into one file
        if combine_sectors and len(hdulist_sectors) > 0:    #ignore if nothing extracted
            #merge extracted tpfs to one large fits-file
            hdulist = self._merge_sectors(
                hdulist_sectors=hdulist_sectors,
                target_id=target_id,
                ffi_header_keys=ffi_header_keys,
                prim_header_keys=prim_header_keys,
                col_keys=col_keys,
                verbose=verbose,
            )
            #save combined file
            if savedir is not None:
                self.save(
                    hdulist=hdulist,
                    filename=filename,
                    directory=savedir,
                )
        else:
            hdulist = hdulist_sectors
        
        #don't return extracted data to save a little bit of memory
        if not return_hduls:
            hdulist = None

        return hdulist

    def download(self,
        coordinates:List[SkyCoord],
        sectors:Union[List[List[int]],List[int]]=None,
        targ_ids:List[str]=None,
        combine_sectors:bool=True,
        nretries:int=3,
        ffi_header_keys:List[str]=None,
        prim_header_keys:List[str]=None,
        col_keys:List[str]=None,
        return_hduls:bool=True,      
        n_jobs:int=None,
        filenames:List[str]=None,
        savedir:str=None,
        verbose:int=None,
        parallel_kwargs:dict=None,
        get_cutouts_kwargs:dict=None,
        ) -> Union[List[fits.HDUList],List[List[fits.HDUList]]]:
        """
            - method to download target pixels files using `astroquery.mast.Tesscut` in a parallelized manner based on coordinates
            - calls `self.extract_target()` in a parallelized loop for every element in `coordinates`

            Parameters
            ----------
                - `coordinates`
                    - `List[SkyCoord]`
                    - coordinates of targets to be extracted
                - `sectors`
                    - `Union[List[int],List[List[int]]]`, optional
                    - sectors to be extracted for the targets
                    - if `List[int]`
                        - will use these sectors for all elements in `coordinates`
                    - if `List[List[int]]`
                        - has to be of same length as `coordinates`
                        - will use the respective set of sectors for the corresponding element in `coordinates`
                    - the default is `None`
                        - will extract all available sectors for every single target
                - `targ_ids`
                    - `List[str]`, optional
                    - ids to assign to the targets for storing to files
                    - will be added to the fits-header of the stored files
                    - will be used for naming the stored files
                    - the default is `None`
                        - will use `coordinates` instead
                - `combine_sectors`
                    - `bool`, optional
                    - whether to combine the sectors into one large `astropy.io.fits.HDUList()`
                        - will be done for each target separately
                    - if `True`
                        - will merge all sectors of any individual target into one large `astropy.io.fits.HDUList()` and return that
                        - the returned parameter will be a list of `astropy.io.fits.HDUList()`
                            - each entry in the list corresponds to one entry in `coordinates`
                    - if `False`
                        - will return the result of each sector of any individual target as separate `astropy.io.fits.HDUList()`
                        - i.e., the returned parameter will be a list of lists of `astropy.io.fits.HDUList()`
                            - each entry in the outer list contains a list of `astropy.io.fits.HDUList()` of each sector
                    - the default is `True`
                - `nretries`
                    - `int`, optional
                    - number of times to try executing `astroquery.mast.Tesscut.get_cutouts()` despite a `requests.exceptions.ConnectionError`
                        - could appear if request is send faster than wifi/network can react
                    - if all retries fail
                        - will return empty result
                    - the default is `3`                
                - `ffi_header_keys`
                    - `List[str]`, optional
                    - only relevant if `combine_sectors==True`
                    - keys from the header of the original files to include in the header of the merged file (`astropy.io.fits.BinaryHDU()`)
                    - will be extracted from first entry in `hdulist_sectors`
                    - the default is `None`
                        - will be set to `['TELESCOP', 'INSTRUME', 'RADESYS', 'EQUINOX', 'FFI_TYPE', 'RA_OBJ', 'DEC_OBJ', 'BJDREFI', 'BJDREFF', 'TIMEUNIT']`
                - `prim_header_keys`
                    - `List[str]`, optional
                    - only relevant if `combine_sectors==True`
                    - keys from the header of the original files to include in the header of the merged files `astropy.io.fits.PrimaryHDU()`
                    - will be extracted from first entry in `hdulist_sectors`
                    - the default is `None`
                        - will be set to `ffi_header_keys`
                - `col_keys`
                    - `List[str]`, optional
                    - only relevant if `combine_sectors==True`
                    - additional information of the individual `HDUList()` instances in `hdulist_sectors` to include as columns in the `astropy.io.fits.BinaryHDU()`
                        - i.e. the tpf-unit
                    - the default is `None`
                        - will be set to `['SECTOR']`         
                - `return_hduls`
                    - `bool`, optional
                    - whether to return the extracted data or not
                    - if set to `False` method will return `None` instead of the data
                    - useful to free some memory in case of large extractions
                    - the default is `True`
                        - will return the extracted data                
                - `n_jobs`
                    - `int`, optional
                    - number of jobs to use for parallelized extraction
                    - overrides `self.n_jobs`
                    - the default is `None`
                        - falls back to `self.n_jobs`
                - `filenames`
                    - `List[str]`, optional
                    - filenames to assign to each target that will be extracted
                    - has to have same length as `coordinates`
                    - the default is `None`
                        - will autogenerate the filenames
                - `savedir`
                    - `str`, optional
                    - directory to store the extracted data to
                    - the default is `None`
                        - files will not be saved
                - `verbose`
                    - `int`, optional
                    - verbosity level
                    - overrides `self.verbose`
                    - the default is `None`
                        - will fall back to `self.verbose`
                - `get_cutouts_kwargs`
                    - `dict`, optional
                    - kwargs to pass to `astroquery.mast.Tesscut.get_cutouts()`
                    - overrides `self.get_cutouts_kwargs`
                    - the default is `None`
                        - will fall back to `self.get_cutouts_kwargs`
                - `parallel_kwargs`
                    - `dict`, optional
                    - kwargs to pass to `joblib.Parallel()`
                    - the default is `None`
                        - will be set to `dict()`
            
            Raises
            ------

            Returns
            -------
                - `hdulists`
                    - `List[astropy.io.fits.HDUList]`, `List[List[astropy.io.fits.HDUList]]`
                    - extracted target tpfs in form of a header-data-unit list
                    - if `combine_sectors == True`
                        - is a 2-d list with lengths
                            - `len(hdulists)`
                                - number of targets that got extracted
                                - same length as coords, if everything was extractable
                            - `len(hdulists[i])`
                                - number of units in the merged `astropy.io.fits.HDUList()`
                    - if `combine_sectors == False`
                        - is a 3-d list with dimensions
                            - `len(hdulists)`
                                - number of targets that got extracted
                                - same length as coords, if everything was extractable
                            - `len(hdulists[i])`
                                - number of sectors for each target that were extractable
                            - `len(hdulists[i][j])`
                                - number of units in the origianl `astropy.io.fits.HDUList()`
                    - if `return_hduls == False`
                        - will be set to list of `None`
                                
            Comments
            --------

        """
        
        #default parameters
        if sectors is None:                 sectors             = [None]*len(coordinates)
        elif isinstance(sectors[0], int):   sectors             = [sectors]*len(coordinates)
        if targ_ids is None:                targ_ids            = [None]*len(coordinates)
        if n_jobs is None:                  n_jobs              = self.n_jobs
        if filenames is None:               filenames           = [None]*len(coordinates)
        if verbose is None:                 verbose             = self.verbose
        if parallel_kwargs is None:         parallel_kwargs     = dict(verbose=verbose)
        if get_cutouts_kwargs is None:      get_cutouts_kwargs  = self.get_cutouts_kwargs

        self.idx = 0
        self.n2extract = len(coordinates)
        
        hdulists = Parallel(n_jobs, **parallel_kwargs)(
            delayed(self.extract_target)(
                coords=coords,
                sectors=secs,
                target_id=target_id,
                combine_sectors=combine_sectors,
                nretries=nretries,
                ffi_header_keys=ffi_header_keys,
                prim_header_keys=prim_header_keys,
                col_keys=col_keys,
                return_hduls=return_hduls,
                filename=filename,
                savedir=savedir,
                verbose=verbose,
                get_cutouts_kwargs=get_cutouts_kwargs,
            ) for coords, secs, target_id, filename in zip(coordinates, sectors, targ_ids, filenames)
        )

        return hdulists

    def save(self,
        hdulist:fits.HDUList,
        filename:str,
        directory:str,
        ) -> None:
        """
            - method to save the extracted data

            Parameters
            ----------
                - `hdulist`
                    - `astropy.io.fits.HDUList`
                    - header-data-unit list to be saved to a .fits file
                - `filename`
                    - `str`
                    - filename to use for saving the file
                - `directory`
                    - `str`
                    - directory to save the file into

            Raises
            ------

            Retuns
            ------

            Comments
            --------
                - implemented separately in case of more complex saving routines in the future
        """
        
        hdulist.writeto(f'{directory}{filename}_cutouts.fits', overwrite=True)


        return

    def plot_result(self,
        hdulist:fits.HDUList,
        fig:Figure=None,
        animate:bool=True,
        pcolormesh_kwargs:dict=None,
        sctr_kwargs:dict=None,
        func_animation_kwargs:dict=None,
        ) -> Tuple[Figure, plt.Axes, manimation.FuncAnimation]:
        """
            - method to plot the extracted result

            Parameters
            ----------  
                - `hdulist`
                    - `astropy.io.fits.HDUList`
                    - one particular extractd hdulist to plot
                    - has to contain the following fields in the header-data-unit at index 1
                        - `'TIME'`
                        - `'FLUX'`
                - `fig`
                    - `matplotlib.figure.Figure`, optional
                    - figure to plot into
                    - the defaul is `None`
                        - will create a new figure
                - `animate`
                    - `bool`, optional
                    - whether to also generate an animation of the extracted frames
                    - the default is `True`
                - `pcolormesh_kwargs`
                    - `dict`, optional
                    - kwargs to pass to `ax.pcolormesh()`
                    - the default is `None`
                        - will be set to `dict(vmin=np.nanmin(tpfs), vmax=np.nanmax(tpfs))`
                        - will generate `vmin` and `vmax` based on maximum and minimum value of tpfs in `hdulist`
                - `sctr_kwargs`
                    - `dict`, optional
                    - kwargs to pass to `ax.scatter()`
                    - the default is `None`
                        - will be set to `dict(cmap='nipy_spectral')`
                - `func_animation_kwargs`
                    - `dict`, optional
                    - kwargs to pass to `matplotlib.animation.FuncAnimation()`
                    - the default is `None`
                        - will be set to `dict()`
                
            Raises
            ------

            Returns
            -------
                - `fig`
                    - `matplotlib.figure.Figure`
                    - generated figure
                - `axs`
                    - `plt.Axes`
                    - axes corresponding to `fig`
                - `anim`
                    - `matplotlib.animation.FuncAnimation`
                    - generated animation
                    - returns `None` if `animate==False`

            Comments
            --------

        """

        def update(
            frame:int,
            ) -> None:
            ax1.set_title(
                f"{targname}\n"
                f"{times[frame]:.2f} [{hdulist[1].header['TUNIT1']}], sector: {sectors[frame]}"
            )
            
            #update frame
            mesh.update(dict(array=tpfs[frame]))
            
            return


        # print(hdulist[1].header["TIMEUNIT"])

        #default parameters
        if pcolormesh_kwargs is None:       pcolormesh_kwargs       = dict()
        if sctr_kwargs is None:             sctr_kwargs             = dict()
        if 'cmap' not in sctr_kwargs.keys():sctr_kwargs['cmap']     = 'nipy_spectral'
        if func_animation_kwargs is None:   func_animation_kwargs   = dict()

        #extract relevant quantities
        times= hdulist[1].data.field('TIME')
        tpfs = hdulist[1].data.field('FLUX')
        try:    sectors = hdulist[1].data.field('SECTOR')
        except: sectors = [hdulist[1].header['SECTOR']]*len(times)
        targname    = hdulist[0].header['TARGNAME']

        if 'vmin' not in pcolormesh_kwargs: pcolormesh_kwargs['vmin'] = np.nanmin(tpfs)
        if 'vmax' not in pcolormesh_kwargs: pcolormesh_kwargs['vmax'] = np.nanmax(tpfs)
        
        #plotting
        if fig is None:
            fig = plt.figure()
        ax1 = fig.add_subplot(111)
        ax1.set_aspect('equal')

        #initialization
        ax1.set_title((
            f"{targname}\n"
            f"{times[0]:.2f} [{hdulist[1].header['TUNIT1']}], sector: {sectors[0]}"
        ))
        #plot frame
        mesh    = ax1.pcolormesh(tpfs[0], **pcolormesh_kwargs)

        ax1.set_xlabel(r'Pixel')
        ax1.set_ylabel(r'Pixel')
        cbar = fig.colorbar(mesh, ax=ax1)
        cbar.set_label(r'Flux $\left[\frac{e^-}{s}\right]$')
                
        fig.tight_layout()

        if animate:
            anim = manimation.FuncAnimation(
                fig,
                func=update,
                fargs=None,
                **func_animation_kwargs
            )
        else:
            anim = None
        
        axs = fig.axes

        return fig, axs, anim

