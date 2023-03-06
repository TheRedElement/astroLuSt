
#TODO: eleanor: clear metadata after every target has been extracted (not working in parallel)

#%%imports

#data manipulation
import numpy as np
import pandas as pd
import re

#plotting
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.colors as mcolors

#os interaction
import os
import shutil
import glob
import io
import sys

#datetime
import time

#web interaction
from urllib import request

#parallel computing
from joblib import Parallel, delayed

#astronomy
from astropy.table import Table
from astroquery.simbad import Simbad
from astroquery.gaia import Gaia
from joblib import Parallel, delayed
import eleanor
import lightkurve as lk
from alerce.core import Alerce


#warnings
import warnings

#%%catch problematic warnings and convert them to exceptions
w2e1 = r".*The following header keyword is invalid or follows an unrecognized non-standard convention:.*"
warnings.filterwarnings("error", message=w2e1)

#%%

#SIMBAD
class SimbadDatabaseInterface:
    """
        - class to interact with the SIMBAD database

        Attributes
        ----------

        Methods
        -------
            - get_ids()

        Dependencies
        ------------
            - numpy
            - pandas
            - re
            - astroquery
            - joblib
    """
    

    
    def __init__(self):

        self.df_ids = pd.DataFrame()

        pass

        

    def get_ids(self,
        input_ids:list,
        nparallelrequests:int=1000, simbad_timeout:int=120,
        show_scanned_strings_at:list=[],
        verbose:int=0) -> None:
        """
            - function to query the SIMBAD database for additional identifiers

            Parameters
            ----------
                - input_ids
                    - list
                    - list containing all the ids to query for
                    - each entry has to be of the same syntax as the SIMBAD query i.e.
                        - 'tic 114886928'
                        - 'gaia dr3 6681944303315818624'
                - nparallelrequests
                    - int, optional
                    - how many requests to run in parallel
                        - i.e. if your DataFrame has N columns the query will be subdivided into n partitions such that 'nparallelrequest' queries will be executed in parallel
                    - useful for large queries
                    - the default is 1000
                - simbad_timeout
                    - int, optional
                    - the timeout to allow for the SIMBAD database before an error is raised
                    - the default is 120 (seconds)
                - show_scanned_strings_at
                    - list
                    - list of indices to display the strings that get scanned with a regular expression to extract the different identifiers and catalogues
                    - the default is []
                - verbose
                    - int, optional
                    - verbosity level
                    - how much additional information to display while executing the function
                    - the default is 0
            
            Raises
            ------

            Returns
            -------

            Comments
            --------

        """

        unique_ids = pd.unique(input_ids)

        #setup SIMBAD
        my_Simbad = Simbad()
        my_Simbad.TIMEOUT = simbad_timeout
        my_Simbad.add_votable_fields("ids")
        my_Simbad.add_votable_fields("typed_id")
        # print(my_Simbad.list_votable_fields())

        #split the query into chuncks for big queries
        intervals = np.arange(0, unique_ids.shape[0], nparallelrequests)
        if intervals[-1] < unique_ids.shape[0]: intervals = np.append(intervals, unique_ids.shape[0])
        
        for idx, (start, end) in enumerate(zip(intervals[:-1], intervals[1:])):
            
            if verbose > 0:
                print(f"Working on partition {idx+1}/{len(intervals)-1}")

            cur_unique_ids = [uid for uid in unique_ids[start:end]]

            #query SIMBAD
            id_table = my_Simbad.query_objects(cur_unique_ids)

            # print(id_table.keys())
            main_ids = id_table["MAIN_ID"]
            ras = id_table["RA"]
            decs = id_table["DEC"]

            #extract IDs out of the query result
            simbad_result = Parallel(n_jobs=3)(delayed(re.findall)(pattern=r"[^|]+", string=str(id)) for id in id_table["IDS"])
            
            # print(simbad_result)

            for iid, ids, mid, ra, dec in zip(cur_unique_ids, simbad_result, main_ids, ras, decs):
                df_temp = pd.DataFrame()
                df_temp["input_id"] = [iid]
                df_temp["main_id"] = [mid]
                df_temp["ra"] = [ra]
                df_temp["dec"] = [dec]
                for id in ids:
                    
                    if id != mid:
                        catalogue = re.match(r"^.+[^\ ](?=\ )|NPM\d|CSI", id)
                        if catalogue is None:
                            print(f'INFO: catalog is None. Corresponding id: {id}')
                            pass
                        else:
                            id_in_cat = id.replace(catalogue[0], "")
                            df_temp[catalogue[0]] = [id_in_cat]
                            df_temp[catalogue[0]] = df_temp[catalogue[0]].str.strip()

                #update main dataframe
                self.df_ids = pd.concat([self.df_ids, df_temp], ignore_index=True)
            
            
            
            #some verbosity
            if len(show_scanned_strings_at) != 0 and verbose > 1:
                print("Scanned strings:")
                print("----------------")
                for idx in show_scanned_strings_at:
                    print(f"    Scanned Target: {id_table['TYPED_ID'][idx]}")
                    print(f"        Query Result: {id_table['IDS'][idx]}\n")
                print()            
        
        #sort self.df_id alphabetically
        self.df_ids.sort_index(axis=1, inplace=True)
        self.df_ids.insert(0, "input_id", self.df_ids.pop("input_id"))
        
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


#ELEANOR
class EleanorDatabaseInterface:

    """
        - class to interact with the eleanor code for downloading lightcurves

        
        Attributes
        ----------
            - tic
                - list[int], optional
                - list of TIC identifiers
                - the default is []
            - sectors
                - list[int]|str, optional
                - list of sectors to extract or "all" if all available sectors shall be considered
                - the default is "all"
            - do_psf
                - bool, optional
                - whether to execute a psf
                - argument of eleanor.multi_sectors
                - the default is False
            - do_pca
                - bool, optional
                - whether to execute a pca
                - argument of eleanor.multi_sectors
                - the default is False
            - aperture_mode
                - str, optional
                - specify which aperture to use ('small', 'normal', 'large')
                - argument of eleanor.TargetData
                - the default is 'normal'
            - regressors
                - str, optional
                - which methods to use to estimate the background
                - argument of eleanot.TargetData
                - the default is 'corners'
            - try_load
                - bool, optional
                - whether to search hidden ~/.eleanor dictionary for alrady existing TPF
                - argument of eleanor.TargetData
                - the default is True
            - height
                - int, optional
                - pixel height of the cutout
                - argument of eleanor.multi_sectors
                - the default is 15
            - width
                - int, optional
                - pixel width of the cutout
                - argument of eleanor.multi_sectors
                - the default is 15
            - bkg_size
                - int, optional
                - argument of eleanor.multi_sectors
                    - see documentation for more info
                - the default is 31
            - clear_metadata_after_extract
                - bool, optional
                - whether to delete all downloaded metadata after the extraction of any target
                - if set to False, this will result in faster download, if the target is downloaded again
                - the default is False


        
        Methods
        -------
            - data_from_eleanor_alltics()
            - data_from_eleanor()
            - plot_result_eleanor()
            - result2pandas()
            - save_npy_eleanor()
                - not maintained
                - might have compatibility issues

        Dependencies
        ------------
            - pandas
            - numpy
            - matplotlib
            - eleanor
            - warnings

    """



    def __init__(self,
        tics:list=[], sectors:list="all",
        do_psf:bool=False, do_pca:bool=False,
        aperture_mode:str="normal", regressors:str="corner", try_load:bool=True,
        height:int=15, width:int=15, bkg_size:int=31,
        clear_metadata_after_extract:bool=False
        ):


        self.tics = tics
        self.sectors = sectors
        self.do_psf = do_psf
        self.do_pca = do_pca
        self.aperture_mode = aperture_mode
        self.regressors = regressors
        self.try_load = try_load
        self.height = height
        self.width = width
        self.bkg_size = bkg_size
        self.clear_metadata_after_extract = clear_metadata_after_extract

        self.metadata_path = "./mastDownload/HLSP"

        pass

    def data_from_eleanor_alltics(self,
        #saving data
        save:str="./",
        quality_expression:str="(datum.quality == 0)",
        include_aperture:bool=False, include_tpf:bool=False,
        #plotting
        plot_result:bool=True,
        aperture_detail:int=50, ylims:list=None,
        fontsize:int=16, figsize:tuple=(16,9),
        save_plot:str=False,
        sleep:float=0,
        n_jobs:int=-1, n_chunks:int=1,
        verbose:int=2
        ):
        """
            - method to extract data for all ids in 'self.tics'

            Parameters
            ----------
                - save
                    - str|bool, optional
                    - path to the directory of where to store the extracted files
                    - the default is "./"
                        - will save to the current working directory
                - quality_expression
                    - str, optional
                    - string containing some boolean condition w.r.t. 'datum.quality'
                    - eval() function will be applied to construct a boolean array
                    - the default is "(datum.quality == 0)"
                - include_aperture
                    - bool, optional
                    - whether to include the used aperture in the extracted file
                    - will store the aperture for every single frame
                        - thus, can lead to quite large files
                    - the default is False
                - include_tpf
                    - bool, optional
                    - whether to include the target-pixel-files in the extracted file
                    - will store the tpf for every single frame
                        - thus, can lead to quite large files
                    - the default is False
                - plot_result
                    - bool, optional
                    - whether to create a plot of the extracted file
                    - the default is True
                - aperture_detail
                    - int, optional
                    - how detailed the aperture shall be depicted
                    - higher values require more computation time
                    - the default is 50
                - ylims
                    - list, tuple, optional
                    - the ylims of the created plot
                    - the default is None
                        - will automatically adapt the ylims
                - fontsize
                    - int, optional
                    - fontsize to use on the created plot
                    - the default is 16
                - figsize
                    - tuple, optional
                    - size of the created figure/plot
                    - the default is (16,9)
                - save_plot
                    - str, optional
                    - location of where to save the created plot
                    - the default is False
                        - will not save the plot
                - sleep
                    - float, optional
                    - number of seconds to sleep after downloading each target
                    - the default is 0
                        - no sleep at all                
                - n_jobs
                    - int, optional
                    - number of workers to use for parallel execution of tasks
                        - '1' is essentially sequential computaion (useful for debugging)
                        - '2' uses 2 CPUs
                        - '-1' will use all available CPUs
                        - '-2' will use all available CPUs-1
                    - for more info see https://joblib.readthedocs.io/en/latest/generated/joblib.Parallel.html
                    - the default is '-1'
                - n_chunks
                    - int, optional
                    - number of chuncks to split the input data into
                        - will take 'self.tics' and split it into n_chunks more or less equal subarrays
                    - after processing of each chunk, the metadata will get deleted, if 'self.clear_metadata_after_extract' is set to true
                    - it is advisable, though not necessary, to choose n_chunks such that each chunck contains an amount of elements that can be evenly distributed across n_jobs
                    - the default is 1
                        - i.e. process all the data in one go
                - verbose
                    - int, optional
                    - verbosity level
                        - also for joblib.Parallel()
                    - higher numbers will show more information
                    - the default is 2
            
            Raises
            ------

            Returns
            -------

            Comments
            --------


        """
        #TODO: include exect timer + runtime estimation for verbose > 2

        def extraction_onetic(cidx, idx, chunk_len, n_chunks, tic, save_plot, sleep):
            """
                - funtion for parallel execution
            """
            print(f"\nExtracting tic{tic} (chunk {cidx+1}/{n_chunks}, {idx+1}/{chunk_len} = {chunk_len*(cidx)+idx+1}/{len(self.tics)} Total)")

            data, sectors, tess_mags, error_msg = self.data_from_eleanor(
                tic, sectors=self.sectors,
                do_psf=self.do_psf, do_pca=self.do_pca,
                aperture_mode=self.aperture_mode, regressors=self.regressors, try_load=self.try_load,
                height=self.height, width=self.width, bkg_size=self.bkg_size,
                verbose=verbose
            )

            if data is not None:

                #Save data
                if isinstance(save, str):
                    df_res = self.result2pandas(
                        save=f"{save}tic{tic}.csv",
                        data=data, sectors=sectors, tess_mags=tess_mags, tic=tic,
                        quality_expression=quality_expression,
                        include_aperture=include_aperture, include_tpf=include_tpf,
                        sep=";"
                    )

                #Plot data
                if plot_result:
                    if isinstance(save_plot, str): save_plot = f"{save_plot}tic{tic}.png"
                    else: save_plot = False
                    fig, axs = self.plot_result_eleanor(
                        data=data, tic=tic, sectors=sectors, tess_mags=tess_mags,
                        quality_expression=quality_expression,
                        aperture_detail=aperture_detail, ylims=ylims,
                        fontsize=fontsize, figsize=figsize, save=save_plot,                  
                    )
                
                extraction_summary = {"TIC":tic, "success":True, "original error message":""}
            else:
                #append failed extractions
                extraction_summary = {"TIC":tic, "success":False, "original error message":error_msg}         

            #sleep after each target
            time.sleep(sleep)

            return extraction_summary

        #split array into n_chuncks chuncks to divide the load
        chunks = np.array_split(self.tics, n_chunks)
        if verbose > 1:
            print(f"INFO: Extracting {len(chunks)} chuncks with shapes {[c.shape for c in chunks]}")

        df_extraction_summary = pd.DataFrame()
        for cidx, chunk in enumerate(chunks):
            print(f"INFO: Working on chunk {cidx+1}/{len(chunks)} with size {chunk.shape[0]}")
            extraction_summary = Parallel(
                n_jobs=n_jobs, verbose=verbose
            )(
                delayed(extraction_onetic)(
                    cidx, idx, len(chunk), n_chunks, tic, save_plot, sleep=sleep,
                    ) for idx, tic in enumerate(chunk)
            )

            #delete metadata after extraction of each chunck
            if self.clear_metadata_after_extract:
                
                try:
                    print('INFO: Removing Metadata...')
                    shutil.rmtree(self.metadata_path)
                except FileNotFoundError as e:
                    print(f'INFO: No Metadata to clear.')
                    print(f'    Original ERROR: {e}')

            df_extraction_summary_chunck = pd.DataFrame(data=extraction_summary)
            df_extraction_summary = pd.concat([df_extraction_summary, df_extraction_summary_chunck], ignore_index=True)


        #short protocoll about extraction
        if verbose > 1:
            print(f"\nExtraction summary:")
            print(df_extraction_summary)
        return df_extraction_summary

    def data_from_eleanor(self,
        tic:int, sectors:list="all", 
        do_psf:bool=False, do_pca:bool=False, 
        aperture_mode:str="normal", regressors:str="corner", try_load:bool=True,
        height:int=15, width:int=15, bkg_size:int=31,
        verbose:int=2) -> tuple:
        """
            - function to download data using eleonor

            Parameters
            ----------
                - tic
                    - int
                    - TIC identifier number of the target
                - sectors
                    - list of int, str, optional
                    - sectors to search in
                    - argument of eleanor.multi_sectors
                    - the default is "all"
                        - will consider all sectors
                - do_psf
                    - bool, optional
                    - whether to execute a psf
                    - argument of eleanor.multi_sectors
                    - the default is False
                - do_pca
                    - bool, optional
                    - whether to execute a pca
                    - argument of eleanor.multi_sectors
                    - the default is False
                - aperture_mode
                    - str, optional
                    - specify which aperture to use ('small', 'normal', 'large')
                    - argument of eleanor.TargetData
                    - the default is 'normal'
                - regressors
                    - str, optional
                    - which methods to use to estimate the background
                    - argument of eleanot.TargetData
                    - the default is 'corners'
                - try_load
                    - bool, optional
                    - whether to search hidden ~/.eleanor dictionary for alrady existing TPF
                    - argument of eleanor.TargetData
                    - the default is True
                - height
                    - int, optional
                    - pixel height of the cutout
                    - argument of eleanor.multi_sectors
                    - the default is 15
                - width
                    - int, optional
                    - pixel width of the cutout
                    - argument of eleanor.multi_sectors
                    - the default is 15
                - bkg_size
                    - int, optional
                    - argument of eleanor.multi_sectors
                        - see documentation for more info
                    - the default is 31
                - verbose
                    -int, optional
                    - verbosity level
                    - the higher the more information will be shown
                    - the default is 2

            Raises
            ------
            
            Returns
            -------
                - data
                    - list
                    - list of the data extracted
                        - entries are the different sectors
                - sectors
                    - list
                    - list of the sectors observed
                - tess_mags
                    - list
                    - contains the tess-magnitudes for each sector
                - error_msg
                    - str
                    - error message if the extraction fails

            Dependencies
            ------------
                - eleanor

            Comments
            --------

        """

        try:
            error_msg = None

            star = eleanor.multi_sectors(tic=tic, sectors=sectors)
            
            data = []
            sectors = []
            tess_mags = []
            try:
                for s in star:
                    datum = eleanor.TargetData(
                        s, height=height, width=width, bkg_size=bkg_size, 
                        do_psf=do_psf, 
                        do_pca=do_pca,
                        aperture_mode=aperture_mode,
                        try_load=try_load,
                        regressors=regressors,
                    )

                    data.append(datum)

                    #Get sector
                    sectors.append(s.sector)

                    #Get TESS-magnitude
                    tess_mags.append(s.tess_mag)
                    
            except Exception as e:
                if verbose > 1:
                    print("WARNING: Error in eleanor.TargetData()")
                    print(f"ORIGINAL ERROR: {e}")
                data = None
                sectors = None
                tess_mags = None
                if len(str(e)) > 80: e = str(e)[:79] + '...'
                error_msg = f"{'eleanor.TargetData()':25s}: {e}"


        except Exception as e:
            if verbose > 1:
                print("WARNING: Error in eleanor.multi_sectors()")
                print(f"\t Maybe TIC {tic} was not observed with TESS?")
                print(f"\t If you run extract_data() in a loop, try to run it separately on TIC {tic}.")
                print(f"ORIGINAL ERROR: {e}")
            
            data = None
            sectors = None
            tess_mags = None
            if len(str(e)) > 80: e = str(e)[:79] + '...'
            error_msg = f"{'eleanor.multi_sectors()':25s}: {e}"
        
        return data, sectors, tess_mags, error_msg

    def plot_result_eleanor(self,
        data:list, tic:int, sectors:list, tess_mags:list,
        quality_expression:str="(datum.quality == 0)",
        aperture_detail:int=50, ylims:list=None,
        fontsize:int=16, figsize:tuple=(16,9), save:str=False
        ):
        """
            - function to autogenerate a summary-plot of the data downloaded using eleonor
        
            Parameters
            ----------
                - data
                    - list
                    - contains the data for each sector
                        - extracted with eleanor
                - tic
                    - str
                    - TIC identifier of the target
                - sectors
                    - list
                    - containing the sectors in which the target has been observed
                - tess_mags
                    - list
                    - contains the tess-magnitudes of the target for each sector           
                - quality_expression
                    - str, optional
                    - string containing some boolean condition w.r.t. 'datum.quality'
                    - eval() function will be applied to construct a boolean array
                    - the default is "(datum.quality == 0)"
                - aperture_detail
                    - int, optional
                    - how highly resolved the outline of the aperture should be depicted
                    - the default is 20
                - ylims
                    - list, optional
                    - list of the same length as sectors
                        - contains list of ylims for the respective sector
                    - the default is None
                        - uses matplotlib default ylims
                - fontsize
                    - int, optional
                    - a measure for the fontsize used in the plot
                        - fontsize of title, ticklabels etc. are scaled with respect to this value as well
                    - the default is 16
                - figsize  
                    - tuple, optional
                    - dimensions of the created figure
                    - the default is (16,9)
                - save
                    - str, bool, optional
                    - whether to save the created image
                    - when a location is given (i.e. a string), the image will be saved to that location
                    - the default is False

            Raises
            ------

            Returns
            -------

            Dependencies
            ------------
                - matplotlib
                - numpy

            Comments
            --------
        """

        if ylims is None:
            ylims = [False]*len(sectors)
        assert len(ylims) == len(sectors), f"'ylims' has to be 'None' or of the same shape as 'sectors'!"

        fig = plt.figure(figsize=figsize)
        fig.suptitle(f"TIC {tic}", fontsize=fontsize+6)

        
        for idx, (datum, sector, tess_mag, ylim) in enumerate(zip(data, sectors, tess_mags, ylims)):
            
            #add subplots
            row = len(sectors)
            column = 2
            pos = (idx+1)*2
            ax2 = fig.add_subplot(row, column, pos)
            ax1 = fig.add_subplot(row, column, pos-1)

            #TPFs
            aperture_mask = datum.aperture.copy()
            aperture_mask[(aperture_mask > 0)] = 1
            aperture_plot = np.kron(aperture_mask, np.ones((aperture_detail, aperture_detail)))
            extent = [0-0.5, datum.aperture.shape[1]-0.5, 0-0.5, datum.aperture.shape[0]-0.5]

            ##plot TPF
            ax1.imshow(datum.tpf[0])
            
            ##plot aperture
            ax1.contour(aperture_plot, levels=[1], colors="r", corner_mask=False, origin='lower', aspect='auto', extent=extent, zorder=2)

            #TODO: Maybe check here if extraction worked (only one sector not working?)
            q = eval(quality_expression)
            ax2.plot(datum.time[q], datum.corr_flux[q]/np.nanmedian(datum.corr_flux[q]), marker=".", linestyle="", color="tab:blue", label=f"Corrected Flux")
            try:
                ax2.plot(datum.time[q], datum.pca_flux[q]/np.nanmedian(datum.pca_flux[q]), marker=".", linestyle="", color="tab:orange", label=f"PCA")
            except:
                pass
            try:
                ax2.plot(datum.time[q], datum.psf_flux[q]/np.nanmedian(datum.psf_flux[q]), marker=".", linestyle="", color="tab:green", label=f"PSF")
            except:
                pass
            
            
            #add legends
            leg = ax2.legend(
                fontsize=fontsize-2,
                title=r"$\mathbf{Sector~%i}$, TESS_mag: %.2f"%(sector, tess_mag), title_fontsize=fontsize-2,
                loc="upper right",
            )
            leg._legend_box.align = "left"
            
            aperture_patch = Line2D([0], [0], color="r", label="Chosen Aperture", linewidth=1)
            imlegend = ax1.legend(handles=[aperture_patch])

            #label

            if ylim != False:
                ax2.set_ylim(*ylim)

            if idx == len(data)-1:
                #only label bottom axis
                ax2.set_xlabel("Time [BJD - 2457000]", fontsize=fontsize)
                ax1.set_xlabel("Pixel", fontsize=fontsize)
            if pos == 0:
                ax1.set_title("Target Pixel Files", fontsize=fontsize+4)
            ax1.set_ylabel("Pixel", fontsize=fontsize)

            ax1.tick_params("both", labelsize=fontsize)
            ax2.tick_params("both", labelsize=fontsize)
            ax2.yaxis.set_major_formatter(plt.FormatStrFormatter("%.1f"))    #only use 1 decimal place for tick-labels


        #add common label to second column
        ax01 = fig.add_subplot(122, frameon=False)   #big hidden axis
        ax01.tick_params(labelcolor='none', which='both', labelsize=fontsize, top=False, bottom=False, left=False, right=False) #hide ticks and ticklabels
        ax01.set_ylabel("Normalized Flux", fontsize=fontsize)    #set axis-label
        
        plt.tight_layout()

        #save if specified
        if type(save) == str:
            plt.savefig(save, dpi=180)
        plt.show()
        
        axs = fig.axes

        return fig, axs
   
    def result2pandas(self,
        data:list, sectors:list, tess_mags:list, tic:str,
        quality_expression:str="(datum.quality == 0)",
        sep=";",
        include_aperture=False, include_tpf=False,
        save:str=False,
        ) -> pd.DataFrame:
        """
            - method to convert the result returned by eleanor to a pandas DataFrame
            - also allows saving of the created DataFrame as .csv

            Parameters
            ----------
                - data
                    - list
                    - contains the data for each sector
                        - extracted with eleanor
                - sectors
                    - list
                    - containing the sectors in which the target has been observed
                - tess_mags
                    - list
                    - contains the tess-magnitudes of the target for each sector           
                - tic
                    - str
                    - TIC identifier of the target
                - quality_expression
                    - str, optional
                    - string containing some boolean condition w.r.t. 'datum.quality'
                    - eval() function will be applied to construct a boolean array
                    - the default is "(datum.quality == 0)"
                - sep
                    - str, optional
                    - separator to use when creating the .csv file
                    - the default is ";"
                        - reason for choosing ";" over "," is that aperture and tpf will be stored as nested list, which contain "," as separators
                - include_aperture
                    - bool, optional
                    - whether to include the used aperture in the extracted file
                    - will store the aperture for every single frame
                        - thus, can lead to quite large files
                    - the default is False
                - include_tpf
                    - bool, optional
                    - whether to include the target-pixel-files in the extracted file
                    - will store the tpf for every single frame
                        - thus, can lead to quite large files
                - save
                    - str|bool, optional
                    - path to the directory of where to store the created csv-file
                    - the default is False
                        - will not save results to .csv file
                
            Raises
            ------

            Returns
            -------

            Comments
            --------

        """

        df_lc = pd.DataFrame(
            columns=["time", "raw_flux", "corr_flux", "pca_flux", "psf_flux", "sector", "q_eleanor", "tess_mag"]+[ "aperture"]*include_aperture+["tpf"]*include_tpf
            )
        if include_aperture:
            df_lc["aperture"].astype(object)
        if include_tpf:
            df_lc["tpf"].astype(object)

        for idx, (datum, sector, tess_mag) in enumerate(zip(data, sectors, tess_mags)):

            q = eval(quality_expression)

            if self.do_pca:
                pca_flux = datum.pca_flux[q]
            else:
                pca_flux = [None]*len(datum.time[q])
            if self.do_psf:
                psf_flux = datum.pca_flux[q]
            else:
                psf_flux = [None]*len(datum.time[q])

            df_datum = pd.DataFrame({
                "time":datum.time[q],
                "raw_flux":datum.raw_flux[q],
                "corr_flux":datum.corr_flux[q],
                "pca_flux":pca_flux,
                "psf_flux":psf_flux,
                "sector":[sector]*len(datum.time[q]),
                "q_eleanor":datum.quality[q],
                "tess_mag":[tess_mag]*len(datum.time[q]),
            })

            if include_aperture:
                df_datum["aperture"] = [datum.aperture]*len(datum.time[q])
            if include_tpf:
                df_datum["tpf"] = datum.tpf[q].tolist()


            df_lc = pd.concat((df_lc, df_datum), ignore_index=True)

        if isinstance(save, str):
            df_lc.to_csv(save, index=False, sep=sep)
            

        return df_lc

    def save_npy_eleanor(self,
        data:list, sectors:list, tess_mag:float, save:str,
        target:str, TIC:str=None, GCVS_class:str=None, GCVS_period:float=None, GCVS_RA:str=None, GCVS_DEC:str=None):
        """
            - function to save the extracted data into an 0-dimensional np.array

            Parameters
            ----------
                - data
                    - list
                    - contains the data for each sector
                        - extracted with eleanor
                - sectors
                    - list
                    - containing the sectors in which the target has been observed
                - tess_mags
                    - list
                    - contains the tess-magnitudes for each sector            
                - save
                    - str
                    - location to save the array to
                    - the default is False
                - target
                    - str
                    - you want to give your target
                        - a good idea is to use the common name or some standard identiier
                - TIC
                    - str, optional
                    - TIC identifier of your target
                    - the default is None
                - GCVS_class
                    - str, optional
                    - class assigned to the target in the literature (i.e. GCVS)
                    - the default is None
                - GCVS_period
                    - float, optional
                    - period noted in the GCVS for the target
                    - the default is None
                - GCVS_RA
                    - str, optional
                    - right ascension noted in the GCVS for the target
                    - the default is None
                - GCVS_DEC
                    - str, optional
                    - declination noted in the GCVS for the target
                    - the default is None

            Raises
            ------

            Returns
            -------

            Comments
            --------
                - NOT MAINTAINED ANYMOER
        
        """

        warnings.warn("WARNING: save_npy_eleanor is not being maintained anymore. It might not be compatible with newer versions of the other methods.")

        savedict = {
            "target":target,
            "TIC":TIC,
            "tess_mags":tess_mag,
            "times":np.array([]),
            "raw_flux":np.array([]),
            "corr_flux":np.array([]),
            "pca_flux":np.array([]),
            "psf_flux":np.array([]),
            "sectors":np.array([]),
            "aperture":[],
            "tpf":[],
            "GCVS_class":GCVS_class,
            "GCVS_period":GCVS_period,
            "GCVS_RA":GCVS_RA,
            "GCVS_DEC":GCVS_DEC
        }
        for idx, (datum, sector) in enumerate(zip(data, sectors)):

            q = datum.quality == 0
            savedict["times"] = np.append(savedict["times"], datum.time[q])
            savedict["raw_flux"] = np.append(savedict["raw_flux"], datum.raw_flux[q])
            savedict["corr_flux"] = np.append(savedict["corr_flux"], datum.corr_flux[q])
            try:
                savedict["pca_flux"] = np.append(savedict["pca_flux"], datum.pca_flux[q])
            except:
                pass
            try:
                savedict["psf_flux"] = np.append(savedict["psf_flux"], datum.psf_flux[q])
            except:
                pass
            savedict["sectors"] = np.append(savedict["sectors"], [sector]*(len(datum.raw_flux[q])))
            savedict["aperture"].append(datum.aperture)
            savedict["tpf"].append(datum.tpf)
            
            # print(sector, datum.aperture.shape)

        # print(len(savedict["aperture"]))
        # print(savedict.keys())

        # print(len(savedict["times"]))
        # print(len(savedict["raw_flux"]))
        # print(len(savedict["corr_flux"]))
        # print(len(savedict["pca_flux"]))
        # print(len(savedict["psf_flux"]))
        # print(len(savedict["sectors"]))
        # print(savedict["GCVS_class"])
        # print(savedict["GCVS_period"])
        np.save(save, savedict)

        return

#GAIA
class GaiaDatabaseInterface:
    """
        - class to interact with the Gaia-archive

        Attributes
        ----------
            -
        
        Methods
        -------

        Dependencies
        ------------
            - astropy
            - astroquery
            - numpy
            - os
            - pandas
            - urllib

    """
    def __init__(self):

        self.gaia_crendetials = None
        self.df_LCs = pd.DataFrame()
        pass


    def get_lc_gaia(self,
        gaia_result:str):
        """
            - function to obtain the light-curve data from a gaia query

            Parameters
            ----------
                - gaia_result
                    - str
                    - location of the saved query-result
                    - has to be a .vot file (votable)
            
            Raises
            ------

            Returns
            -------
                - results
                    - list
                    - list of astropy.Table objects
                        - each table contains the result for one of the rows in the input-votable

            Dependencies
            ------------
                - os
                - astropy
                - urllib
                - numpy
            
            Comments
            --------
        """


        res = Table.read(gaia_result, format="votable")

        urls = np.array(res["datalink_url"].data.data)

        results = []
        for idx, url in enumerate(urls):
            tempfilename = f"tempfile_1{idx}"
            lcfilename = f"FT CVn{idx}.xml"

            request.urlretrieve(url, tempfilename)
            a_url = Table.read(tempfilename, format="votable")["access_url"].data.data[0]

            try:
                request.urlretrieve(a_url, lcfilename)
                res2 = Table.read(lcfilename, format="votable")
            except Exception as e:
                print("WARNING: Error in request.urlretrieve")
                print(f"original ERROR: {e}")
                res2 = None

            results.append(res2)

            os.remove(tempfilename)

        return results

    def remove_all_jobs(self,
        pd_filter:str=None, gaia_credentials:str=None,
        login_before:bool=False, logout_after:bool=False,
        verbose:int=0):
        """
            - method to remove all jobs stored in the Gaia-Archive

            Parameters
            ----------
                - pd_filter
                    - str, optional
                    - string that will be evaluated as a boolean filter
                        - serves to filter all jobs for some specific ones
                    - must have the following structure
                        - "(jobs['colname1'] == 'argument1')&(jobs['colname2'] != 'argument2')" ect.
                        - colname1 and colname2 are hereby substituted by one of the following
                            - jobid
                            - failed
                            - creationDate
                            - creationTime
                            - endDate
                            - endTime
                            - startDate
                            - startTime
                            - responseStatus
                            - phase
                        - argument1 and argument 2 are substituted by some value for the boolean expression
                        - some useful examples
                            - removing all jobs created at the present day

                                >>> today = pd.to_datetime('today').date()
                                >>> pd_filter = f"(jobs['creationDate'] == '{today}')"
                                >>> DI.remove_all_jobs(pd_filter=pd_filter, verbose=0)

                    - the default is None
                        - will delete all jobs in the archive
                - gaia_credentials
                    - str, optional
                    - path to your gaia credentials file
                    - required if login_before is True and no credentials file has been passed yet
                    - the default is None
                - login_before
                    - bool, optional
                    - whether to log into the gaia archive before deleting the jobs
                    - not necessary if you login somwhere in your code before calling 'remove_all_jobs()'
                    - the default is False
                - logout_after
                    - bool, optional
                    - whether to log out of the gaia archive after deleting the jobs
                    - the default is False
                - verbose
                    - int, optional
                    - how much additional information to display
                    - the default is 0

            Raises
            ------
                - ValueError
                    - if the user is not logged in correctly

            Returns
            -------

            Dependencies
            ------------
                - astroquery
                - pandas
            
            Comments
            --------
        """

        if login_before:
            if self.gaia_crendetials is None:
                if gaia_credentials is None:
                    raise ValueError(
                        "Not able to log into Gaia."
                        "Provide a path to your credentials file as the argument 'gaia_credentials'")
                else:
                    self.gaia_crendetials = gaia_credentials


            #login
            Gaia.login(credentials_file=self.gaia_crendetials)
        
        #get all jobs
        all_jobs = Gaia.search_async_jobs(verbose=False)
        
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
        Gaia.remove_jobs(to_remove)
        
        #logout
        if logout_after:
            Gaia.logout()

        return

#ALeRCE
class AlerceDatabaseInterface:

    """
        - class to interact with the ZTF database via the Alerce python API

        Attributes
        ----------

        Methods
        -------
            - crossmatch_by_coordinates()
            - download_lightcurves()

        Dependencies
        ------------
            - Alerce
            - joblib
            - matplotlib
            - numpy
            - pandas
        Comments
        --------
    """


    def __init__(self,
        ):
        self.alerce = Alerce()

        return

    def crossmerge_by_coordinates(
        self,
        df_left:pd.DataFrame,
        ra_colname:str, dec_colname:str, radius:float,
        sleep:float=0,
        n_jobs:int=-1, verbose:int=0,
        ) -> pd.DataFrame:
        """
            - method to crossmerge 'df_left' with the ZTF catalog by coordinates (cone search)
                - will take each row in 'df_left' and find the corresponding target in the ZTF catalog via coordinates
                - will then proceed to append to each matched entry in the ZTF catalog the input row from 'df_left'
                - will combine everything into one huge table

            Parameters
            ----------
                - df_left
                    - pd.DataFrame
                    - table to be crossmerged with ZTF
                    - must contain ra and dec as columns
                - ra_colname
                    - str
                    - name of the column to be considered as Right Ascension
                - dec_colname
                    - str
                    - name of the column to be considered as Declination
                - radius
                    - float
                    - radius to use for the cone search
                - sleep
                    - float, optional
                    - number of seconds to sleep after downloading each target
                    - the default is 0
                        - no sleep at all                    
                - n_jobs
                    - int, optional
                    - number of jobs to be used by joblib.Parallel()
                    - the default is -1
                        - will use all available resources
                -  verbose
                    - int, optional
                    - verbosity level
                    - the default is 0

            Raises
            ------

            Returns
            -------

            Dependencies
            ------------
                - Alerce
                - joblib
                - pandas
                - numpy

            Comments
            --------

        """

        def query_one(
            idx:int, inrow:pd.DataFrame,
            ra_colname:str, dec_colname:str, radius:float,
            total_targets:int,
            sleep:float,
            ):

            print(f"\nExtracting #{idx+1}/{total_targets}")

            error_msg = None
            success = True
            
            try:
                df = self.alerce.query_objects(
                    format='pandas',
                    ra=inrow[ra_colname], dec=inrow[dec_colname], radius=radius
                )

                df.rename(columns={'meanra':'raj2000', 'meandec':'dej2000'}, inplace=True)
                df = df.add_suffix('_ztf')

                # print(idx, df.columns)

                #append all entries of inrow to each of the extracted rows in df
                df.loc[:, inrow.index] = inrow.values

            except Exception as e:
                #empty placeholder DataFrame
                df = pd.DataFrame()
                if verbose > 1:
                    print("WARNING: Error in alerce.query_objects()")
                    print(f"ORIGINAL ERROR: {e}")
                    if len(str(e)) > 80: e = str(e)[:79] + '...'
                    error_msg = f"{'alerce.query_objects()':25s}: {e}"
                    success = False

            #sleep after each target
            time.sleep(sleep)

            return df, idx, success, error_msg


        result = Parallel(n_jobs, verbose=verbose)(
            delayed(query_one)(
                idx=idx, inrow=inrow,
                ra_colname=ra_colname, dec_colname=dec_colname, radius=radius,
                total_targets=df_left.shape[0],
                sleep=sleep,
            ) for idx, inrow in df_left.iterrows()
        )

        result = np.array(result)

        self.df_error_msgs_crossmerge = pd.DataFrame(
            data=result[:,1:],
            columns=['idx','success','original error message'],
        )

        df = pd.concat(result[:,0], ignore_index=True)


        return df

    def download_lightcurves(self,
        ztf_ids:list,
        #saving data
        save:str="./",
        #plotting
        plot_result:bool=True, save_plot:str=False, close_plots:bool=True,
        #calculating
        sleep:float=0,
        n_jobs:int=-1, verbose:int=0
        ) -> None:
        """
            - function to download all lightcurves corresponding to the ZTF ids in 'ztf_ids'

            Parameters
            ----------
                - save
                    - str, bool, optional
                    - directory of where to store the downloaded lightcurves to
                    - if set to False, will not save the data
                    - save has to end with a slash ('/')
                    - the default is './'
                - plot_result
                    - bool, optional
                    - whether to plot the lightcurve of the downloaded data
                        - will create one plot for each target
                    - the default is True
                - save_plot
                    - str, optional   
                    - directory of where to store the created plots
                    - save_plot has to end with a slash ('/')
                    - the default is False
                        - will not save the plots
                - close_plots
                    - bool, optional
                    - whether to close the plots immediately after creation
                    - useful if one wants to save the plots, but not view at them right away
                    - the default is True
                - sleep
                    - float, optional
                    - number of seconds to sleep after downloading each target
                    - the default is 0
                        - no sleep at all
                - n_jobs
                    - int, optional
                    - number of jobs to be used by joblib.Parallel()
                    - the default is -1
                        - will use all available resources
                -  verbose
                    - int, optional
                    - verbosity level
                    - the default is 0

            Raises
            ------
                - assertion error
                    - if 'save' or 'save_plots' are not formatted correctly
                
            Returns
            -------

            Dependencies
            ------------
                - Alerce
                - joblib
                - matplotlib
                - pandas
                - time

            Comments
            --------

        """

        def query_one(
            ztf_id:str,
            idx:int,
            total_targets:int,
            sleep:float,
        ):

            print(f"\nExtracting {ztf_id} (#{idx+1}/{total_targets})")

            error_msg = None
            success = True

            try:
                df = self.alerce.query_detections(
                    ztf_id,
                    format='pandas'
                )

            except Exception as e:
                #empty placeholder DataFrame
                df = pd.DataFrame()
                if verbose > 1:
                    print("WARNING: Error in alerce.query_objects()")
                    print(f"ORIGINAL ERROR: {e}")
                    if len(str(e)) > 80: e = str(e)[:79] + '...'
                    error_msg = f"{'alerce.query_objects()':25s}: {e}"
                    success = False
                # print(len(df['detections']))

            # self.df_error_msgs_lcdownload.loc[len(self.df_error_msgs_lcdownload)] = [ztf_id, success, error_msg]


            if plot_result:
                
                cm = plt.cm.get_cmap('viridis')
                newcolors = cm(np.linspace(0, 1, 256))
                newcolors[        :1*256//3, :] = mcolors.to_rgba('tab:green')
                newcolors[1*256//3:2*256//3, :] = mcolors.to_rgba('tab:red')
                newcolors[2*256//3:3*256//3, :] = mcolors.to_rgba('tab:purple')
                newcmap = mcolors.ListedColormap(newcolors)

                fig = plt.figure()

                fig.suptitle(ztf_id)

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

                plt.tight_layout()
                if isinstance(save_plot, str): plt.savefig(f'{save_plot}{ztf_id}.png')

                plt.show()

                if close_plots: plt.close()
            
            if isinstance(save, str): df.to_csv(f'{save}{ztf_id}.csv', index=False)


            # print(df)
            # print(df.columns)
            # print(df.shape)

            #sleep after downloading one target
            time.sleep(sleep)

            return df, ztf_id, success, error_msg

        if isinstance(save, str):
            assert save[-1] == '/' or save[-1] == '\\', \
                '"save" has to end either with a slash ("/") or backslash ("\\")'
        if isinstance(save_plot, str):
            assert save_plot[-1] == '/' or save_plot[-1] == '\\', \
                '"save_plot" has to end either with a slash ("/") or backslash ("\\")'

        result = Parallel(n_jobs, verbose=verbose)(
            delayed(query_one)(
                ztf_id=ztf_id, idx=idx,
                total_targets=len(ztf_ids),
                sleep=sleep,
            ) for idx, ztf_id in enumerate(ztf_ids)
        )

        result = np.array(result)

        self.df_error_msgs_lcdownload = pd.DataFrame(
            data=result[:,1:],
            columns=['ztf','success','original error message'],
        )        

        return 


#lightcurve package
class LightkurveInterface:

    def __init__(self, tics):
        
        if tics.dtype == object:
            raise TypeError(f'"tics" has to have dtype of np.float64 or np.int64 and not {tics.dtype}!')
        self.tics = tics

        return

    def get_target_lc(self,
        tpf,
        threshold:float=15,
        reference_pixel:str='center'
        ):
        """
            - function to extract the target lightcurve from a given tpf-object

            Parameters
            ----------
                - tpf
                    - tpf-object
                    - contains a series of target pixel files
                - threshold
                    - float, optional
                    - threshold for selecting the pixels for the aperture
                    - the default is 15
                - reference_pixel
                    - str, optional
                    - which pixel to use as a reference for the aperture determination
                    - the default is 'center'
            
            Raises
            ------

            Returns
            -------
                - target_lc
                    - lightkurve object
                    - the lightcurve of the target
                - n_target_pixels
                    - int
                    - number of pixels used in the aperture
                - target_mask
                    - np.ndarray
                    - boolean mask defining the used aperture
                
            Comments
            --------
        """

        #create aperture mask
        target_mask = tpf.create_threshold_mask(threshold=threshold, reference_pixel=reference_pixel)
        
        #count to normalize and relate to backgruond
        n_target_pixels = target_mask.sum()
        target_lc = tpf.to_lightcurve(aperture_mask=target_mask)

        return target_lc, n_target_pixels, target_mask
    
    def get_background_estimate(self,
        tpf, n_target_pixels,
        threshold:float=0.001, reference_pixel:str=None,
        ):
        """
            - function to extract the background lightcurve from a given tpf-object

            Parameters
            ----------
                - tpf
                    - tpf-object
                    - contains a series of target pixel files
                - n_target_pixels
                    - int
                    - number of pixels used in the aperture
                - threshold
                    - float, optional
                    - threshold for selecting the pixels for the background-"aperture"
                    - the default is 0.001
                - reference_pixel
                    - str, optional
                    - which pixel to use as a reference for the background-"aperture" determination
                    - the default is None
            
            Raises
            ------

            Returns
            -------
                - background_estimate_lc
                    - lightkurve object
                    - the lightcurve of the estimated background
                - background_mask
                    - np.ndarray
                    - boolean mask defining the used background-"aperture"
                
            Comments
            --------
        """

        #get background pixels
        background_mask = ~tpf.create_threshold_mask(threshold=threshold, reference_pixel=reference_pixel)
        
        #normalize per pixel
        n_background_pixels = background_mask.sum()
        background_lc_per_pixel = tpf.to_lightcurve(aperture_mask=background_mask) / n_background_pixels
        
        #rescale to match target pixels
        background_estimate_lc = background_lc_per_pixel * n_target_pixels
        
        return background_estimate_lc, background_mask
    
    def download_lightcurves_tess(self,
        #saving data
        save:str="./",
        sectors:list='all',
        threshold_target:float=15, threshold_background:float=0.001,
        reference_pixel_target:str='center', reference_pixel_background:str=None,
        height:int=15, width:int=15,
        quality_expression:str="(df_res['q_lightkurve'] == 0)",
        include_aperture:bool=False, include_tpf:bool=False,
        #plotting
        plot_result:bool=True,
        aperture_detail:int=50, ylims:list=None,
        fontsize:int=16, figsize:tuple=(16,9),
        save_plot:str=False,
        sleep:float=0,
        n_jobs:int=-1, n_chunks:int=1,
        verbose:int=2
        ):
        """
            - method to extract data for all ids in 'self.tics'

            Parameters
            ----------
                - save
                    - str|bool, optional
                    - path to the directory of where to store the extracted files
                    - the default is "./"
                        - will save to the current working directory
                - sectors
                    - list
                    - list of sectors to search in
                    - the default is 'all'
                        - will consider all sectors                
                - quality_expression
                    - str, optional
                    - string containing some boolean condition w.r.t. "df_res['q_lightkurve']"
                    - eval() function will be applied to construct a boolean array
                    - the default is "(df_res['q_lightkurve'] == 0)"
                - include_aperture
                    - bool, optional
                    - whether to include the used aperture in the extracted file
                    - will store the aperture for every single frame
                        - thus, can lead to quite large files
                    - the default is False
                - include_tpf
                    - bool, optional
                    - whether to include the target-pixel-files in the extracted file
                    - will store the tpf for every single frame
                        - thus, can lead to quite large files
                    - the default is False
                - plot_result
                    - bool, optional
                    - whether to create a plot of the extracted file
                    - the default is True
                - aperture_detail
                    - NOT USED  
                    - int, optional
                    - how detailed the aperture shall be depicted
                    - higher values require more computation time
                    - the default is 50
                - ylims
                    - list, tuple, optional
                    - the ylims of the created plot
                    - the default is None
                        - will automatically adapt the ylims
                - fontsize
                    - int, optional
                    - fontsize to use on the created plot
                    - the default is 16
                - figsize
                    - tuple, optional
                    - size of the created figure/plot
                    - the default is (16,9)
                - save_plot
                    - str, optional
                    - location of where to save the created plot
                    - the default is False
                        - will not save the plot
                - sleep
                    - float, optional
                    - number of seconds to sleep after downloading each target
                    - the default is 0
                        - no sleep at all                
                - n_jobs
                    - int, optional
                    - number of workers to use for parallel execution of tasks
                        - '1' is essentially sequential computaion (useful for debugging)
                        - '2' uses 2 CPUs
                        - '-1' will use all available CPUs
                        - '-2' will use all available CPUs-1
                    - for more info see https://joblib.readthedocs.io/en/latest/generated/joblib.Parallel.html
                    - the default is '-1'
                - n_chunks
                    - NOT USED
                    - int, optional
                    - number of chuncks to split the input data into
                        - will take 'self.tics' and split it into n_chunks more or less equal subarrays
                    - after processing of each chunk, the metadata will get deleted, if 'self.clear_metadata_after_extract' is set to true
                    - it is advisable, though not necessary, to choose n_chunks such that each chunck contains an amount of elements that can be evenly distributed across n_jobs
                    - the default is 1
                        - i.e. process all the data in one go
                - verbose
                    - int, optional
                    - verbosity level
                        - also for joblib.Parallel()
                    - higher numbers will show more information
                    - the default is 2
            
            Raises
            ------

            Returns
            -------

            Comments
            --------


        """

        if n_jobs > 1:
            raise NotImplementedError('Parallelization not implemented yet! Use n_jobs = 1!')

        def download_one(cidx, idx, chunk_len, n_chunks, tic, save_plot, sleep):
            
            tic = int(tic)

            print(f"\nExtracting tic{tic} (chunk {cidx+1}/{n_chunks}, {idx+1}/{chunk_len} = {chunk_len*(cidx)+idx+1}/{len(self.tics)} Total)")
            
            tic, df_res, tpfs, apertures, success, error_msg = \
                self.download_one_lightcurve_tess(
                    tic=int(tic), sectors=sectors, 
                    threshold_target=threshold_target, threshold_background=threshold_background,
                    reference_pixel_target=reference_pixel_target, reference_pixel_background=reference_pixel_background,
                    height=height, width=width,
                    include_aperture=include_aperture, include_tpf=include_tpf,
                    save=save,
                    plot_result=plot_result,
                    ylims=ylims,
                    fontsize=fontsize,
                    figsize=figsize,
                    save_plot=save_plot,
                    sleep=sleep,
                    verbose=verbose,                
                )

            plt.close()

            return tic, df_res, tpfs, apertures, success, error_msg

        #split array into n_chuncks chuncks to divide the load
        chunks = np.array_split(self.tics, n_chunks)
        if verbose > 1:
            print(f"INFO: Extracting {len(chunks)} chuncks with shapes {[c.shape for c in chunks]}")

        extraction_summaries = []
        for cidx, chunk in enumerate(chunks):
            res = Parallel(
                n_jobs=n_jobs, verbose=verbose
            )(
                delayed(download_one)(
                    cidx, idx, len(chunk), n_chunks, tic, save_plot, sleep=sleep,
                    ) for idx, tic in enumerate(chunk)
            )
        
            res = np.array(res)

            df_extraction_summary_chunck = pd.DataFrame(
                data=np.array([
                    res[:,0],
                    res[:,4],
                    res[:,5],
                ]).T,
                columns=['tic', 'success', 'original error message']
            )

            extraction_summaries.append(df_extraction_summary_chunck)
        
        self.df_extraction_summary = pd.concat(extraction_summaries)

        return
    
    def download_one_lightcurve_tess(self,
        tic:int, sectors:list="all", 
        threshold_target:float=15, threshold_background:float=0.001,
        reference_pixel_target:str='center', reference_pixel_background:str=None,
        height:int=15, width:int=15,
        include_aperture:bool=False, include_tpf:bool=False,
        save:str=False,
        #plotting
        plot_result:bool=True,
        ylims:list=None,
        fontsize:int=16, figsize:tuple=(16,9),
        save_plot:str=False,
        #misc        
        sleep:float=0,
        verbose:int=2
        ):
        #TODO: Implement include_tpf
        """
            - method to extract data for all ids in 'self.tics'

            Parameters
            ----------
                - tic
                    - int
                    - TESS Input Catalog identifier to download
                - sectors
                    - list
                    - list of sectors to search in
                    - the default is 'all'
                        - will consider all sectors                
                - threshold_target
                    - float, optional
                    - threshold for selecting the pixels for the target-aperture
                    - the default is 15
                - threshold_background
                    - float, optional
                    - threshold for selecting the pixels for the background-'aperture'
                    - the default is 15
                - reference_pixel_target
                    - str, optional
                    - which pixel to use as a reference for the target-aperture determination
                    - the default is None                    
                - reference_pixel_background
                    - str, optional
                    - which pixel to use as a reference for the background-"aperture" determination
                    - the default is None                    
                - height
                    - int, optional
                    - pixel height of the cutout
                    - the default is 15
                - width
                    - int, optional
                    - pixel width of the cutout
                    - the default is 15                
                - include_aperture
                    - bool, optional
                    - whether to include the used aperture in the extracted file
                    - will store the aperture for every single frame
                        - thus, can lead to quite large files
                    - the default is False
                - include_tpf
                    - NOT IMPLEMENTED YET
                    - bool, optional
                    - whether to include the target-pixel-files in the extracted file
                    - will store the tpf for every single frame
                        - thus, can lead to quite large files
                    - the default is False
                - save
                    - str|bool, optional
                    - path to the directory of where to store the extracted files
                    - the default is False
                        - will not save the files
                - plot_result
                    - bool, optional
                    - whether to create a plot of the extracted file
                    - the default is True
                - ylims
                    - list, tuple, optional
                    - the ylims of the created plot
                    - the default is None
                        - will automatically adapt the ylims
                - fontsize
                    - int, optional
                    - fontsize to use on the created plot
                    - the default is 16
                - figsize
                    - tuple, optional
                    - size of the created figure/plot
                    - the default is (16,9)
                - save_plot
                    - str, optional
                    - location of where to save the created plot
                    - the default is False
                        - will not save the plot
                - sleep
                    - float, optional
                    - number of seconds to sleep after downloading each target
                    - the default is 0
                        - no sleep at all                
                - verbose
                    - int, optional
                    - verbosity level
                        - also for joblib.Parallel()
                    - higher numbers will show more information
                    - the default is 2
            
            Raises
            ------

            Returns
            -------

            Comments
            --------


        """        

        assert isinstance(tic, int), f'"tic" has to be of type int and not {type(tic)}'

        cutout_size = (height, width)
        res = lk.search_tesscut(f'TIC {tic}')
        try:
            tpfs = res.download_all(cutout_size=cutout_size)


            print(f'INFO: Target ({tic}) found in sectors: {tpfs.sector} (downloading for sectors {sectors})')

            if sectors == 'all': sectors = tpfs.sector
            
            data = []
            apertures = {}


            if len(tpfs[np.isin(tpfs.sector, sectors)]) > 0:

                tpfs2download = tpfs[np.isin(tpfs.sector, sectors)]
                texps = res.exptime.value[np.isin(tpfs.sector, sectors)]
                try:
                    for idx, (texp, tpf) in enumerate(zip(texps, tpfs2download)):

                        error_msg = None
                        success = True
                        sector = tpf.sector
                        
                        #target
                        target_lc, n_target_pixels, target_mask = self.get_target_lc(
                            tpf,
                            threshold=threshold_target,
                            reference_pixel=reference_pixel_target,
                        )
                        #background
                        background_estimate_lc, background_mask = self.get_background_estimate(
                            tpf, n_target_pixels,
                            threshold=threshold_background, reference_pixel=reference_pixel_background,                
                        )

                        #correct lc for background
                        corrected_lc = target_lc - background_estimate_lc.flux
                        
                        df = corrected_lc.to_pandas()
                        df.rename(columns={'flux':'corr_flux', 'quality':'q_lightkurve'}, inplace=True)
                        df['time'] = target_lc.time.value
                        df['raw_flux'] = target_lc.flux.value
                        df['sector'] = sector
                        df['t_exp'] = texp

                        if include_aperture:
                            df['aperture'] = [target_mask]*len(df)
                        # if include_tpf:
                        #     df['tpf'] = tpf.flux.value.to_numpy()

                        data.append(df)

                        apertures[sector] = target_mask
                    
                    df_res = pd.concat(data, ignore_index=True)
                    if isinstance(save, str):
                        df_res.to_csv(save+f'tic{tic}.csv', sep=';', index=False)

                    if plot_result:
                        fig, axs = self.plot_lightcurve_extraction(
                            df_res, tic, tpfs, apertures,
                            ylims=ylims,
                            fontsize=fontsize, figsize=figsize,
                            save=save_plot
                            )
                        
                except Exception as e:
                    #empty placeholder DataFrame
                    df_res = pd.DataFrame()
                    if verbose > 1:
                        print(f"WARNING: Error while downloading TIC{tic}")
                        print(f"ORIGINAL ERROR: {e}")
                        if len(str(e)) > 80: e = str(e)[:79] + '...'
                        error_msg = f"{'alerce.query_objects()':25s}: {e}"
                        success = False

            else:
                print(f'INFO: Target ({tic}) not found in requested sectors ({sectors})')
                error_msg = f'INFO: Target ({tic}) not found in requested sectors ({sectors})'
                success = False
                df_res = None

        except Exception as e:
            #empty placeholder DataFrame
            df_res = pd.DataFrame()
            if verbose > 1:
                print(f"WARNING: Error in lightkurves download_all()")
                print(f"ORIGINAL ERROR: {e}")
                if len(str(e)) > 80: e = str(e)[:79] + '...'
                error_msg = f"{'alerce.query_objects()':25s}: {e}"
                success = False
                tpfs = None
                apertures = None

        time.sleep(sleep)

        return tic, df_res, tpfs, apertures, success, error_msg
        
    def plot_lightcurve_extraction(self,
        df_res:pd.DataFrame, tic:int, tpfs, apertures:dict,
        quality_expression:str="(df_res['q_lightkurve'] == 0)",
        ylims:list=None,
        fontsize:int=16, figsize:tuple=(16,9), save:str=False
        ):

        sectors = pd.unique(df_res['sector'])

        if ylims is None:
            ylims = [False]*len(sectors)
        assert len(ylims) == len(sectors), f"'ylims' has to be 'None' or of the same shape as 'sectors'!"

        fig = plt.figure(figsize=figsize)
        fig.suptitle(f"TIC {tic}", fontsize=fontsize+6)

        
        for idx, (sector, ylim) in enumerate(zip(sectors, ylims)):

            cur_sect_bool = (df_res['sector'] == sector)

            #add subplots
            row = len(sectors)
            column = 2
            pos = (idx+1)*2
            ax2 = fig.add_subplot(row, column, pos)
            ax1 = fig.add_subplot(row, column, pos-1)

            
            #TPFs
            aperture_mask = apertures[sector]

            ##plot TPF
            tpfs[idx].plot(ax=ax1, aperture_mask=aperture_mask)
            
            #TODO: Maybe check here if extraction worked (only one sector not working?)
            q = eval(quality_expression)

            # ax2.plot(df_res['time-245700'], df_res['raw_flux']/np.nanmedian(df_res['raw_flux']), marker=".", linestyle="", color="tab:blue", label=f"Raw Flux")
            ax2.plot(df_res[cur_sect_bool]['time'], df_res[cur_sect_bool]['corr_flux']/np.nanmedian(df_res[cur_sect_bool]['corr_flux']), marker=".", linestyle="", color="tab:blue", label=f"Corrected Flux")
            
            
            #add legends
            leg = ax2.legend(
                fontsize=fontsize-2,
                title=r"$\mathbf{Sector~%i}$"%(sector), title_fontsize=fontsize-2,
                loc="upper right",
            )
            leg._legend_box.align = "left"
            
            aperture_patch = Line2D([0], [0], color="r", label="Chosen Aperture", linewidth=1)
            imlegend = ax1.legend(handles=[aperture_patch])

            #label

            if ylim != False:
                ax2.set_ylim(*ylim)

            if idx == len(sectors)-1:
                #only label bottom axis
                ax2.set_xlabel("Time [BJD - 2457000]", fontsize=fontsize)
                ax1.set_xlabel("Pixel", fontsize=fontsize)
            if pos == 0:
                ax1.set_title("Target Pixel Files", fontsize=fontsize+4)
            ax1.set_ylabel("Pixel", fontsize=fontsize)

            ax1.tick_params("both", labelsize=fontsize)
            ax2.tick_params("both", labelsize=fontsize)
            ax2.yaxis.set_major_formatter(plt.FormatStrFormatter("%.1f"))    #only use 1 decimal place for tick-labels

            ax1.set_title('')

        #add common label to second column
        ax01 = fig.add_subplot(122, frameon=False)   #big hidden axis
        ax01.tick_params(labelcolor='none', which='both', labelsize=fontsize, top=False, bottom=False, left=False, right=False) #hide ticks and ticklabels
        ax01.set_ylabel("Normalized Flux", fontsize=fontsize)    #set axis-label
        

        plt.tight_layout()

        #save if specified
        if type(save) == str:
            plt.savefig(save+f'tic{tic}.png', dpi=180)
        plt.show()
        
        axs = fig.axes

        return fig, axs

#%%Testing

# targets = [
#     "KIC 5006817", "RR Lyr", "TV Boo"
# ]

# SDI = SimbadDatabaseInterface()
# ids = SDI.get_ids(
#     targets
# )
# tics = SDI.df_ids["TIC"]

# EDI = EleanorDatabaseInterface(
#     tics=tics[1:2], sectors="all"
#     )

# EDI.data_from_eleanor_alltics(
#     # save="./",
#     save=False,
#     plot_result=False, save_plot=False,
#     sleep=2
# )





# GID = GaiaDatabaseInterface()
# GID.gaia_crendetials = "../credentials_gaia.txt"

# filter = "(jobs['phase'] == 'ERROR')"
# GID.remove_all_jobs(pd_filter=filter, login_before=True, logout_after=True)





# ADI = AlerceDatabaseInterface()

# df = pd.DataFrame(
#     data=np.array([
#         [10054,	12.39495833,	27.02213889,],
#         [10088,	353.7751667,	np.inf,],#41.10291667,],
#         [10140,	16.294625,	34.21841667,],
#         [10147,	359.6756667,	41.48880556,],
#     ]),
#     columns=['id', 'ra', 'dec']
# )

# df_ztf = ADI.crossmerge_by_coordinates(
#     df_left=df,
#     ra_colname='ra', dec_colname='dec', radius=10,
#     sleep=2E-3,
#     n_jobs=-1, verbose=2
# )

# print(ADI.df_error_msgs_crossmerge)

# ADI.download_lightcurves(
#     df_ztf['oid_ztf'],
#     save=False,
#     # save=alerce_extraction_dir,
#     plot_result=False, save_plot=False, close_plots=True,
#     sleep=2E-3,
#     n_jobs=1, verbose=2
# )
# print(ADI.df_error_msgs_lcdownload)


# LKI = LightkurveInterface(
#     tics=tics.astype(np.float64)
# )

# LKI.download_lightcurves_tess(
#     #saving data
#     save="./",
#     sectors='all',
#     quality_expression="(datum.quality == 0)",
#     include_aperture=True, include_tpf=False,
#     #plotting
#     plot_result=True,
#     aperture_detail=50, ylims=None,
#     fontsize=16, figsize=(16,9),
#     save_plot="./",
#     sleep=0,
#     n_jobs=1, n_chunks=1,
#     verbose=2
# )

# LKI.df_extraction_summary