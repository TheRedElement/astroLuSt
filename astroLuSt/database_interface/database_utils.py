

#%%imports

#data manipulation
import numpy as np
import pandas as pd
import re

#plotting
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

#os interaction
import os

#web interaction
from urllib import request

#astronomy
from astropy.table import Table
from astroquery.simbad import Simbad
from astroquery.gaia import Gaia
from joblib import Parallel, delayed
import eleanor


#warnings
import warnings

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
                            print(id)
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
        height:int=15, width:int=15, bkg_size:int=31        
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
        verbose:int=0
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
                - verbose
                    - int, optional
                    - verbosity level
                    - higher numbers will show more information
            
            Raises
            ------

            Returns
            -------

            Comments
            --------


        """

        extraction_fail = []

        for idx, tic in enumerate(self.tics):

            print(f"\nExtracting tic{tic} ({idx+1}/{len(self.tics)})")

            data, sectors, tess_mags, error_msg = self.data_from_eleanor(
                tic, sectors=self.sectors,
                do_psf=self.do_psf, do_pca=self.do_pca,
                aperture_mode=self.aperture_mode, regressors=self.regressors, try_load=self.try_load,
                height=self.height, width=self.width, bkg_size=self.bkg_size, 
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
                    fig, axs = self.plot_result_eleanor(
                        data=data, tic=tic, sectors=sectors, tess_mags=tess_mags,
                        quality_expression=quality_expression,
                        aperture_detail=aperture_detail, ylims=ylims,
                        fontsize=fontsize, figsize=figsize, save=f"{save_plot}tic{tic}.png",                  
                    )
                
            else:
                #append failed extractions
                extraction_fail.append(tic)

        #short protocoll about extraction
        if verbose > 1:
            print(f"\nExtraction failed for:")
            for ef in extraction_fail:
                print(f"\t{ef[1]} ({ef[0]})")
            print()        

        return

    def data_from_eleanor(self,
        tic:int, sectors:list="all", 
        do_psf:bool=False, do_pca:bool=False, 
        aperture_mode:str="normal", regressors:str="corner", try_load:bool=True,
        height:int=15, width:int=15, bkg_size:int=31) -> tuple:
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
                print("WARNING: Error in eleanor.TargetData()")
                print(f"ORIGINAL ERROR: {e}")
                data = None
                sectors = None
                tess_mags = None
                error_msg = f"{'eleanor.TargetData()':25s}: {e}"


        except Exception as e:
            print("WARNING: Error in eleanor.multi_sectors()")
            print(f"\t Maybe TIC {tic} was not observed with TESS?")
            print(f"\t If you run extract_data() in a loop, try to run it separately on TIC {tic}.")
            print(f"ORIGINAL ERROR: {e}")
            
            data = None
            sectors = None
            tess_mags = None
            error_msg = f"{'eleanor.multi_sectors()':25s}: {e}"
            
        
        return data, sectors, tess_mags, error_msg

    def plot_result_eleanor(self,
        data:list, tic:int, sectors:list, tess_mags:list,
        quality_expression:str="(datum.quality == 0)",
        aperture_detail:int=50, ylims:list=None,
        fontsize:int=16, figsize:tuple=(16,9), save:str=False):
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
        fig.suptitle(f"tic{tic}", fontsize=fontsize+6)

        
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
            columns=["times", "raw_flux", "corr_flux", "pca_flux", "psf_flux", "sector", "q_eleanor", "tess_mag"]+[ "aperture"]*include_aperture+["tpf"]*include_tpf
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
                "times":datum.time[q],
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
#     save="./",
#     plot_result=True, save_plot="./",
# )





# GID = GaiaDatabaseInterface()
# GID.gaia_crendetials = "../credentials_gaia.txt"

# filter = "(jobs['phase'] == 'ERROR')"
# GID.remove_all_jobs(pd_filter=filter, login_before=True, logout_after=True)


