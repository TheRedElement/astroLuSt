
#TODO: eleanor parts not working with pandas yet -> use csv

#%%imports


#%%


class DatabaseInterface:

    def __init__(self):
        import pandas as pd
        
        self.gaia_crendetials = None

        self.df_ids = pd.DataFrame()
        self.df_LCs = pd.DataFrame()
        pass

    #SIMBAD
    def get_ids(self,
        input_ids:list[str],
        nparallelrequests:int=1000, simbad_timeout:int=120,
        show_scanned_strings_at:list[int]=[],
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
        import numpy as np
        import pandas as pd
        import re

        from astroquery.simbad import Simbad
        from joblib import Parallel, delayed

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
    def data_from_eleanor_alltics(self):

        return

    def data_from_eleanor(self,
        tic:str, sectors:list|str="all", 
        do_psf:bool=False, do_pca:bool=False, 
        aperture_mode:str="normal", regressors:str="corner", try_load:bool=True,
        height:int=15, width:int=15, bkg_size:int=31) -> tuple[list, list, float, str]:
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
        import eleanor

        try:
            error_msg = None
            star = eleanor.multi_sectors(tic=tic, sectors=sectors)
            
            data = []
            sectors = []
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
                    tess_mag = s.tess_mag
                    
            except Exception as e:
                print("WARNING: Error in eleanor.TargetData()")
                print(f"ORIGINAL ERROR: {e}")
                data = None
                sectors = None
                tess_mag = None
                error_msg = f"{'eleanor.TargetData()':25s}: {e}"


        except Exception as e:
            print("WARNING: Error in eleanor.multi_sectors()")
            print(f"\t Maybe TIC {tic} was not observed with TESS?")
            print(f"\t If you run extract_data() in a loop, try to run it separately on TIC {tic}.")
            print(f"ORIGINAL ERROR: {e}")
            
            data = None
            sectors = None
            tess_mag = None
            error_msg = f"{'eleanor.multi_sectors()':25s}: {e}"
            
        
        return data, sectors, tess_mag, error_msg

    def plot_result_eleanor(self,
        data:list, target:str, TIC:str, sectors:list, tess_mag:float,
        aperture_detail:int=50, ylims:list|None=None,
        fontsize:int=16, figsize:tuple=(16,9), save:str|bool=False):
        """
            - function to autogenerate a summary-plot of the data downloaded using eleonor
        
            Parameters
            ----------
                - data
                    - list
                    - contains the data for each sector
                        - extracted with eleanor
                - target
                    - str
                    - name under which the target shall be depiced in the plot
                        - shown in the title of the figure
                - TIC
                    - str
                    - TIC identifier of the target
                - sectors
                    - list
                    - containing the sectors in which the target has been observed
                - tess_mag
                    - float
                    - contains the tess-magnitude of the target           
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

        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
        import numpy as np

        if ylims is None:
            ylims = [False]*len(sectors)
        assert len(ylims) == len(sectors), f"'ylims' has to be 'None' or of the same shape as 'sectors'!"

        fig = plt.figure(figsize=figsize)
        fig.suptitle(f"{target} ({TIC}) \n[TESS-mag = {tess_mag}]", fontsize=fontsize+6)

        
        for idx, (datum, sector, ylim) in enumerate(zip(data, sectors, ylims)):
            
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
            q = datum.quality == 0
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
                title=r"$\mathbf{Sector~%i}$"%(sector), title_fontsize=fontsize-2,
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
        
        return

    def save_npy_eleanor(self,
        data:list, sectors:list, tess_mag:float, save:str|bool,
        target:str, TIC:str|None=None, GCVS_class:str|None=None, GCVS_period:float|None=None, GCVS_RA:str|None=None, GCVS_DEC:str|None=None):
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

            Dependencies
            ------------
                - numpy
            
            Comments
            --------
        
        """
        import numpy as np

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
        import os
        from astropy.table import Table
        from urllib import request
        import numpy as np


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
    pd_filter:str|None=None, gaia_credentials:str|None=None,
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
        from astroquery.gaia import Gaia
        import pandas as pd

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
# ID = DatabaseInterface()
# ID.gaia_crendetials = "../credentials_gaia.txt"

# filter = "(jobs['phase'] == 'ERROR')"
# ID.remove_all_jobs(pd_filter=filter, login_before=True, logout_after=True)


# import joblib


# projection = "TSNE"
# clf_name = "DBSCAN"
# dataset = "DataSet4"

# X_train_best_model = joblib.load(f"../saved_data_structures/ML_Model/sector_wise/model_UMAP_HDBSCAN_X.pkl")
# best_model_y_filter = joblib.load(f"../saved_data_structures/ML_Model/sector_wise/model_{projection}_{clf_name}_TIC_ary.pkl")
# input_ids = [f"tic {id}" for id in best_model_y_filter]

# # display(df_ids)

# queries = DatabaseInterface()
# queries.get_ids(
#     input_ids[:200],
#     nparallelrequests=500, simbad_timeout=60,
#     show_scanned_strings_at=[], verbose=1)

# df = queries.df_ids

#%%
