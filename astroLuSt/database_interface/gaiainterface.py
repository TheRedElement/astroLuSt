
#%%imports
from astropy.table import Table
from astroquery.gaia import Gaia
import numpy as np
import os
import pandas as pd
from urllib import request


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
