
#%%imports
from alerce.core import Alerce
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import time

from astroLuSt.monitoring.timers import ExecTimer


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
        
        self.ET = ExecTimer()

        return

    def crossmerge_by_coordinates(
        self,
        df_left:pd.DataFrame,
        ra_colname:str, dec_colname:str, radius:float,
        sleep:float=0,
        n_jobs:int=-1, verbose:int=0,
        timeit:bool=False,
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
                - timeit
                    - bool, optional
                    - whether to time the execution
                    - the default is False

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

        if timeit:
            self.ET.checkpoint_start('AlerceDatabaseInterface().crossmerge_by_coordinates()')

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

        result = np.array(result, dtype=object)

        self.df_error_msgs_crossmerge = pd.DataFrame(
            data=result[:,1:],
            columns=['idx','success','original error message'],
        )

        df = pd.concat(result[:,0], ignore_index=True)

        if timeit:
            self.ET.checkpoint_end('AlerceDatabaseInterface().crossmerge_by_coordinates()')


        return df

    def download_lightcurves(self,
        ztf_ids:list,
        #saving data
        save:str="./",
        #plotting
        plot_result:bool=True, save_plot:str=False, close_plots:bool=True,
        #calculating
        sleep:float=0,
        n_jobs:int=-1, verbose:int=0,
        timeit:bool=False,
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
                - timeit
                    - bool, optional
                    - whether to time the execution
                    - the default is False                    

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

        if timeit:
            self.ET.checkpoint_start('AlerceDatabaseInterface().download_lightcurves()')


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

        result = np.array(result, dtype=object)

        self.df_error_msgs_lcdownload = pd.DataFrame(
            data=result[:,1:],
            columns=['ztf','success','original error message'],
        )        

        if timeit:
            self.ET.checkpoint_end('AlerceDatabaseInterface().download_lightcurves()')


        return 

