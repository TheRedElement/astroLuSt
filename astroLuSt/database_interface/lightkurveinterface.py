
#%%imports
from joblib import Parallel, delayed
import lightkurve as lk
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import time



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

