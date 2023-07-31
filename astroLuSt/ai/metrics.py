
#%%imports
import matplotlib.colors as mcolors
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from typing import Union, Tuple, Literal
import warnings

from astroLuSt.visualization.plotting import generate_colors


#%%classes
class MultiConfusionMatrix:
    """
        - class to create a multi-model confusion matrix
        - inspired by the Weights&Biases Parallel-Coordinates plot 
            - https://wandb.ai/wandb/plots/reports/Confusion-Matrix-Usage-and-Examples--VmlldzozMDg1NTM (last access: 2023/07/20)

        Attributes
        ----------
            - `score_decimals`
                - int, optional
                - number of decimals to round `score` to when displaying
                - only relevant if `m_labels == 'score'`
                - the default is 2
            - `text_colors` 
                - str, tuple, list, optional
                - colors to use for displaying
                    - model/bar labels in `plot_func='multi'`
                    - cell-values in `plot_func='single'`
                - if str
                    - will use that color for all bars/cells
                - if tuple
                    - has to be RGBA tuple
                    - will use that color for all bars/cells
                - if list
                    - for `plot_func='multi'`
                        - will use entry 0 for bar 0, entry 1 for bar 1 ect.
                    - for `plot_func='single'`
                        - length has to be equal to `confmat.size`
                        - will display colors from top-left to bottom right (in reading direction)                    
                - the default is `None`
                    - will autogenerate colors
            - `cmap`
                - str, mcolors.Colormap, optional
                - colormap to use for coloring the different models
                - the default is `None`
                    - will be set to `'nipy_spectral'`
            - `vmin`
                - float, optional
                - minimum value of the colormapping
                - used in scaling the colormap
                - argument of `astroLuSt.visualization.plotting.generate_colors()`
                - the default is `None`
            - `vmax`
                - float, optional
                - maximum value of the colormapping
                - used in scaling the colormap
                - argument of `astroLuSt.visualization.plotting.generate_colors()`
                - the default is `None`
            - `vcenter`
                - float, optional
                - center value of the colormapping
                - used in scaling the colormap
                - argument of `astroLuSt.visualization.plotting.generate_colors()`
                - the default is `None`
            - `verbose`
                - int, optional
                - verbosity level
                - the deafault is 0
            - `fig_kwargs`
                - dict, optional
                - kwargs to pass to `plt.figure()`
                - the default is `None`
                    - will be set to `dict(figsize=(9,9))`

        Methods
        -------
            - `__pad()`
            - `get_multi_confmat()`
            - `plot_bar()`
            - `plot_singlemodel()`
            - `plot_multimodel()`
            - `plot_result()`

        Dependencies
        ------------
            - matplotlib
            - numpy
            - sklearn
            - typing
            - warnings
        
        Comments
        --------

    """

    def __init__(self,
        score_decimals:int=2,
        text_colors:Union[str,tuple,list]=None,
        cmap:Union[str,mcolors.Colormap]=None, vmin:float=None, vmax:float=None, vcenter:float=None,
        verbose:int=0,
        fig_kwargs:dict=None,
        ) -> None:

        self.score_decimals = score_decimals
        self.text_colors    = text_colors
        if cmap is None:        self.cmap       = 'nipy_spectral'
        else:                   self.cmap       = cmap
        self.vmin       = vmin
        self.vmax       = vmax
        self.vcenter    = vcenter
        self.verbose    = verbose
        if fig_kwargs is None:  self.fig_kwargs = dict(figsize=(9,9))
        else:                   self.fig_kwargs = fig_kwargs
        
        
        return

    def __repr__(self):

        return (
            f'MultiConfusionMatrix(\n'
            f'    score_decimals={repr(self.score_decimals)},\n'
            f'    cmap={repr(self.cmap)}, vmin={self.vmin}, vmax={self.vmax}, vcenter={self.vcenter},\n'
            f'    verbose={repr(self.verbose)},\n'
            f'    fig_kwargs={repr(self.fig_kwargs)},\n'
            f')'
        )

    def __pad(self,
        y_true:Union[np.ndarray,list], y_pred:Union[np.ndarray,list],
        ) -> Tuple[np.ndarray,np.ndarray]:
        """
            - private method to pad all arrays in `y_true` and `y_pred` to have the same length

            Parameters
            ----------
                - `y_true`
                    - np.ndarray, list, optional
                    - ground truth labels
                    - has to be 2d
                        - `shape = (nmodels,nsampels)`
                        - `nsamples` can vary by model
                    - the default is `None`
                        - will use `confmats` instead of `y_true` and `y_pred`
                - `y_pred`
                    - np.ndarray, optional
                    - model predictions
                    - has to be 2d
                        - `shape = (nmodels,nsamples)`
                        - `nsamples` can vary by model
                    - the default is `None`
                        - will use `confmats` instead of `y_true` and `y_pred`

            Raises
            ------

            Returns
            -------
                - `y_true_pad`
                    - np.ndarray, optional
                    - padded version of `y_true`
                - `y_pred_pad`
                    - np.ndarray, optional
                    - padded version of `y_pred`            

            Comments
            --------
                - 
        """
        
        maxlen = np.max(np.array([[len(yt),len(yp)] for yt, yp in zip(y_true, y_pred)]))

        y_true_pad = np.full((len(y_true), maxlen), np.nan, dtype=np.float64)
        y_pred_pad = np.full((len(y_pred), maxlen), np.nan, dtype=np.float64)
        for idx, (yt, yp) in enumerate(zip(y_true, y_pred)):
            y_true_pad[idx, :yt.shape[0]] = yt
            y_pred_pad[idx, :yp.shape[0]] = yp
        
        return y_true_pad, y_pred_pad
    

    def get_multi_confmat(self,
        y_true:Union[np.ndarray,list], y_pred:Union[np.ndarray,list],
        sample_weight:np.ndarray=None,
        normalize:Literal['true','pred','all']=None,
        verbose:int=None,
        ) -> np.ndarray:
        """
            - method to generate a multi-confusion matrix

            Parameters
            ----------
                - `y_true`
                    - np.ndarray, list, optional
                    - ground truth labels
                    - has to be 2d
                        - `shape = (nmodels,nsampels)`
                        - `nsamples` can vary by model
                    - the default is `None`
                        - will use `confmats` instead of `y_true` and `y_pred`
                - `y_pred`
                    - np.ndarray, list, optional
                    - model predictions
                    - has to be 2d
                        - `shape = (nmodels,nsamples)`
                        - `nsamples` can vary by model
                    - the default is `None`
                        - will use `confmats` instead of `y_true` and `y_pred`
                - `sample_weight`
                    - np.ndarray, optional
                    - sample weights to use in `sklearn.metrics.confusion_matrix()`
                    - the default is `None`
                - `normalize`
                    -  Literal['true','pred','all'], optinonal
                    - how to normalize the confusion matrix
                    - if `'true'` is passed
                        - normalize w.r.t. `y_true`
                    - if `'pred'` is passed
                        - normalize w.r.t. `y_pred`
                    - if `'all'` is passed
                        - normalize w.r.t. all confusion matrix cells
                    - will be passed to `sklearn.metrics.confusion_matrix()`
                    - the default is `None`
                        - no normalization
                - `verbose`
                    - int, optional
                    - verbosity level
                    - overrides `self.verbose`
                    - the deafault is `None`
                        - will fall back to `self.verbose`

            Raises
            ------
                - `ValueError`
                    - if `y_true` and `y_pred` have a different lengths

            Returns
            -------
                - `multi_confmat`
                    - np.ndarray
                    - 3d of shape `(nmodels,nclasses,nclasses)`
                    - multi-confusion-matrix for the ensemble of models

            Comments
            --------
        """

        #default parameters
        if verbose is None: verbose = self.verbose

        #pad if 2d and necessary
        try:
            _ = y_true[0][0], y_pred[0][0]      #check if 2d
            y_true, y_pred = self.__pad(y_true=y_true, y_pred=y_pred)
        except:
            #make sure y_true and y_pred have the same length
            if len(y_true) != len(y_pred):
                raise ValueError(
                    f'If `y_true` and `y_pred` are 1d, they have to have the same lengths but have {len(y_true)} and {len(y_pred)}!'
                )
            else:
                pass

        #convert to numpy array (if lists got passed)
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        #check if 1d array was passed
        if len(y_true.shape) < 2:
            y_true = y_true.reshape(1,-1)
            warnings.warn(message=(
                f'`y_true` has to be two dimensional but has shape {y_true.shape}!\n'
                f'    Called `y_true.reshape(1,-1)`. Therefore you might not get the expected result.'
            ))
        if len(y_pred.shape) < 2:
            y_pred = y_pred.reshape(1,-1)
            warnings.warn(message=(
                f'`y_pred` has to be two dimensional but has shape {y_pred.shape}!'
                f'    Called `y_pred.reshape(1,-1)`. Therefore you might not get the expected result.'
            ))

        #get unique labels
        uniques = np.unique([y_true,y_pred])
        uniques = uniques[np.isfinite(uniques)]

        #initialize multi-confusion matrix
        multi_confmat = np.zeros((len(y_true), len(uniques), len(uniques)))

        #get confusion matrix for each model
        for midx, (yt, yp) in enumerate(zip(y_true, y_pred)):
            
            #remove non-finite values (padded values ect.)
            finite_bool = (np.isfinite(yt)&np.isfinite(yp))
            yt = yt[finite_bool]
            yp = yp[finite_bool]
            
            #get mapping of current unique labels to indices
            c_uniques = np.unique([yt,yp])
            idx_map = {c:np.where(uniques==c)[0][0] for c in c_uniques}
            
            #caluculate confusion matrix for current model (inverse definition to sklearn)
            cm = confusion_matrix(y_true=yt, y_pred=yp, normalize=normalize, sample_weight=sample_weight).T
            
            #include calculated confusion matrix in multi-confusion matrix
            for iidx, cui in enumerate(c_uniques):
                for jidx, cuj in enumerate(c_uniques):
                    multi_confmat[midx, idx_map[cui], idx_map[cuj]] = cm[iidx,jidx]
            
            #show resulting matrix
            if verbose > 3:
                CMD = ConfusionMatrixDisplay(multi_confmat[midx], display_labels=uniques)
                CMD.plot()
                plt.show()

        return multi_confmat

    def plot_bar(self,
        ax:plt.Axes,
        score:np.ndarray,
        m_labels:Union[list,Literal['score']]=None, score_decimals:int=None,
        text_colors:Union[str,tuple,list]=None,
        cmap:Union[str,mcolors.Colormap]=None, vmin:float=None, vmax:float=None, vcenter:float=None,
        ) -> None:
        """
            - method to create a bar-plot in one panel (`ax`)
                - i.e. confusion matrix cell of one class-combination

            Parameters
            ----------
                - `ax`
                    - plt.Axes
                    - axis to plot onto
                - `score`
                    - np.ndarray
                    - scores of one class-combination for all models
                - `m_labels`
                    - np.ndarray, Literal['score'], optional
                    - labels to show for the different models (entries in `score`)
                    - if `'score'` is passed
                        - will show the respective model's (normalized) score                    
                    - the default is `None`
                        - no labels shown
                - `score_decimals`
                    - int, optional
                    - number of decimals to round each model's `score` to when displaying
                    - only relevant if `m_labels == 'score'`
                    - overrides `self.score_decimals` 
                    - the default is `None`
                        - will fall back to `self.score_decimals`
                - `text_colors` 
                    - str, tuple, list, optional
                    - colors to use for displaying model/bar labels
                    - if str
                        - will use that color for all bars
                    - if tuple
                        - has to be RGBA tuple
                        - will use that color for all bars
                    - if list
                        - will use entry 0 for bar 0, entry 1 for bar 1 ect.
                    - the default is `None`
                        - will autogenerate colors
                        - will use the the last color of `cmap` for the first half of the bars
                        - will use the the first color of `cmap` for the bottom half of the bars
                - `cmap`
                    - str, mcolors.Colormap, optional
                    - colormap to use for coloring the different models
                    - overrides `self.cmap`
                    - the default is `None`
                        - will fall back to `self.cmap`
                - `vmin`
                    - float, optional
                    - minimum value of the colormapping
                    - used in scaling the colormap
                    - overrides `self.vmin`
                    - the default is `None`
                        - will fall back to `self.vmin`
                - `vmax`
                    - float, optional
                    - maximum value of the colormapping
                    - used in scaling the colormap
                    - argument of `astroLuSt.visualization.plotting.generate_colors()`
                    - overrides `self.vmax`
                    - the default is `None`
                        - will fall back to `self.vmax`
                - `vcenter`
                    - float, optional
                    - center value of the colormapping
                    - used in scaling the colormap
                    - argument of `astroLuSt.visualization.plotting.generate_colors()`
                    - overrides `self.vcenter`
                    - the default is `None`
                        - will fall back to `self.vcenter`

            Raises
            ------
                - TypeError
                    - if `m_labels` is of the wrong type

            Returns
            -------

            Comments
            --------
        """

        #default parameters
        if m_labels == 'score': m_labels = np.round(score, score_decimals)
        elif isinstance(m_labels, (list, np.ndarray)): m_labels = m_labels
        elif m_labels is None:  m_labels = []
        else: raise TypeError('`m_labels` has to be either a list, np.ndarray, or `"score"`')

        #generate colors for the bars and text (m_labels)
        colors = generate_colors(len(score)+1, vmin, vmax, vcenter, cmap=cmap)
        
        if text_colors is None:
            text_colors = colors.copy()
            text_colors[:len(text_colors)//2] = colors[-1]
            text_colors[len(text_colors)//2:] = colors[0]
        elif isinstance(text_colors, (str, tuple)):
            text_colors = [text_colors]*score.shape[0]

        
        #create barplor
        bars = ax.barh(
            y=np.arange(score.shape[0])[::-1], width=score,
            color=colors,
        )

        #add model labels if desired
        for idx, (b, mlab, tc) in enumerate(zip(bars, m_labels, text_colors)):
            ax.text(
                x=0.01*max(ax.get_xlim()), y=b.get_y()+b.get_height()/2,
                s=mlab,
                c=tc, va='center',
                # backgroundcolor='w'
            )
        
        ax.grid(visible=True, axis='x')

        return

    def plot_singlemodel(self,
        confmat:np.ndarray,
        labels:np.ndarray=None, score_decimals:int=None,
        text_colors:Union[str,tuple,list]=None,
        cmap:Union[str,mcolors.Colormap]=None, vmin:float=None, vmax:float=None,
        fig_kwargs:dict=None,
        pcolormesh_kwargs:dict=None,
        ) -> Tuple[Figure,plt.Axes]:
        """
            - method to produce classic confusion matrix
            - similar to `sklearn.metrics.ConfusionMatrixDisplay`
                - BUT axes defined inversely
                - https://scikit-learn.org/stable/modules/generated/sklearn.metrics.ConfusionMatrixDisplay.html#sklearn.metrics.ConfusionMatrixDisplay (last access: 2023/07/20)

            Parameters
            ----------
                - `confmat`
                    - np.ndarray
                    - array containing confusion matrix
                    - has to be of shape `(nclasses, nclasses)`
                - `labels`
                    - np.ndarray, optional
                    - labels to display for different classes present in `y_true` and `y_pred`
                    - will assign labels in ascending orders for the values present in `y_true` and `y_pred`
                    - the default is `None`
                        - will generate labels using `np.arange(confmats.shape[-1])`
                - `score_decimals`
                    - int, optional
                    - number of decimals to round each model's `score` to when displaying
                    - only relevant if `m_labels == 'score'`
                    - overrides `self.score_decimals` 
                    - the default is `None`
                        - will fall back to `self.score_decimals`
                - `text_colors` 
                    - str, tuple, list, optional
                    - colors to use displaying text in each cell
                    - if str
                        - will use that color for all cells
                    - if tuple
                        - has to be RGBA tuple
                        - will use that color for all cells
                    - list
                        - length has to be equal to `confmat.size`
                        - will display colors from top-left to bottom right (in reading direction)                    
                    - overwrites `self.text_colors`
                    - the default is `None`
                        - will fall back to `self.text_colors`
                        - will autogenerate colors
                            - inverse to cmap
                - `cmap`
                    - str, mcolors.Colormap, optional
                    - colormap to use for coloring the different models
                    - overrides `self.cmap`
                    - the default is `None`
                        - will fall back to `self.cmap`
                - `vmin`
                    - float, optional
                    - minimum value of the colormapping
                    - used in scaling the colormap
                    - overrides `self.vmin`
                    - the default is `None`
                        - will fall back to `self.vmin`
                - `vmax`
                    - float, optional
                    - maximum value of the colormapping
                    - used in scaling the colormap
                    - argument of `astroLuSt.visualization.plotting.generate_colors()`
                    - overrides `self.vmax`
                    - the default is `None`
                        - will fall back to `self.vmax`
                - `fig_kwargs`
                    - dict, optional
                    - kwargs to pass to `plt.figure()`
                    - overrides `self.fig_kwargs`
                    - the default is `None`
                        - will fall back to `self.fig_kwargs`
                - `pcolormesh_kwargs`
                    - dict, optional
                    - kwargs to pass to `ax.pcolormesh()`
                    - the default is `None`
                        - will be set to `dict()`

            Raises
            ------

            Returns
            -------
                - `fig`
                    - Figure
                    - created matplotlib figure
                - `axs`
                    - plt.Axes
                    - axes corresponding to `fig`

            Comments
            --------
        """

        if text_colors is None:         text_colors         = self.text_colors
        if cmap is None:                cmap                = self.cmap
        if score_decimals is None:      score_decimals      = self.score_decimals
        if fig_kwargs is None:          fig_kwargs          = self.fig_kwargs
        if pcolormesh_kwargs is None:   pcolormesh_kwargs   = dict()

        #generate colors for cell text
        colors = generate_colors(confmat.size, vmin=vmin, vmax=vmax, cmap=cmap)

        if text_colors is None:
            text_colors = np.empty_like(colors)
            idxs = np.argsort(confmat.flatten())[::-1]
            text_colors[idxs] = colors
            text_colors = text_colors.reshape(-1,confmat.shape[-1],4)
        elif isinstance(text_colors, str):
            text_colors = np.full(confmat.shape, text_colors)
        elif isinstance(text_colors, tuple):
            text_colors = np.full((*confmat.shape,4), text_colors)
        else:
            assert len(text_colors) == confmat.size, f'the length of `text_colors` has to be equal to `confmat.size`!'
            text_colors = np.array(text_colors)
            text_colors = text_colors.reshape(*confmat.shape,-1)


        #coordinates for plotting
        x = np.arange(confmat.shape[-1])
        
        #plot
        fig = plt.figure(**fig_kwargs)
        ax1 = fig.add_subplot(111)

        #plot confmat
        mesh = ax1.pcolormesh(x, x, confmat, cmap=cmap, vmin=vmin, vmax=vmax, **pcolormesh_kwargs)

        #add text
        for row in x:
            for col in x:
                c = text_colors[row, col]
                if isinstance(c[0], str):
                    c = c[0]
                else:
                    c = c
                ax1.text(
                    x=x[col], y=x[row],
                    s=np.round(confmat[row,col], score_decimals),
                    color=c, ha='center', va='center'
                ) 

        #labelling
        ax1.set_xticks(x, labels=labels[:x.shape[0]])
        ax1.set_yticks(x, labels=labels[:x.shape[0]])

        ax1.invert_yaxis()
        ax1.set_xlabel('True')
        ax1.set_ylabel('Predicted')

        axs = fig.axes

        return fig, axs

    def plot_multimodel(self,
        confmats:np.ndarray,
        labels:np.ndarray=None, m_labels:Union[np.ndarray,Literal['score']]=None, score_decimals:int=None,
        text_colors:Union[str,tuple,list]=None,
        cmap:Union[str,mcolors.Colormap]=None, vmin:float=None, vmax:float=None, vcenter:float=None,
        subplots_kwargs:dict=None,
        fig_kwargs:dict=None,
        ) -> Tuple[Figure,plt.Axes]:
        """
            - method to produce the confusion-matrix plot containing results of multiple models

            Parameters
            ----------
                - `confmats`
                    - np.ndarray
                    - array containing confusion matrices for the models
                    - has to be of shape `(nmodels, nclasses, nclasses)`
                - `labels`
                    - np.ndarray, optional
                    - labels to display for different classes present in `y_true` and `y_pred`
                    - will assign labels in ascending orders for the values present in `y_true` and `y_pred`
                    - the default is `None`
                        - will generate labels using `np.arange(confmats.shape[-1])`
                - `m_labels`
                    - np.ndarray, Literal['score'], optional
                    - labels to show for the different models (axis 1 of `y_true` and `y_pred`)
                    - if `'score'` is passed
                        - will show the respective model's (normalized) score                    
                    - the default is `None`
                        - no labels shown
                - `score_decimals`
                    - int, optional
                    - number of decimals to round each model's `score` to when displaying
                    - only relevant if `m_labels == 'score'`
                    - overrides `self.score_decimals` 
                    - the default is `None`
                        - will fall back to `self.score_decimals`
                - `text_colors` 
                    - str, tuple, list, optional
                    - colors to use for displaying model/bar labels
                    - if str
                        - will use that color for all bars
                    - if tuple
                        - has to be RGBA tuple
                        - will use that color for all bars
                    - if list
                        - will use entry 0 for bar 0, entry 1 for bar 1 ect.
                    - overwrites `self.text_colors`
                    - the default is `None`
                        - will fall back to `self.text_colors`
                - `cmap`
                    - str, mcolors.Colormap, optional
                    - colormap to use for coloring the different models
                    - overrides `self.cmap`
                    - the default is `None`
                        - will fall back to `self.cmap`
                - `vmin`
                    - float, optional
                    - minimum value of the colormapping
                    - used in scaling the colormap
                    - overrides `self.vmin`
                    - the default is `None`
                        - will fall back to `self.vmin`
                - `vmax`
                    - float, optional
                    - maximum value of the colormapping
                    - used in scaling the colormap
                    - argument of `astroLuSt.visualization.plotting.generate_colors()`
                    - overrides `self.vmax`
                    - the default is `None`
                        - will fall back to `self.vmax`
                - `vcenter`
                    - float, optional
                    - center value of the colormapping
                    - used in scaling the colormap
                    - argument of `astroLuSt.visualization.plotting.generate_colors()`
                    - overrides `self.vcenter`
                    - the default is `None`
                        - will fall back to `self.vcenter`
                - `subplots_kwargs`
                    - dict, optional
                    - kwargs to pass to `plt.subplots()`
                    - the default is `None`
                        - will be set to `dict(sharex='all', sharey='all')`
                - `fig_kwargs`
                    - dict, optional
                    - kwargs to pass to `plt.figure()`
                    - overrides `self.fig_kwargs`
                    - the default is `None`
                        - will fall back to `self.fig_kwargs`
                        
            Raises
            ------
                - `ValueError`
                    - if the length of `labels` is to low
                        - i.e. less than the unique elements in the combined set of `y_true` and `y_pred`

            Returns
            -------
                - `fig`
                    - Figure
                    - created matplotlib figure
                - `axs`
                    - plt.Axes
                    - axes corresponding to `fig`

            Comments
            --------
                
        """

        #default parameters
        if m_labels is None:        m_labels        = []
        if score_decimals is None:  score_decimals  = self.score_decimals
        if text_colors is None:     text_colors     = self.text_colors
        if cmap is None:            cmap            = self.cmap
        if subplots_kwargs is None: subplots_kwargs = dict(sharex='all', sharey='all')
        if fig_kwargs is None:      fig_kwargs      = self.fig_kwargs

        nrowscols = confmats.shape[-1]
        if labels is None: labels                   = np.arange(nrowscols)
        #catch errors
        if len(labels) < nrowscols: raise ValueError(f'`labels` has to be at least of length equal to the number of unique classes in `y_true` and `y_pred` ({nrowscols}) but has length {len(labels)}!')


        #plotting
        fig, axs = plt.subplots(
            nrows=nrowscols, ncols=nrowscols,
            **subplots_kwargs,
            **fig_kwargs,
        )
        
        for row in range(nrowscols):
            for col in range(nrowscols):

                #plot barchart
                self.plot_bar(
                    ax=axs[row,col],
                    score=confmats[:,row,col], m_labels=m_labels, score_decimals=score_decimals,
                    text_colors=text_colors,
                    cmap=cmap, vmin=vmin, vmax=vmax, vcenter=vcenter,
                )

                #set axis labels
                if col == 0:            axs[row,col].set_ylabel(labels[row])
                if row == nrowscols-1:  axs[row,col].set_xlabel(labels[col])
                
                axs[row,col].set_yticklabels([])

        #figure labels
        plt.figtext(0.5, 0.0, 'True',      rotation=0 , fontsize='large')
        plt.figtext(0.0, 0.5, 'Predicted', rotation=90, fontsize='large')

        plt.tight_layout()

        axs = fig.axes

        return fig, axs

    def plot_result(self,
        y_true:Union[np.ndarray,list], y_pred:Union[np.ndarray,list],
        confmats:np.ndarray=None,
        labels:np.ndarray=None,
        sample_weight:np.ndarray=None,
        normalize:Literal['true','pred','all']=None,
        plot_func:Literal['multi', 'single', 'auto']='auto',
        verbose:int=None,
        plot_multimodel_kwargs:dict=None,
        plot_singlemodel_kwargs:dict=None,
        ) -> Tuple[Figure,plt.Axes]:
        """
            - method to produce the plot of the multi-confusion-matrix

            Parameters
            ----------
                - `y_true`
                    - np.ndarray, list, optional
                    - ground truth labels
                    - has to be 2d
                        - `shape = (nmodels,nsampels)`
                        - `nsamples` can vary by model
                    - the default is `None`
                        - will use `confmats` instead of `y_true` and `y_pred`
                - `y_pred`
                    - np.ndarray, list, optional
                    - model predictions
                    - has to be 2d
                        - `shape = (nmodels,nsamples)`
                        - `nsamples` can vary by model
                    - the default is `None`
                        - will use `confmats` instead of `y_true` and `y_pred`
                - `confmats`
                    - np.ndarray, optional
                    - array containing confusion matrices for the models
                    - has to be of shape `(nmodels, nclasses, nclasses)`            
                    - if `y_true` and `y_pred` are also not None
                        - will use `y_true` and `y_pred` instead
                    - the default is `None`
                        - will use `y_true` and `y_pred` instead
                - `labels`
                    - np.ndarray, optional
                    - labels to display for different classes present in `y_true` and `y_pred`
                    - will assign labels in ascending orders for the values present in `y_true` and `y_pred`
                    - the default is `None`
                        - will autogenerate labels
                            - will use the unique values present in `y_true` and `y_pred` if both are not `None`
                            - will use `np.arange(confmats.shape[-1])` if `y_true` and `y_pred` are both `None`
                - `sample_weight`
                    - np.ndarray, optional
                    - sample weights to use in `sklearn.metrics.confusion_matrix()`
                    - the default is `None`
                - `normalize`
                    -  Literal['true','pred','all'], optinonal
                    - how to normalize the confusion matrix
                    - if `'true'` is passed
                        - normalize w.r.t. `y_true`
                    - if `'pred'` is passed
                        - normalize w.r.t. `y_pred`
                    - if `'all'` is passed
                        - normalize w.r.t. all confusion matrix cells
                    - will be passed to `sklearn.metrics.confusion_matrix()`
                    - the default is `None`
                        - no normalization
                - `plot_func`
                    - Literal['auto','multi','single'], optional
                    - method to use for deciding how to display the confusion matrix
                    - if `'auto'`
                        - will automatically decide
                    - if `'multi'`
                        - will use `self.plot_multimodel()` even if only one model passed
                    - if `single`
                        - will use `self.plot_singlemodel()`
                        - will plot the first entry of confmats even if multiple are passed
                            - i.e. `confmats[0]`
                    - the default is `auto`
                - `verbose`
                    - int, optional
                    - verbosity level
                    - overrides `self.verbose`
                    - the deafault is `None`
                        - will fall back to `self.verbose`
                - `plot_multimodel_kwargs`
                    - dict, optional
                    - kwargs to pass to `self.plot_multimodel()`
                    - the default is `None`
                        - will be set to `dict()`
                - `plot_singlemodel_kwargs`
                    - dict, optional
                    - kwargs to pass to `self.plot_singlemodel()`
                    - the default is `None`
                        - will be set to `dict()`

            Raises
            ------
                - `ValueError`
                    - if `y_true`, `y_pred`, and `confmats` are all `None`
                    - if `confmats` is not a square matrix
                    - if a wrong value is passed as `plot_func`

            Returns
            -------
                - `fig`
                    - Figure
                    - created matplotlib figure
                - `axs`
                    - plt.Axes
                    - axes corresponding to `fig`

            Comments
            --------            
        
        """
        
        #default values
        if plot_multimodel_kwargs is None:  plot_multimodel_kwargs  = dict()
        if plot_singlemodel_kwargs is None: plot_singlemodel_kwargs = dict()

        #get confusion matrices for all models (Transpose because sklearn.metric.confusion_matrix is inversely defined to this method)
        #initialize labels
        if y_true is not None and y_pred is not None:
            
            #generate multi-confusion matrix
            confmats = self.get_multi_confmat(y_true=y_true, y_pred=y_pred, sample_weight=sample_weight, normalize=normalize, verbose=verbose)

            if labels is None: labels = np.unique([y_true, y_pred])
            
        elif y_true is None or y_pred is None and confmats is not None:
            if labels is None: labels = np.arange(confmats.shape[-1])
            confmats = confmats
        else:
            raise ValueError(f'Either `y_true` and `y_pred` have to be not `None` or `confmats` has to be not `None`, but all are `None`!')
        
        #reshape confmats if wrong shape has been passed
        if len(confmats.shape) != 3:
            confmats = confmats.reshape(1, *confmats.shape)
        
        #check if all shapes are correct
        if confmats.shape[1] != confmats.shape[2]: raise ValueError(f'Confusion matrices have to be square matrices but `confmats` has shape {confmats.shape}')

        #decide on plotting strategy
        if plot_func == 'auto':
            if confmats.shape[0] == 1:  plot_func = 'single'
            else:                       plot_func = 'multi'

        #create plots
        if plot_func == 'multi':
            fig, axs = self.plot_multimodel(
                confmats=confmats,
                labels=labels,
                **plot_multimodel_kwargs,
            )
        elif plot_func == 'single':
            fig, axs = self.plot_singlemodel(
                confmats[0],
                labels=labels,
                **plot_singlemodel_kwargs,
            )
        else:
            raise ValueError(f'`plot_func` has to be one of `["multi", "single", "auto"] but is {plot_func}')

        return fig, axs
