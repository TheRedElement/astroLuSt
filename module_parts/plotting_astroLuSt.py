
    ###################
    #Steinwender Lukas#
    ###################

#______________________________________________________________________________
#Class containing useful stuff for plotting
#TODO: implement color_generator()
#TODO: plot_ax(): add multiple x/y-axes (twinx)?
#TODO: plot_ax(): add 3d plotting

class Plot_LuSt:
    """
        Class containing methods useful for plotting

        Methods
        -------
            - plot_ax
                --> function for easy multipanel plots
            - hexcolor_extract
                --> function for creating a (more or less) continuous color series
            - wavelength2rgb
                --> function to convert a given wavelength into its corresponding RGB-value
            - color_generator:
                --> function to generate a list of RGB-values for colors in spectral order

        Attributes
        ----------
            
        Dependencies
        ------------
            - numpy
            - matplotlib
            - scipy
            - copy
            - re
    
        Comments
        --------
    """

    def __init__(self):
        pass
        
    # def __repr__():

    def plot_ax(xvals, yvals, colors=None, markers=None, markersizes=None, labels=None, linestyles=None, alphas=None,  #curve specifications
                smooths=None, smoothdegrees=None, smoothresolutions=None,           #curve manipulations
                xerrs=None, yerrs=None, capsizes=None, errcolors=None,              #error specifications
                positions=None, zorders=None,                                       #positions/order specifications
                invert_xaxis=None, invert_yaxis=None,                               #axis-specifications
                xlabs=None, ylabs=None, suptitle="", num=None,                      #titling specifications
                axlegend=False, figlegend=False, figsize=(16,9), fontsize=16,        #layout specifications
                timeit=False, verbose=False):                                       #additional functionality
        #TODO: add multiple x/y-axes (twinx)?
        #TODO: add 3d plotting
        #TODO: make initialization correct (str could not be converted to float)
        """
            Function which automatically creates a figure with subplots fitting the
                provided list of arrays.
            Different parameters can be specified which can be specified in the
                fig.add_subplot() function as well.
            Will return the fugure as well as a list of the axes to use for later
                configurations.
            One has to call plt.show() outside of the function to show it
            
            
            Parameters
            ----------
                - xvals
                    --> list/np.ndarray
                    --> list of x-values to plot
                - yvals
                    --> list/np.ndarray
                    --> list of y-values corresponding to xvals
                - colors
                    --> list/np.ndarray, optional
                    --> list of colors to use for plotting the datasets.
                    --> the default is None
                        ~~> will plot all in "tab:blue"
                - markers
                    --> list/np.ndarray, optional
                    --> list of markers to use for plotting the datasets.
                    --> the default is None
                        ~~> will result in all "."
                - markersizes
                    --> list/np.ndarray, optional
                    --> list of markersiszes to use for plotting the datasets
                    --> The default is None
                        ~~> will result in all haivng markersize=10.
                - labels
                    --> list/np.ndarray, optional
                    --> dataset-labels corresponding to the dataset of xvals
                    --> the default is None
                        ~~>will result in no labels.
                - linestyles
                    --> list/np.ndarray, optional
                    --> list of linestyles to use for plotting the datasets.
                    --> the default is None
                        ~~> will result in all ""
                - alphas
                    --> list/np.ndarray, optional
                    --> list of the alphas to use on the curve
                    --> the default is None
                        ~~> will result in all default values
                - smooths
                    --> list/np.ndarray of bools, optional
                    --> list of wether to interpolate and smooth the curve or not
                        ~~> there will not be x/y-errors in smoothed curves, even if passed to the function!
                    --> the default is None
                        ~~> will result in all curves not being smoothed (i.e. all set to False)
                - smoothdegrees
                    --> list/np.ndarray, optional
                    --> list of polynomial degrees to use when smoothing for each curve
                    --> the default is None
                        ~~> will create k=3 splines for smoothing
                - smoothresolutions
                    --> list/np.ndarray, optional
                    --> reolution of the smoothed curve, i.e. number of points to use for interpolation
                    --> the default is None
                        ~~> will result in all curves having a resolution of 100
                - xerrs
                    --> list/np.ndarray, optional
                    --> x-errors corresponding to xvals
                    --> the default is None
                - yerrs
                    --> list/np.ndarray, optional
                    --> y-errors corresponding to yvals.
                    --> the default is None.
                - capsizes
                    --> list/np.ndarray, optional
                    --> capsizes to use for plotting errorbars
                    --> The default is None
                - errcolors 
                    --> list/np.ndarray, optional
                    --> list of colors to use for plotting the errors
                    --> the default is None
                        ~~> will plot all in "tab:blue"
                - positions 
                    --> list/np.array, optional
                    --> list of positions of the datasets.
                    --> used to position each dataset to its respective subplot.
                    --> has to contain matplotlibs "add_subplot" indexing system values (integer of length 3)
                    --> the default None
                        ~~> plot all dataset from top left to bottom right.
                - zorders
                    --> list/np.array, optional
                    --> list of values for zorder
                    --> describes wich data-series is in front, second, in background etc.
                        ~~> the lower the number the further back
                    --> the default None
                        ~~> will retain the order of the provided input data.
                - invert_xaxis
                    --> list/np.array, optional
                    --> list of booleans
                        ~~> if the entry is True the xaxis of the corresponding subplot will be inverted
                    --> the default is None
                        ~~> will flip none of the xaxes
                - invert_yaxis
                    --> list/np.array, optional
                    --> list of booleans
                        ~~> if the entry is True the yaxis of the corresponding subplot will be inverted
                    --> the default is None
                        ~~> will flip none of the yaxes
                - xlabs
                    --> list/np.array, optional
                    --> labels of the x-axes
                    --> the default is None
                        ~~> will result in "x".
                - ylabs
                    --> list/np.array, optional
                    --> labels of the y-axes
                    --> the default is None
                        ~~> will result in "y".
                - suptitle
                    --> string, optional
                    --> suptitle of the figure
                    --> the default is "".
                - num
                    --> str, optional
                    --> number of the figure
                    --> the default is None.
                - axlegend
                    --> bool, optional
                    --> if set to True will create a legend for each axis
                    --> the default is None
                - figlegend
                    --> bool, optional
                    --> if set to True will create a global legend for the figure instead of
                        separate legends for each axis
                    --> the default is None
                - figsize
                    --> tuple, optional
                    --> size of the figure
                    --> the default is (16,9)
                - fontsize
                    --> int, optional
                    --> fontsize to use for labels and ticks
                        ~~> fontsize of title and legend will be adjusted accordingly
                    --> The default is None
                        ~~> Will result in 16
                - timeit
                    --> bool, optional
                    --> specify wether to time the task and return the information or not.
                    --> the default is False
                - verbose
                    --> bool, optional
                    --> specify wether to print additional information implemented in the function
                    --> the default is False
                
        
            Returns
            -------
            fig : matplotlib.figure - object
                created figure
            axs : matplotlib axes - object
                axes of created figure to change settings later on
        
            Comments
            --------
                If one wants to share axis between plots simply call
                >>>axs[iidx].get_shared_y_axes().join(axs[iidx],axs[jidx])
                >>>axs[iidx].set_yticklabels([])
                axs is hereby the second return value of the function and thus a list
                    of all axes
        
            Dependencies
            ------------
            numpy
            matplotlib
            scipy
            copy
        """
        import numpy as np
        import matplotlib.pyplot as plt
        from scipy.interpolate import make_interp_spline, BSpline #used for smoothing
        import copy
        from module_parts.utility_astroLuSt import Time_stuff
        
        ###################################################
        #Check if xvals and yvals are of the correct shape#
        ###################################################

        for vals, valname in zip([xvals, yvals], ["xvals", "yvals"]):
            try:
                len(vals)
            except:
                raise ValueError(f"{valname:s} has to be an array of shape (k,l,m)!\n"
                                 "Here k is the number of subplots created,\n"
                                 "l is the number of curves in the respective subplot (can differ for different k),\n"
                                 "m is the number of datapoints in the respective curve (can differ for different l).\n"
                                 f"Try something like:[[np.array([{valname:s}(curve1(subplot1))],curve2,..)],subplot2,...]"
                                 )
            for xv in vals:
                try:
                    len(xv) 
                except:
                    raise ValueError(f"{valname:s} has to be an array of shape (k,l,m)!\n"
                                     "Here k is the number of subplots created,\n"
                                     "l is the number of curves in the respective subplot (can differ for different k),\n"
                                     "m is the number of datapoints in the respective curve (can differ for different l).\n"
                                     f"Try something like:[[np.array([{valname:s}(curve1(subplot1))],curve2,..)],subplot2,...]"
                                     )
                for xvi in xv:
                    try:
                        len(xvi)
                    except:
                        raise ValueError(f"{valname:s} has to be an array of shape (k,l,m)!\n"
                                         "Here k is the number of subplots created,\n"
                                         "l is the number of curves in the respective subplot (can differ for different k),\n"
                                         "m is the number of datapoints in the respective curve (can differ for different l).\n"
                                         f"Try something like:[[np.array([{valname:s}(curve1(subplot1))],curve2,..)],subplot2,...]"
                                         )


        ###################################
        #Initialize parameters accordingly#
        ###################################

        if colors is None:
            colors = copy.deepcopy(xvals)
            #initialize all colors with "tab:blue"
            for ciidx, ci in enumerate(colors):
                for cjidx, cj in enumerate(ci):
                    colors[ciidx][cjidx] = "tab:blue"
        if markers is None:
            markers = copy.deepcopy(xvals)
            #initialize all markers with "."
            for miidx, mi in enumerate(markers):
                for mjidx, mj in enumerate(mi):
                    markers[miidx][mjidx] = "."
        if markersizes is None:
            markersizes = copy.deepcopy(xvals)
            #initialize all markers with "."
            for miidx, mi in enumerate(markersizes):
                for mjidx, mj in enumerate(mi):
                    markersizes[miidx][mjidx] = 10
        if labels is None:
            labels = copy.deepcopy(xvals)
            #initialize all labels with None
            for liidx, li in enumerate(labels):
                for ljidx, lj in enumerate(li):
                    labels[liidx][ljidx] = None
        if linestyles is None:
            linestyles = copy.deepcopy(xvals)
            #initialize all linestyles with ""
            for liidx, li in enumerate(linestyles):
                for ljidx, lj in enumerate(li):
                    linestyles[liidx][ljidx] = ""
        if alphas is None:
            alphas = copy.deepcopy(xvals)
            #initialize all alphas with None
            for aiidx, ai in enumerate(alphas):
                for ajidx, aj in enumerate(ai):
                    alphas[aiidx][ajidx] = None
        if smooths is None:
            #initialize all smooths with False
            smooths = copy.deepcopy(xvals)
            for siidx, si in enumerate(smooths):
                for sjidx, sj in enumerate(si):
                    smooths[siidx][sjidx] = False
        if smoothdegrees is None:
            #initialize all smoothdegrees with 3
            smoothdegrees = copy.deepcopy(xvals)
            for siidx, si in enumerate(smoothdegrees):
                for sjidx, sj in enumerate(si):
                    smoothdegrees[siidx][sjidx] = 3
        if smoothresolutions is None:
            #initialize all smoothresolutinos with 100
            smoothresolutions = copy.deepcopy(xvals)
            for siidx, si in enumerate(smoothresolutions):
                for sjidx, sj in enumerate(si):
                    smoothresolutions[siidx][sjidx] = 100
        if xerrs is None:
            #initialize all xerrs with 0
            xerrs = copy.deepcopy(xvals)
            for xiidx, xi in enumerate(xerrs):
                for xjidx, xj in enumerate(xi):
                    xerrs[xiidx][xjidx] = None
        if yerrs is None:
            #initialize all yerrs with 0
            yerrs = copy.deepcopy(xvals)
            for yiidx, yi in enumerate(yerrs):
                for yjidx, yj in enumerate(yi):
                    yerrs[yiidx][yjidx] = None
        if capsizes is None:
            #initialize all capsizes with None
            capsizes = copy.deepcopy(xvals)
            for ciidx, ci in enumerate(capsizes):
                for cjidx, cj in enumerate(ci):
                    capsizes[ciidx][cjidx] = None
        if errcolors is None:
            #initialize all colors with "tab:blue"
            errcolors = copy.deepcopy(xvals)
            for eciidx, eci in enumerate(errcolors):
                for ecjidx, ecj in enumerate(eci):
                    errcolors[eciidx][ecjidx] = "tab:blue"
        if positions is None:
            v1 = int(np.ceil(np.sqrt(len(xvals))))
            positions = np.empty(len(xvals), dtype=int)
            for pidx, p in enumerate(positions):
                subfig_idx = int(str(v1)+str(v1)+str(pidx+1))
                positions[pidx] = subfig_idx
        if zorders is None:
            #initialize all zorders with 1
            zorders = copy.deepcopy(xvals)
            for ziidx, zi in enumerate(zorders):
                for zjidx, zj in enumerate(zi):
                    zorders[ziidx][zjidx] = 1
        if invert_yaxis is None:
            #initialize with all False
            invert_yaxis = [False]*len(xvals)
        if invert_xaxis is None:
            #initialize with all False
            invert_xaxis = [False]*len(xvals)
        if xlabs is None:
            #initialize all xaxis with "x"
            xlabs = ["x"]*len(xvals)
        if ylabs is None:
            #initialize all yaxis with "y"
            ylabs = ["y"]*len(xvals)
        if suptitle is None:
            suptitle = ""

        #################################
        #check if all shapes are correct#
        #################################
            
        shape_check1_name = ["xvals", "yvals", "colors", "markers", "markersizes", "labels", "linestyles", "alphas",
                             "smooths", "smoothdegrees", "smoothresolutions",
                             "xerrs", "yerrs",
                             "capsizes", "errcolors",
                             "zorders"]
        shape_check1 = [xvals, yvals, colors, markers, markersizes, labels, linestyles, alphas,
                        smooths, smoothdegrees, smoothresolutions,
                        xerrs, yerrs,
                        capsizes, errcolors,
                        zorders] 
        for sciidx, sci in enumerate(shape_check1):
            # print(shape_check1_name[sciidx], "=", sci)
            for scjidx, scj in enumerate(shape_check1):
                var1 = shape_check1_name[sciidx]
                var2 = shape_check1_name[scjidx]
                # print(var1, "vs", var2)
                if len(sci) == len(scj):
                    for scii, scjj in zip(sci, scj):
                        if len(scii) != len(scjj):
                            raise ValueError(f"Shape of {var1:s} has to be shape of {var2:s}")

                else:
                    raise ValueError(f"Shape of {var1:s} has to be shape of {var2:s}")
        
        shape_check2_name = ["positions",
                             "invert_xaxis", "invert_yaxis",
                             "xlabs", "ylabs"]
        shape_check2 = [positions,
                        invert_xaxis, invert_yaxis,
                        xlabs, ylabs]
        for sc2, sc2n in zip(shape_check2, shape_check2_name):
            # print(sc2n, "=", sc2)
            if len(xvals) > len(sc2):
                raise ValueError(f"length of {'xvals':s} has to be smaller or equal length of {sc2n:s}"%("xvals", sc2n))

        ####################################
        #check if all datatypes are correct#
        ####################################
            
        if num != None and type(num) != str:
            raise ValueError("num has to be either None or some string")
        if type(figlegend) != bool:
            raise TypeError("figlegend has to be of type bool!")
        if type(axlegend) != bool:
            raise TypeError("axlegend has to be of type bool!")
        
        #time execution
        if timeit:
            task = Time_stuff("plot_ax")
            task.start_task()   

        ###################################
        #create subplots - actual function#
        ###################################
        
        fig = plt.figure(num=num, figsize=figsize)      #create figure, assign number and size
        plt.suptitle(suptitle, fontsize=fontsize+2)     #create suptitle of specified
        
        #iterate over all lists to get specifications for corresponding subplot
        for xi, yi, color, marker, markersize, label, linestyle, alpha, smooth, smoothdegree, smoothresolution, xerr, yerr, capsize, errcolor, position, zorder, invert_x, invert_y, xl, yl in zip(xvals, yvals, colors, markers, markersizes, labels, linestyles, alphas,
                                                                                                                                                                                                   smooths, smoothdegrees, smoothresolutions,
                                                                                                                                                                                                   xerrs, yerrs, capsizes, errcolors,
                                                                                                                                                                                                   positions, zorders,
                                                                                                                                                                                                   invert_xaxis, invert_yaxis,
                                                                                                                                                                                                   xlabs, ylabs):
            ax = fig.add_subplot(position)                  #add subplot on specified positions
            ax.set_xlabel(xl, fontsize=(fontsize-2))        #set xlabel if specified, else set to "x"
            ax.set_ylabel(yl, fontsize=(fontsize-2))        #set ylabel if specified, else set to "y"
            ax.tick_params(axis="x", labelsize=fontsize-2)  #set size of xticks
            ax.tick_params(axis="y", labelsize=fontsize-2)  #set size of yticks

            curvenumber = 0
            #iterate over all datasets to get specifications of dataset for current subplot
            for xii, yii, ci, marki, marksi, labi, stylei, alphi, sm, smd, smr, xerri, yerri, capsizei, eci, zi in zip(xi, yi, color, marker, markersize, label, linestyle, alpha,
                                                                                                                       smooth, smoothdegree, smoothresolution,
                                                                                                                       xerr, yerr, capsize, errcolor,
                                                                                                                       zorder):
                
                #smooth if smoothig is pecified for the curve is true
                if sm and verbose:
                    print("\n---------------------------------------")
                    print("WARNING: Smoothing the following curve:")
                    print("position: %s, curve number: %i"%(position, curvenumber))
                    print("The errors attached to this curve will")
                    print("be deleted for the plotting process!")                
                    print("---------------------------------------\n")
                    sortidx = np.argsort(xii)                       #get sorted indices
                    xii, yii = xii[sortidx], yii[sortidx]           #sort arrays (spline needs sorted array)
                    spl = make_interp_spline(xii, yii, k=smd)       #type: BSpline
                    xii = np.linspace(xii.min(), xii.max(), smr)    #redefine xii to be smoothed version
                    yii = spl(xii)                                  #redefine yii to be smoothed version
                    xerri = None                                    #ignore errors
                    yerri = None                                    #ignore errors
                
                
                #create errorbar with specified parameters plot: If no errors are specified == normal plot
                ax.errorbar(xii, yii, xerr=xerri, yerr=yerri, capsize=capsizei, zorder=zi, alpha=alphi,
                            color=ci, ecolor=eci, linestyle=stylei, marker=marki, markersize=marksi, figure=fig, label=labi)
                
                #create legend for axis, if figlegend is False
                if (not figlegend and axlegend):
                    ax.legend(fontsize=fontsize-2)
                    
                curvenumber += 1 #update index of curve
            
            #if invert_y is True invert the y axis for the corresponding subplot
            if invert_y:
                ax.invert_yaxis()    
            #if invert_x is True invert the x axis for the corresponding subplot
            if invert_x:
                ax.invert_xaxis()    
                
        axs = fig.axes          #extract all axes
        
        #create global legend for all subplots
        if figlegend:
            lines_labels = [ax.get_legend_handles_labels() for ax in axs]
            lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
            plt.figlegend(lines, labels, fontsize=fontsize-2)

        #time execution
        if timeit:
            task.end_task()

        
        return fig, axs

    def hexcolor_extract(testplot=False, timeit=False):
        """
            - File credits: https://cloford.com/resources/colours/500col.htm
                --> last acces: 11.02.2021
            - extracts names and hex-values of 554 colors in more or less spectral order
            - useful for not using same color twice in plotting
                --> to do this use:

                    for c in colors["hex"]:
                        x = np.array([...])
                        y = np.array([...])
                        plt.plot(x,y, color=c)

                    where x and y are values for the data series

            Parameters
            ----------
                - testplot
                    --> bool, optional
                    --> generates a plot to visualize all the extracted colors
                    --> the default is False.
                - timeit
                    --> bool, optional
                    --> specify wether to time the task and return the information or not.
                    --> the default is False

            Returns
            -------
                - colors
                    --> dict
                    --> dictonary of colors in more or less spectral order.

            Comments
            --------

            Dependencies
            ------------
                - re
                - numpy
                - matplotlib        
        """
        import re
        import numpy as np
        import matplotlib.pyplot as plt
        from module_parts.utility_astroLuSt import Time_stuff
        
        #time execution
        if timeit:
            task = Time_stuff("hexcolor_extract")
            task.start_task()

        infile = open("files/Colorcodes.txt", "r")
        colors = {"names":[],"hex":[]}
        lines = infile.readlines()
        for line in lines:
            
            #get all color names from given file and append to colors dictionary
            col_names = re.findall(".*?(?=\t\s\t)",line)    #capture everything bevor tabspacetab
            if len(col_names) != 0:
                colors["names"].append(col_names[0])
            
            #get all color hex_values from given file and append to colors dictionary        
            col_hex = re.findall("(?<=\#)\w+", line)    #capture everything after #
            if len(col_hex) != 0:
                colors["hex"].append("#" + col_hex[0])                  
            
            infile.close()
        
        if testplot:
            x = np.arange(0,len(colors["hex"]),1)
            y = x
            fig = plt.figure()
            plt.suptitle("Testplot to visualize colors extracted")
            plt.scatter(x, y, color=colors["hex"], marker=".", figure=fig)
            plt.xlabel("indices colors[\"hex\"]", fontsize=16)
            plt.ylabel("y=x", fontsize=16)
            plt.xticks(fontsize=16)
            plt.yticks(fontsize=16)
            plt.show()
            
        #time execution
        if timeit:
            task.end_task()
                
        return colors

    def wavelength2rgb(wavelength, gamma=0.8):
        """
            - function to convert a given wavelength of visible light to a RGB value.
            - wavelength in nanometers (380 nm - 750 nm)
            - Code from: http://www.noah.org/wiki/Wavelength_to_RGB_in_Python

            Parameters
            ----------
                - wavelength
                    --> float
                    --> the wavelength to convert
                - gamma
                    --> float, optional
                    --> some scaling exponent
                        ~~> will change the colors
                        ~~> will change the smoothness of the boundaries between 2 colors

            Returns
            -------
                - RGB
                    --> an array of RGB-values of the inserted wavelength

            Comments
            --------

            Dependencies
            ------------
                - numpy
        """
        import numpy as np

        #check if wavelength is in corret range
        if wavelength < 380 or wavelength > 750:
            raise ValueError("wavelength has to be in range(380, 750) nanometers!")

        wavelength = float(wavelength)
        if wavelength >= 380 and wavelength <= 440:
            attenuation = 0.3 + 0.7 * (wavelength - 380) / (440 - 380)
            R = ((-(wavelength - 440) / (440 - 380)) * attenuation) ** gamma
            G = 0.0
            B = (1.0 * attenuation) ** gamma
        elif wavelength >= 440 and wavelength <= 490:
            R = 0.0
            G = ((wavelength - 440) / (490 - 440)) ** gamma
            B = 1.0
        elif wavelength >= 490 and wavelength <= 510:
            R = 0.0
            G = 1.0
            B = (-(wavelength - 510) / (510 - 490)) ** gamma
        elif wavelength >= 510 and wavelength <= 580:
            R = ((wavelength - 510) / (580 - 510)) ** gamma
            G = 1.0
            B = 0.0
        elif wavelength >= 580 and wavelength <= 645:
            R = 1.0
            G = (-(wavelength - 645) / (645 - 580)) ** gamma
            B = 0.0
        elif wavelength >= 645 and wavelength <= 750:
            attenuation = 0.3 + 0.7 * (750 - wavelength) / (750 - 645)
            R = (1.0 * attenuation) ** gamma
            G = 0.0
            B = 0.0
        else:
            R = 0.0
            G = 0.0
            B = 0.0
        R *= 255
        G *= 255
        B *= 255

        RGB = np.array([int(R), int(G), int(B)])
        return RGB

    def color_generator(ncolors, wavelength_range=[380, 750], gamma=0.8, testplot=False, timeit=False):
        """
            - function to create a list of ncolors RGB-values for colors in spectral order

            Parameters
            ----------
                - ncolors
                    --> int, optional
                    --> number of colors to generate
                -wavelength_range
                    --> list/np.array, optional
                    --> list containing the maximum and minimum wavelength to consider for the color generation
                    --> the default is [380, 750]
                - gamma
                    --> float, optional
                    --> some scaling exponent
                        ~~> will change the colors
                        ~~> will change the smoothness of the boundaries between 2 colors
                - testplot
                    --> bool, optional
                    --> wether to create a testplot to visualize the created colors
                    --> the default is False
                - timeit
                    --> bool, optional
                    --> specify wether to time the task and return the information or not.
                    --> the default is False    
            Returns
            -------
                - colors
                    --> an array of the generated RGB-values in spectral order

            Comments
            --------

            Dependencies
            ------------
                - numpy
                - matplotlib
        """
        import numpy as np
        import matplotlib.pyplot as plt
        from module_parts.utility_astroLuSt import Time_stuff
        
        #time execution
        if timeit:
            task = Time_stuff("color_generator")
            task.start_task()

        wavelengths = np.linspace(np.min(wavelength_range), np.max(wavelength_range), ncolors)        
        colors = []
        for wl in wavelengths:
            c = Plot_LuSt.wavelength2rgb(wl, gamma=gamma) / 255
            colors.append(c)

        if testplot:
            x = np.arange(0,len(colors),1)
            y = x
            fig = plt.figure()
            plt.suptitle("Testplot to visualize generated colors", fontsize=18)
            plt.scatter(x, y, color=colors, marker=".", figure=fig)
            plt.xlabel("indices colors", fontsize=16)
            plt.ylabel("y=x", fontsize=16)
            plt.xticks(fontsize=16)
            plt.yticks(fontsize=16)
            plt.show()
        
        #time execution
        if timeit:
            task.end_task()
        
        return colors