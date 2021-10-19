
    ###################
    #Steinwender Lukas#
    ###################

#______________________________________________________________________________
#Class containing useful stuff for plotting
#TODO: get plot_ax(), color_generator(), hexcolor_extract(), inside this class
class Plot_LuSt:

    def __init__(self):
        pass

#______________________________________________________________________________
#TODO: add multiple x/y-axes (twinx)?
#TODO: add 3d plotting
#TODO: add verbose
def plot_ax(xvals, yvals, colors=None, markers=None, markersizes=None, labels=None, linestyles=None, alphas=None,  #curve specifications
            smooths=None, smoothdegrees=None, smoothresolutions=None,           #curve manipulations
            xerrs=None, yerrs=None, capsizes=None, errcolors=None,              #error specifications
            positions=None, zorders=None,                                       #positions/order specifications
            invert_xaxis=None, invert_yaxis=None,                               #axis-specifications
            xlabs=None, ylabs=None, suptitle="", num="",                        #titling specifications
            axlegend=True, figlegend=False, figsize=(16,9), fontsize=16,        #layout specifications
            timeit=False):                                                      #additional functionality
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
    xvals : list/np.ndarray
        list of x-values to plot.
    yvals : list/np.ndarray
        list of y-values corresponding to xvals.
    colors : list/np.ndarray, optional
        list of colors to use for plotting the datasets.
        The default is None, which will plot all in "tab:blue".
    markers : list/np.ndarray, optional
        list of markers to use for plotting the datasets.
        The default is None, which will result in all ".".
    markersizes : list/np.ndarray, optional
        list of markersiszes to use for plotting the datasets.
        The default is None, which will result in all haivng markersize=10.
    labels : list/np.ndarray, optional
        dataset-labels corresponding to the dataset of xvals.
        The default is None, which will result in no labels.
    linestyles : list/np.ndarray, optional
        list of linestyles to use for plotting the datasets.
        The default is None, which will result in all "".
    alphas : list/np.ndarray, optional
        list of the alphas to use on the curve.
        The default is None, which will result in all default values.
    smooths : list/np.ndarray of bools, optional
        list of wether to interpolate and smooth the curve or not.
        But there will not be errors in smoothed curves, even if passed to the function!
        The default is None, which will result in all curves not being smoothed (i.e. all set to False).
    smoothdegrees : list/np.ndarray, optional
        list of polynomial degrees to use when smoothing for each curve.
        The default is None which will create k=3 splines for smoothing
    smoothresolutions : 
        reolution of the smoothed curve, i.e. number of points to use for interpolation
        the default is None, which will result in all curves having a resolution of 100
    xerrs : list/np.ndarray, optional
        x-errors corresponding to xvals.
        The default is None.
    yerrs : list/np.ndarray, optional
        y-errors corresponding to yvals.
        The default is None.
    capsizes : list/np.ndarray, optional
        capsizes to use for plotting errorbars.
        The default is None.
    errcolors : list/np.ndarray, optional
        list of colors to use for plotting the errors.
        The default is None, which will plot all in "tab:blue".
    positions : list/np.array, optional
        list of positions of the datasets.
        used to position each dataset to its respective subplot.
        has to contain the "add_subplot" indexing system values (integer of length 3)
        The default None, which is to plot all dataset from top left to bottom right.
    zorders : list/np.array, optional
        list of values vor zorder.
        describes wich data-series is in front, second, in background etc.
        The default None, which will retain the order of the provided input data.
    invert_xaxis : list/np.array, optional
        list of booleans. If the entry is True the xaxis of the corresponding subplot will be inverted.
        The default is None, which will flip none of the xaxes.
    invert_yaxis : list/np.array, optional
        list of booleans. If the entry is True the yaxis of the corresponding subplot will be inverted.
        The default is None, which will flip none of the yaxes.
    xlabs : list/np.array, optional
        labels of the x-axes.
        The default is None, which will result in "x".
    ylabs : string, optional
        labels of the y-axes.
        The default is None, wich will result in "y".
    suptitle : string, optional
        suptitle of the figure.
        The default is "".
    num : str, optional
        number of the figure.
        The default is "".
    axlegend : bool, optional
        If set to True will create a legend for each axis
        The default is True
    figlegend : bool, optional
        If set to True will create a global legend for the figure instead of
            separate legends for each axis.
        The default is False
    figsize : tuple, optional
        size of the figure.
        The default is (16,9).
    fontsize : int, optional
        fontsize to use for labels and ticks.
        fontsize of title and legend will be adjusted accordingly.
        The default is 16.
    timeit : bool, optional
        Specify wether to time the task ad return the information or not.
        The default is False
        

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
    from utility_astroLuSt import Time_stuff
    
    #time execution
    if timeit:
        task = Time_stuff("plot_ax")
        task.start_task()
    
    ##########################################
    #initialize not provided data accordingly#
    ##########################################
    
    if colors is None:
        colors = copy.deepcopy(xvals)
        #initialize all colors with "tab:blue"
        for ciidx, ci in enumerate(colors):
            for cjidx, cj in enumerate(ci):
                colors[ciidx][cjidx] = "tab:blue"
        # print(colors)
    if markers is None:
        markers = copy.deepcopy(xvals)
        #initialize all markers with "."
        for miidx, mi in enumerate(markers):
            for mjidx, mj in enumerate(mi):
                markers[miidx][mjidx] = "."
        # print(markers)
    if markersizes is None:
        markersizes = copy.deepcopy(xvals)
        #initialize all markers with "."
        for miidx, mi in enumerate(markersizes):
            for mjidx, mj in enumerate(mi):
                markersizes[miidx][mjidx] = 10
        # print(markersizes)
    if labels is None:
        labels = copy.deepcopy(xvals)
        #initialize all labels with None
        for liidx, li in enumerate(labels):
            for ljidx, lj in enumerate(li):
                labels[liidx][ljidx] = None
        # print(labels)
    if linestyles is None:
        linestyles = copy.deepcopy(xvals)
        #initialize all linestyles with ""
        for liidx, li in enumerate(linestyles):
            for ljidx, lj in enumerate(li):
                linestyles[liidx][ljidx] = ""
        # print(linestyles)
    if alphas is None:
        alphas = copy.deepcopy(xvals)
        #initialize all alpha with None
        for aiidx, ai in enumerate(alphas):
            for ajidx, aj in enumerate(ai):
                alphas[aiidx][ajidx] = None
        # print(alphas)
    if smooths is None:
        smooths = copy.deepcopy(xvals)
        #initialize all smooths with None
        for siidx, si in enumerate(smooths):
            for sjidx, sj in enumerate(si):
                smooths[siidx][sjidx] = False
        # print(smooth)
    if smoothdegrees is None:
        smoothdegrees = copy.deepcopy(xvals)
        #initialize all smoothdegrees with None
        for siidx, si in enumerate(smoothdegrees):
            for sjidx, sj in enumerate(si):
                smoothdegrees[siidx][sjidx] = 3
        # print(smoothdegrees)
    if smoothresolutions is None:
        smoothresolutions = copy.deepcopy(xvals)
        #initialize all smoothresolutions with None
        for siidx, si in enumerate(smoothresolutions):
            for sjidx, sj in enumerate(si):
                smoothresolutions[siidx][sjidx] = 100
        # print(smoothresolutions)
    if xerrs is None:
        xerrs = copy.deepcopy(xvals)
        #initialize all xerrs with None
        for xiidx, xi in enumerate(xerrs):
            for xjidx, xj in enumerate(xi):
                xerrs[xiidx][xjidx] = None
        # print(xerrs)
    if yerrs is None:
        yerrs = copy.deepcopy(xvals)
        #initialize all yerrs with None
        for yiidx, yi in enumerate(yerrs):
            for yjidx, yj in enumerate(yi):
                yerrs[yiidx][yjidx] = None
        # print(yerrs)
    if capsizes is None:
        capsizes = copy.deepcopy(xvals)
        #initialize all capsizes with None
        for ciidx, ci in enumerate(capsizes):
            for cjidx, cj in enumerate(ci):
                capsizes[ciidx][cjidx] = None
        # print(capsizes)
    if errcolors is None:
        errcolors = copy.deepcopy(xvals)
        #initialize all colors with "tab:blue"
        for eciidx, eci in enumerate(errcolors):
            for ecjidx, ecj in enumerate(eci):
                errcolors[eciidx][ecjidx] = "tab:blue"
        # print(errcolors)
    if positions is None:
        v1 = int(np.ceil(np.sqrt(len(xvals))))
        positions = np.empty(len(xvals), dtype=int)
        for pidx, p in enumerate(positions):
            subfig_idx = int(str(v1)+str(v1)+str(pidx+1))
            positions[pidx] = subfig_idx
        # print(positions)
    if zorders is None:
        zorders = copy.deepcopy(xvals)
        #initialize all zorders with 1
        for ziidx, zi in enumerate(zorders):
            for zjidx, zj in enumerate(zi):
                zorders[ziidx][zjidx] = 1
        # print(zorders)
    if invert_yaxis is None:
        #initialize with all False
        invert_yaxis = [False]*len(xvals)
        # print(invert_yaxis)
    if invert_xaxis is None:
        #initialize with all False
        invert_xaxis = [False]*len(xvals)
        # print(invert_xaxis)
    if xlabs is None:
        #initialize all xaxis with "x"
        xlabs = ["x"]*len(xvals)
        # print(xlabs)
    if ylabs is None:
        #initialize all yaxis with "y"
        ylabs = ["y"]*len(xvals)
        # print(ylabs)
        
    
    #################################
    #check if all shapes are correct#
    #################################
    
    shape_check1_name = ["xvals", "yvals", "colors", "markers", "markersizes", "labels", "linestyles", "alphas",
                         "smooths", "smoothdegrees", "smoothresolutions",
                         "xerrs", "yerrs", "capsizes", "errcolors",
                         "zorders"]
    shape_check1 = [xvals, yvals, colors, markers, markersizes, labels, linestyles, alphas,
                    smooths, smoothdegrees, smoothresolutions,
                    xerrs, yerrs, capsizes, errcolors,
                    zorders] 
    for sciidx, sci in enumerate(shape_check1):
        for scjidx, scj in enumerate(shape_check1):
            var1 = shape_check1_name[sciidx]
            var2 = shape_check1_name[scjidx]
            if len(sci) == len(scj):
                for scii, scjj in zip(sci, scj):
                    if len(scii) != len(scjj):
                        raise ValueError("Shape of %s has to be shape of %s"%(var1, var2))

            else:
                raise ValueError("Shape of %s has to be shape of %s"%(var1, var2))
    
    
    shape_check2_name = ["positions",
                         "invert_xaxis", "invert_yaxis",
                         "xlabs", "ylabs"]
    shape_check2 = [positions,
                    invert_xaxis, invert_yaxis,
                    xlabs, ylabs]
    for sc2, sc2n in zip(shape_check2, shape_check2_name):
        if len(xvals) > len(sc2):
            raise ValueError("length of %s has to be smaller or equal length of %s"%("xvals", sc2n))

    ####################################
    #check if all datatypes are correct#
    ####################################
    
    if type(figlegend) != bool:
        raise ValueError("figlegend has to be of type bool!")
    if type(axlegend) != bool:
        raise ValueError("axlegend has to be of type bool!")
    
    

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
            if sm:
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

#############################
# TEST FOR plot_ax function #  
#############################
 
# import numpy as np
# import matplotlib.pyplot as plt
# xi = np.linspace(0,20,21)

# xvals = [[xi,xi],[xi],[xi]]
# yvals = [[xi,xi**2],[xi*3],[xi]]
# colors=[["r","g"],["r"],["k"]]
# markers = [[".","x"],[","],["^"]]
# markersizes = [[50,2],[12],[8]]
# labels = [["c1", "c2"],["c1"],["c2"]]
# linestyles = [["","-."],["-"],[":"]]
# alphas = [[0.5,1],[1],[1]]
# smooths = [[False, True], [True],[True]]
# smoothdegrees = [[3,3],[3],[2]]
# smoothresolutions = [[100,4],[4],[300]]

# xerrs = [[[0.5]*len(xi), [1]*len(xi)],[[2]*len(xi)],[None]]
# yerrs = [[[0.5]*len(xi), [1]*len(xi)],[[2]*len(xi)],[None]]
# capsizes = [[3,5],[3],[None]]
# errcolors = [["r","g"],["b"],["c"]]

# positions = [211, 224, 223]
# zorders = [[1,2],[1],[1]]

# invert_yaxis = [True, False, False]
# invert_xaxis = [False, True, False]

# yaxlabs = ["y1", "y2", "y3"]
# xaxlabs = ["x1", "x2", "x3"]

# suptitle="Testtitle"
# num = "Testplot"

# figlegend = True
# figsize = (5,5)
# fontsize = 16

# fig, axs = plot_ax(xvals, yvals, colors=colors, markers=markers, markersizes=markersizes, labels=labels, linestyles=linestyles, alphas=alphas,
#                     smooths=smooths, smoothdegrees=smoothdegrees, smoothresolutions=smoothresolutions,
#                     xerrs=xerrs, yerrs=yerrs, capsizes=capsizes, errcolors=errcolors,
#                     positions=positions, zorders=zorders,
#                     invert_yaxis=invert_yaxis, invert_xaxis=invert_xaxis,
#                     xlabs=xaxlabs, ylabs=yaxlabs, suptitle=suptitle, num=num,
#                     figlegend=figlegend, figsize=figsize, fontsize=fontsize,
#                     timeit=True)
# #use to share axis
# axs[1].get_shared_y_axes().join(axs[1],axs[2])
# axs[1].set_yticklabels([])
# plt.tight_layout()
# plt.show()



#______________________________________________________________________________
#function to generate as distinct colors as possible (useful for plotting)
#TODO: Not working yet
def color_generator(ncolors, testplot=False, timeit=False):
    
    import numpy as np
    import matplotlib.pyplot as plt
    from utility_astroLuSt import Time_stuff
    
    #time execution
    if timeit:
        task = Time_stuff("color_generator")
        task.start_task()

    
    # vhex = np.vectorize(hex)
    # colorrange = np.arange(0,ncolors,1)
    # hexcolors = vhex(colorrange)
    # print(hexcolors)
    
    nrgb = int(ncolors**(1/3))
    print(nrgb)
    R = np.linspace(0,1,nrgb)
    G = np.linspace(0,1,nrgb)
    B = np.linspace(0,1,nrgb)
    
    
    colors = []
    for r in R:
        for g in G:
            for b in B:
                color = (r,g,b)
                colors.append(color)
                
    print(len(colors))
    
    if testplot:
        x = np.arange(0,len(colors),1)
        y = x
        fig = plt.figure()
        plt.suptitle("Testplot to visualize colors extracted")
        plt.scatter(x, y, color=colors, marker=".", figure=fig)
        plt.xlabel("indices colors[\"hex\"]", fontsize=16)
        plt.ylabel("y=x", fontsize=16)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.show()
    
    #time execution
    if timeit:
        task.end_task()
    
    return 
    
    
    
# color_generator(1000, testplot=True)
  
#______________________________________________________________________________
#define function to get hex_values for 554 different colors
#simplyfies plotting with different colors

def hexcolor_extract(testplot=False, timeit=False):
    """
    
    file credits: https://cloford.com/resources/colours/500col.htm
    last acces: 11.02.2021
    extracts names and hex-values of 554 colors in more or less spektral order
    useful for not using same color twice in plotting
        to do this use:
            
        for c in colors["hex"]:
            x = np.array([...])
            y = np.array([...])
            plt.plot(x,y, color=c)
                
        where x and y are values for the data series
    Parameters
    ----------
    testplot : bool, optional
        Generates a plot to visualize all the extracted colors. The default is False.
    timeit : bool, optional
        Specify wether to time the task ad return the information or not.
        The default is False
        

    Returns
    -------
    colors : dict
        Dictonary of colors in more or less spectral order.

    """
    
    
    import re
    import numpy as np
    import matplotlib.pyplot as plt
    from utility_astroLuSt import Time_stuff
      
    
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
