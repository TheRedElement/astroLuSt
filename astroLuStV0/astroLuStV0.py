
    ###################
    #Steinwender Lukas#
    ###################



#______________________________________________________________________________
#Class to time tasks
class Time_stuff:
    """
        Class to time the runnig-time of code from one point in the code
        to another.
        
        The function then prints the starttime, endtime and needed time

        Methods
        -------
            - start_task:   saves the point in time for a task as starting point
            - end_task:     saves the point in time for a task as starting point
        
        Attributes
        ----------
            - task
                --> str
                --> name of the task
            - start
                --> datetime object, optional
                --> the point in time, at which the process started
                --> the default is None, since it will get set by start_task
            - end
                --> datetime object, optional
                --> the point in time, at which the process ended
                --> the default is None, since it will get set by end_task

        Dependencies
        ------------
            - datetime


        Comments
        --------
    """

    def __init__(self, task):        
        self.task = task
        self.start = None
        self.end = None
        self.needed_time = None

    def __repr__(self):
        return f"time_stuff('{self.task}', {self.start}, {self.end}, {self.needed_time})"

    def start_task(self):
        """
            Function to denote the starting point of a task.

            Parameters
            ----------
            
            Raises
            ------

            Returns
            -------

            Dependencies
            ------------
                - datetime

            Comments
            --------
        """
        from datetime import datetime

        self.start = datetime.now()
        printing = self.start.strftime("%Y-%m-%d %H:%M:%S.%f")
        print(f"--> Started {self.task} at time {printing}")
    
    def end_task(self):
        """
            Function to denote the endpoint of a task.
            Also prints out the time needed for the execution.

            Parameters
            ----------

            Raises
            ------

            Returns
            -------

            Dependencies
            ------------
                - datetime

            Comments
            --------
            Needs to be called after start_task
        """
        from datetime import datetime

        self.end = datetime.now()
        self.needed_time = (self.end - self.start)
        printing = self.end.strftime("%Y-%m-%d %H:%M:%S.%f")
        print(f"--> Time needed for {self.task}: {self.needed_time}")
        print(f"--> Finished {self.task} at time {printing}")


# task1 = Time_stuff("T1")
# task1.start_task()
# for i in range(10**7):
#     i2 = i+1
# task1.end_task()
# print(task1)

    
#______________________________________________________________________________
#Class for printing tables
#TODO: Add method to output latex template for table
#TODO: Add something like add_section()
#%%
class Table_LuSt:
    """
        Class to quickly print nice tables
        
        Methods
        -------
            - print_header: prints out the header of the table
            - print_rows:   prints out the rows of the table
            - print_table:  prints out the whole table
            - latex_template: TODO
        
        Attributes
        ----------
            - rows
                --> list of lists, optional
                --> list containing
                    ~~> a list for each row containing
                        + the entries for each cell in that row
                --> each row has to have the same length
                --> the default is None
            - header
                --> list, optional
                --> list containing
                    ~~> the entries for the header
                --> the default is None
            - formatstr
                --> list of lists, optional
                --> list containing
                    ~~> a list for each row containing
                        + the formatstring for each cell in that row
                --> has to be of same dimension as rows
                --> the default is None

        Dependencies
        ------------
            - re


        Comments
        --------
    """

    def __init__(self, rows=None, header=None, formatstr=None):
        if rows is None:
            self.rows = []
        #check if all rows have the same length
        elif all(len(row) == len(next(iter(rows))) for row in iter(rows)):
            self.rows = rows
        else:
            raise ValueError("All lists in rows (all rows) have to have the same length!")
        if header is None:
            self.header = []
        if formatstr is None:
            self.formatstr = []
        else:
            self.formatstr = formatstr
        self.num_width = str(len(str(len(self.rows)))+2)   #length of the string of the number of rows +2
        self.tablewidth = None  #width of the output table

    def __repr__(self):
        return ("\n"
                f"Table_LuSt(\n"
                f"rows = {self.rows},\n" 
                f"header = {self.header},\n"
                f"formatstr = {self.formatstr})")
        
    def add_row(self, row, fstring=None):
        """
            Function to add another row to the table.
            Adding a row also results in adding a formatstring for this row.

            Parameters
            ----------
                - row
                    --> list
                    --> list containing
                        ~~> the entries for each cell
                -fstring
                    --> list, optional
                    --> list containing
                        ~~> the corresponding formatstrings for the entries
                    --> the default is None
            
            Raises
            ------
                -ValueError
                    --> If the row and fstring are not of the same length

            Returns
            -------

            Dependencies
            ------------

            Comments
            --------
        """
        self.rows.append(row)
        if fstring is None and len(self.formatstr) == 0:
            fstring = [""]*len(row)
        elif fstring is None and len(self.formatstr) != 0:
            fstring = self.formatstr[0]
        self.formatstr.append(fstring)

        if len(row) != len(fstring):
            raise ValueError("len(row) as to be equal to len(fstring)")

    def print_header(self):
        """
            Function to print out the header of the created table.

            Parameters
            ----------
            
            Raises
            ------

            Returns
            -------

            Dependencies
            ------------
                - re

            Comments
            --------
                - sets self.tablewidth to the width of the header in characters
        """
        import re
        
        #create empty header if none was provided
        if len(self.header) == 0:
            self.header = [""]*len(self.rows[0])
        
        #initialize header with a rownumber column
        header_print = "{:^"+self.num_width+"}|"
        header_print = header_print.format("#")
        #fill in the rest of the header and print it
        for fs in self.formatstr[0]:
            fs_digit = re.findall(r"\d+", fs)[0]
            header_print += "{:^"+fs_digit+"s}|"
        to_print = header_print[:-1].format(*self.header)
        print(to_print)
        self.tablewidth = len(to_print)  #total width the table will have 

    def print_rows(self):
        """
            Function to print out the rows of the table.

            Parameters
            ----------
            
            Raises
            ------

            Returns
            -------

            Dependencies
            ------------
            
            Comments
            --------
        """
        #fill formatstring list, if not enough formatstrings were provided
        if len(self.rows) < len(self.formatstr):
            diff = abs(len(self.rows) - len(self.formatstr))
            to_add = [self.formatstr[-1]]*diff
            for ta in to_add:
                self.formatstr.insert(0, ta)
        row_num = 1 #keep track of the rownumber to enumerate table
        
        #print rows
        for row, fstring in zip(self.rows, self.formatstr):
            #initialize row with its rownumber
            row_print = "{:^"+self.num_width+"}|"
            row_print = row_print.format(row_num)
            #add row-entries
            for fs in fstring:
                row_print += "{:^"+fs[1:]+"}|"
            row_num += 1
            print(row_print[:-1].format(*row))

    def print_table(self):
        #TODO: Add something like add_section()
        """
            Function to combine print_header and print_rows to display a nice table

            Parameters
            ----------
            
            Raises
            ------

            Returns
            -------

            Dependencies
            ------------
                

            Comments
            --------
        """
        Table_LuSt.print_header(self)
        print("="*self.tablewidth)
        Table_LuSt.print_rows(self)
        print("-"*self.tablewidth)

    def latex_template(self):
        """
        
            Parameters
            ----------
            
            Raises
            ------

            Returns
            -------

            Dependencies
            ------------
                

            Comments
            --------
        """        
        #TODO: implement
        pass


# header = ["C1", "C2", "C3", "C4"]
# fstring = ["%10s", "%8d", "%6.2f", "%7.1e"]
# fstring2 = 2*[["%10s", "%8d", "%6.3f", "%7.1e"]]
# row1 = ["t11", 246, 2.45, 1E6]
# row2 = ["t21", 1, 26.45, 1.2E-3]
# row3 = ["t31", 4, 2.45, 8.2E-2]
# row4 = ["t41", 8, 82.45, 1.2E-2]
# rows = [row1, row2]

# table = Table_LuSt(rows=rows, formatstr=fstring2)
# # table = Table_LuSt()
# table.header = header
# table.add_row(row3, fstring=fstring)
# table.add_row(row4)
# print(table)
# table.print_table()
# table.print_header()
# table.print_rows()

# print(table.__dict__)


#%%Plotting
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

  
    
#%%Data Aanlysis
#______________________________________________________________________________
#Class containing useful stuff for data analysis
#TODO: get linspace_def(), pdm(), periodic_shift(), phase2time(), fold(),
#      sigma_clipping(), phase_binning(), lc_error() inside this class
class Data_LuSt:

    def __init__(self):
        pass

#______________________________________________________________________________
#function to define linspace with variable resolution
def linspace_def(centers, widths=None, linspace_range=[0,1] ,
                 nintervals=50, nbins=100, spreads=None, maxiter=100000,
                 go_exact=True, testplot=False, verbose=False, timeit=False):
    """
    Function to generate a linspace in the range of linspace_range with higher 
        resolved areas around given centers.
    The final array has ideally nbins entries, but mostly it is more than that.
    The user can however get to the exact solution by seting go_exact = True.
    The resolution follows a gaussian for each given center.
    The size of these high-res-areas is defined by widths, wich is equal to
        the standard deviation in the gaussian.
    The spilling over the border of the chosen high-res areas can be varied by
        changing the parameter spreads.

    Parameters
    ----------
    centers : np.array/list
        Defines the centers of the areas that will be higher resolved
    widths : np.array/list, optional
        Defines the widths (in terms of standard deviation for a gaussian).
        Those widths define the interval which will be higher resolved
        The default is None, which will result in all ones.
    linspace_range : np.array/list, optional
        Range on which the linspace will be defined.
        The default is [0,1].
    nintervals : int, optional
        Number of intervals to use for computing the distribution of nbins.
        i.e. for 50 intervals all nbins get distributed over 50 intervals-
        The default is 50.
    nbins : int, optional
        Total number of bins that are wished in the final array.
        Due to rounding and the use of a distribution the exact number
            is very unlikely to be reached.
        Because of this the number of bins is in general lower than requested.
        Just play until the best result is reached ;)
        The default is 100.
    spreads : np.array/list, optional
        Parameter to control the amount of overflowing of the distribution over
            the given high-resolution parts.
        The default is None, which will result in no additional overflow.
    maxiter : int, optional
        Parameter to define the maximum number of iterations to take to get as
            close to the desired nbins as possible.
        The default is 100000
    go_exact : bool, optional
        If True random points will be cut from the final result to achive 
            exactly the amount of requested nbins
        The default is True
    testplot : bool, optional
        If True will produce a test-plot that shows the underlying distribution
            as well as the resulting array.
        The default is False.
    verbose : bool, optional
        If True will show messages defined by the creator of the function.
        Eg.: Length of the output, number of iterations.
        The default is False
    timeit : bool, optional
        Specify wether to time the task ad return the information or not.
        The default is False
        

    Raises
    ------
    ValueError
        DESCRIPTION.

    Returns
    -------
    combined_linspace : np.array
        A linspace that has higher resolved areas where the user requests them.
        Those areas are defined by centers and widths, respectively.

    Comments
    --------
        If you don't get the right length of your array right away, vary nbins
            and nintervals until you get close enough ;)

    Dependencies
    ------------
    numpy
    matplotlib.pyplot
    scipy.stats

    """
    
    import numpy as np
    import matplotlib.pyplot as plt
    import scipy.stats as sps

    #time execution
    if timeit:
        task = Time_stuff("linspace_def")
        task.start_task()


    ##########################################
    #initialize not provided data accordingly#
    ##########################################
    
    if widths ==  None:
        widths = np.ones_like(centers)
    if spreads == None:
        #initialize all spreads with "1"
        spreads = np.ones_like(centers)
    


    #################################
    #check if all shapes are correct#
    #################################
    
    if nintervals > nbins:
        raise ValueError("nintervals has to be SMALLER than nbins!")
    
    shape_check1_name = ["centers", "widths", "spreads"]
    shape_check1 = [centers, widths, spreads]
    for sciidx, sci in enumerate(shape_check1):
        for scjidx, scj in enumerate(shape_check1):
            var1 = shape_check1_name[sciidx]
            var2 = shape_check1_name[scjidx]
            if len(sci) != len(scj):
                raise ValueError("Shape of %s has to be shape of %s"%(var1, var2))

    ####################################
    #check if all datatypes are correct#
    ####################################
    
    if type(linspace_range) != np.ndarray and type(linspace_range) != list:
        raise ValueError("input_array has to be of type np.array or list!")
    if type(centers) != np.ndarray and type(centers) != list:
        raise ValueError("centers has to be of type np.array or list!")
    if type(widths) != np.ndarray and type(widths) != list:
        raise ValueError("widths has to be of type np.array or list!")        
    if type(spreads) != np.ndarray and type(spreads) != list:
        raise ValueError("spreads has to be of type np.array or list!")
    if type(nintervals) != int:
        raise ValueError("nintervals has to be of type int!")
    if type(nbins) != int:
        raise ValueError("nbins has to be of type int!")
    if type(maxiter) != int:
        raise ValueError("maxiter has to be of type int!")
    if type(go_exact) != bool:
        raise ValueError("go_exact has to be of type bool!")
    if type(testplot) != bool:
        raise ValueError("testplot has to be of type bool!")
    if type(verbose) != bool:
        raise ValueError("verbose has to be of type bool!")
    if type(timeit) != bool:
        raise ValueError("timeit has to be of type bool!")
    
    #initial definitions and conversions
    centers   = np.array(centers)
    widths    = np.array(widths)
    linspace_range = np.array(linspace_range)
    intervals = np.linspace(linspace_range.min(), linspace_range.max(), nintervals+1)
    
    #generate datapoint distribution
    dist = 0
    for c, w, s in zip(centers, widths, spreads):
        gauss = sps.norm.pdf(intervals, loc=c, scale=s*w)
        gauss /= gauss.max()        #to give all gaussians the same weight
        dist += gauss               #superposition all gaussians to get distribution


    #test various bins to get as close to nbins as possible
    iteration = 0
    testbin = nbins/2
    delta = 1E18
    while delta > 1E-6 and iteration < maxiter:
                 
        testdist = dist/np.sum(dist)            #normalize so all values add up to 1
        testdist *= testbin
        # dist *= nbins                   #rescale to add up to the number of bins
    
        
        #create combined linspace
        combined_linspace = np.array([])
        for i1, i2, bins in zip(intervals[:-1], intervals[1:], testdist):
            ls = np.linspace(i1,i2,int(bins), endpoint=False)                       #don't include endpoint to ensure non-overlapping
            combined_linspace = np.append(combined_linspace, ls)
        
        #add border points
        combined_linspace = np.insert(combined_linspace,0,linspace_range.min())
        combined_linspace = np.append(combined_linspace,linspace_range.max())
    
        delta = (nbins-combined_linspace.shape[0])
        testbin += 1
        iteration += 1
        
    #make sure to get sorted array
    combined_linspace = np.sort(combined_linspace)
    
    #cut random points but not the borders to get exactly to requested nbins
    if go_exact:
        n_to_cut = combined_linspace.shape[0] - nbins
        remove = np.random.randint(1, combined_linspace.shape[0]-1, size = n_to_cut)
        combined_linspace = np.delete(combined_linspace, remove)
    
    if verbose:
        print("Number of iterations: %s"%(iteration))
        print("Shape of combined_linspace: %s"%combined_linspace.shape)
        print("Desired shape: %s"%nbins)
        print("Range of linspace: [%g, %g]"%(combined_linspace.min(), combined_linspace.max()))
        if go_exact:
            print("Number of cut datapoints: %s"%(n_to_cut))
    
    if testplot:
        y_test = np.ones_like(combined_linspace)
        
        fig = plt.figure()
        plt.suptitle("Testplot to visualize generated linspace", fontsize=18)
        plt.plot(combined_linspace, y_test, color="k", marker=".", alpha=0.5, linestyle="", zorder=4)
        plt.scatter(intervals, dist, color="gainsboro", zorder=1, marker=".", figure=fig)
        plt.vlines(centers, dist.min(), dist.max(), colors="b")
        plt.vlines(centers+widths, dist.min(), dist.max(), colors="r")
        plt.vlines(centers-widths, dist.min(), dist.max(), colors="r")
        plt.xlabel("x", fontsize=16)
        plt.ylabel("number of points", fontsize=16)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.show()
          
    #time execution
    if timeit:
        task.end_task()

    
    return combined_linspace
    

##################################
# TEST FOR linspace_def function #  
##################################

# import numpy as np
# centers = [-10, 50]
# widths = [4, 5]
# a = np.linspace(-100,100,1000)
# nintervals=50
# nbins=200
# spreads=[20,3]
# linspace_def(centers=centers, widths=widths, linspace_range=a, nintervals=nintervals, nbins=nbins, spreads=spreads, go_exact=True, testplot=True, verbose=True, timeit=True)


#______________________________________________________________________________
#function to execute a phase dispersion minimization
#TODO: implement
def pdm():
    
    pass


#______________________________________________________________________________
#function to shift an array in a periodic interval
def periodic_shift(input_array, shift, borders, timeit=False, testplot=False, verbose=False):
    """
    Function to shift an array considering periodic boundaries.

    Parameters
    ----------
    input_array : np.array
        array to be shifted along an interval with periodic boundaries.
    shift : float/int
        size of the shift to apply to the array.
    borders : list/np.array
        upper and lower boundary of the periodic interval.
    timeit : bool, optional
        wether to time the execution. The default is False.
    testplot : bool, optional
        wether to show a testplot. The default is False.
    verbose : bool, optional
        wether to output information about the result. The default is False.

    Raises
    ------
    TypeError
        DESCRIPTION.

    Returns
    -------
    shifted : np.arra
        array shifted by shift along the periodic interval in borders.

    """
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    
    #time execution
    if timeit:
        task = Time_stuff("periodic_shift")
        task.start_task()



    ################################
    #check if all types are correct#
    ################################
    
    if type(input_array) != np.ndarray:
        raise TypeError("input_array has to be of type np.ndarray!")
    if (type(shift) != float) and (type(shift) != int):
        raise TypeError("shift has to be of type int or float!")
    if (type(borders) != np.ndarray) and (type(borders) != list):
        raise TypeError("borders has to be of type np.array or list!")
    if (type(timeit) != bool):
        raise TypeError("timeit has to be of type bool")
    if (type(testplot) != bool):
        raise TypeError("testplot has to be of type bool")
    if (type(verbose) != bool):
        raise TypeError("verbose has to be of type bool")


    #############
    #shift array#
    #############
    lower_bound = np.min(borders)
    upper_bound = np.max(borders)

    
    #apply shift
    shifted = input_array+shift
    
    #resproject into interval
    out_of_lower_bounds = (shifted < lower_bound)
    out_of_upper_bounds = (shifted > upper_bound)

    lower_deltas = lower_bound-shifted[out_of_lower_bounds]
    shifted[out_of_lower_bounds] = upper_bound - lower_deltas    
    upper_deltas = shifted[out_of_upper_bounds]-upper_bound
    shifted[out_of_upper_bounds] =lower_bound + upper_deltas
    

    if verbose:
        print("input_array: %a"%input_array)
        print("shifted_array: %a"%shifted)
        print("shift: %g"%shift)
        print("boundaries: %g, %g"%(lower_bound, upper_bound))
    
    if testplot:
        y_test = np.ones_like(shifted)
        
        fig = plt.figure()
        plt.suptitle("Testplot to visualize shift")
        # plt.plot(shifted[out_of_lower_bounds], y_test[out_of_lower_bounds]-1, color="b", marker=".", alpha=1, linestyle="", zorder=4)
        # plt.plot(shifted[out_of_upper_bounds], y_test[out_of_upper_bounds]+1, color="r", marker=".", alpha=1, linestyle="", zorder=4)
        plt.plot(shifted, y_test, color="r", marker="x", alpha=1, linestyle="", zorder=4, label="shifted array")
        plt.plot(input_array, y_test, color="k", marker=".", alpha=1, linestyle="", zorder=4, label="original array")
        plt.vlines([lower_bound, upper_bound], ymin=y_test.min()-1, ymax=y_test.max()+1, color="g", linestyle="--", label="boundaries")
        plt.xlabel("x", fontsize=16)
        plt.ylabel("y", fontsize=16)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.legend()
        plt.show()

    #time execution
    if timeit:
        task.end_task()
    
    return shifted

##################################
#Test for periodic_shift function#
##################################

# import numpy as np
# input_array = np.linspace(0,9,10)
# shift = 0.5
# borders = [0,7]
# shifted = periodic_shift(input_array, shift, borders, timeit=True, testplot=True, verbose=True)


#______________________________________________________________________________
#function to convert a phase array to its respective period
def phase2time(phases, period, timeit=False):
    """
    converts a given array of phases into its respective time equivalent

    Parameters
    ----------
    phases : np.array, float
        The phases to convert to times
    period : float
        The given period the phase describes

    Returns
    -------
    time : np.array, float
        The resulting time array, when the phases are converted
    timeit : bool, optional
        Specify wether to time the task ad return the information or not.
        The default is False
        
        
    """
    
    #time execution
    if timeit:
        task = Time_stuff("phase2time")
        task.start_task()


    time = phases*period
    
    #time execution
    if timeit:
        task.end_task()

    return time


#______________________________________________________________________________
#function to fold a time-array by a specified period
def fold(time, period, timeit=False):
    """
    takes an array of times and folds it with a specified period
    returns folded array of phases    

    Parameters
    ----------
    time : np.array
        times to be folded with the specified period.
    period : int
        Period to fold the times with.
    timeit : bool, optional
        Specify wether to time the task ad return the information or not.

    Returns
    -------
    phases_folded : np.array
        phases corresponding to the given time folded with period.

    """

    import numpy as np
    
    #time execution
    if timeit:
        task = Time_stuff("fold")
        task.start_task()

    
    
    delta_t = time-time.min()
    phases = delta_t/period
    
    #fold phases by getting the remainder of the division by the ones-value 
    #this equals getting the decimal numbers of that specific value
    #+1 because else a division by 0 would occur
    #floor always rounds down a value to the ones (returns everything before decimal point)
    phases_folded = (phases)-np.floor(phases) - 0.5

    #time execution
    if timeit:
        task.end_task()

    
    return phases_folded


#______________________________________________________________________________
#function to execute sigma_clipping
#TODO: check if working correctly (zip(intervals[:-1], intervals[1:])
def sigma_clipping(fluxes, fluxes_mean, phases, phases_mean, intervals, clip_value_top, clip_value_bottom, times=[], timeit=False):
    """
    cuts out all datapoints of fluxes (and phases and times) array which are outside of the interval
        [clip_value_bottom, clip_value_top] and returns the remaining array
    used to get rid of outliers
    clip_value_bottom and clip_value_top are usually defined as n*sigma, with 
        n = 1,2,3,... and sigma the STABW (Variance?)
    if times is not specified, it will return an array of None with same size as fluxes    

    Parameters
    ----------
    fluxes : np.array
        fluxes to be cut.
    fluxes_mean : np.array
        mean values of fluxes to use as reference for clipping.
    phases : np.array
        phases to be cut.
    phases_mean : np.array 
        phases of fluxes_mean.
    intervals : np.array
        DESCRIPTION.
    clip_value_top : np.array
        top border of clipping.
    clip_value_bottom : np.array
        bottom border of clipping.
    times : np.array, list, optional
        times of fluxes (if existent). The default is [].
    timeit : bool, optional
        Specify wether to time the task ad return the information or not.
        The default is False
        

    Returns
    -------
    fluxes_sigcut : Fluxes after cutting out all values above clip_value_top and below clip_value bottom
    phases_sigcut : Phases after cutting out all values above clip_value_top and below clip_value bottom
    times_sigcut : Times after cutting out all values above clip_value_top and below clip_value bottom
    cut_f : All cut-out values of fluxes
    cut_p : All cut-out values of phases
    cut_t : All cut-out values of times

    """

    import numpy as np

    #time execution
    if timeit:
        task = Time_stuff("sigma_clipping")
        task.start_task()

    
    times = np.array(times)
    #initiate saving arrays
 
    if len(times) == 0:
        times = np.array([None]*len(fluxes))
    elif len(times) != len(fluxes):
        raise ValueError("fluxes and times have to be of same legth!")
    elif len(fluxes) != len(phases):
        raise ValueError("fluxes and phases have to be of same length!")

    times_sigcut = np.array([])    
    fluxes_sigcut = np.array([])
    phases_sigcut = np.array([])
    cut_p = np.array([])
    cut_f = np.array([])
    cut_t = np.array([])

    intervalsp = np.roll(intervals,1)    

    for iv, ivp in zip(intervals[1:], intervalsp[1:]):
        
        bool_iv = ((phases <= iv) & (phases > ivp))
        bool_mean = ((phases_mean <= iv) & (phases_mean > ivp))
        upper_flux = fluxes_mean[bool_mean] + clip_value_top[bool_mean]
        lower_flux = fluxes_mean[bool_mean] - clip_value_bottom[bool_mean]
        # print(len(upper_flux), len(lower_flux))
        
        fluxes_iv = fluxes[bool_iv]
        phases_iv = phases[bool_iv]
        times_iv  = times[bool_iv]
        
        bool_fluxcut = ((fluxes_iv < upper_flux) & (fluxes_iv > lower_flux))
        fluxes_cut = fluxes_iv[bool_fluxcut]
        phases_cut = phases_iv[bool_fluxcut]
        times_cut  = times_iv[bool_fluxcut]
        cut_fluxes = fluxes_iv[~(bool_fluxcut)] 
        cut_phases = phases_iv[~(bool_fluxcut)]  
        cut_times  = times_iv[~(bool_fluxcut)]
                 
        fluxes_sigcut = np.append(fluxes_sigcut, fluxes_cut)
        phases_sigcut = np.append(phases_sigcut, phases_cut)
        times_sigcut  = np.append(times_sigcut, times_cut)
        cut_f = np.append(cut_f, cut_fluxes)
        cut_p = np.append(cut_p, cut_phases)
        cut_t = np.append(cut_t, cut_times)
#    print(len(fluxes_sigcut), len(cut_f), len(fluxes))
    
    #time execution
    if timeit:
        task.end_task()

                    
    return fluxes_sigcut, phases_sigcut, times_sigcut, cut_f, cut_p, cut_t


#______________________________________________________________________________
#Function to execute binning in phase
#TODO: check if working correctly (zip(intervals[:-1], intervals[1:])
def phase_binning(fluxes, phases, nintervals, nbins, centers, widths, spreads=None, verbose=False, testplot=False, timeit=False):
    """
    Function to execute binning in phase on some given timeseries.
    Additional parameters allow the user to define different resolutions in 
        in different areas of the timeseries.
    returns the mean flux, phase and variance of each interval.

    Parameters
    ----------
    fluxes : np.array
        fluxes to be binned.
    phases : np.array
        phases to be binned.
    nintervals : int
        number of intervals to distribute nbins on.
    nbins : int
        number of bins you wish to have in the final array.
        is most of the time not fullfilled.
        play with widths and spreads to adjust the number.
    centers : np.array/list
        centers of areas which one wants a higher resolution on.
        will become the centers of a gaussian for this area representing the
            distribution of nbins
    widths : np.array/list
        widths of the areas which one wants a higher resolution on.
        will become the standard deviation in the gaussian for this area
            representing the distribution of nbins
    spreads : np.array/list, optional
        used to define the spill over the boundaries of centers +/- widths.
        The default is None, which will result in non spill of the gaussian
    verbose : bool, optional
        wether to show messages integrated in the function or not
        The default is False.
    testplot : bool, optinoal
        wether to show a testplot of the chosen distribution for the higher resolved area
        The default is False
    timeit : bool, optional
        Specify wether to time the task ad return the information or not.
        The default is False


    Returns
    -------
    phases_mean : TYPE
        DESCRIPTION.
    fluxes_mean : TYPE
        DESCRIPTION.
    fluxes_sigm : TYPE
        DESCRIPTION.
    intervals : TYPE
        DESCRIPTION.

    """
    

    import numpy as np

    
    #time execution
    if timeit:
        task = Time_stuff("phase_binning")
        task.start_task()



    intervals = linspace_def(centers,
                             widths,
                             linspace_range=[phases.min(),phases.max()],
                             nintervals=nintervals,
                             nbins=nbins,
                             spreads=spreads,
                             testplot=testplot
                             )
    
    #saving arrays for mean LC    
    phases_mean = np.array([])    
    fluxes_mean = np.array([])
    fluxes_sigm = np.array([])
    
    #calculate mean flux for each interval
    for iv1, iv2 in zip(intervals[1:], intervals[:-1]):  
        bool_iv = ((phases <= iv1) & (phases > iv2))
        
        #calc mean phase, mean flux and standard deviation of flux for each interval        for pidx, p in enumerate(phases):
        mean_phase = np.mean(phases[bool_iv])
        mean_flux  = np.mean(fluxes[bool_iv])
        sigm_flux  = np.std(fluxes[bool_iv])

        phases_mean = np.append(phases_mean, mean_phase)
        fluxes_mean = np.append(fluxes_mean, mean_flux)
        fluxes_sigm = np.append(fluxes_sigm, sigm_flux)    
     
    if verbose:
        print("\n"+20*"-")
        print("Phase_Binning:")
        print("shape of binned phases     : %s"%phases_mean.shape)
        print("shape of binned fluxes     : %s"%fluxes_mean.shape)
        print("shape of binned flux errors: %s"%fluxes_sigm.shape)
        print("shape of intervals used    : %s"%intervals.shape)
        

    #time execution
    if timeit:
        task.end_task()

    
    return phases_mean, fluxes_mean, fluxes_sigm, intervals


#______________________________________________________________________________
#function to estimate error of LC

def lc_error(fluxes, times, delta_t_points, timeit=False):
    """    
    estimates the error of a lightcurve given the respective time series and a
        time-difference condition (delta_t_points - maximum difference between 2 data points
        of same cluster).
    The LC will be divided into clusters of Datapoints, which are nearer to their
        respective neighbours, than the delta_t_points.
    returns an array, which assigns errors to the LC. Those errors are the standard
        deviation of the respective intervals. All values of the LC in the same
        interval get the same errors assigned.
    returns mean values of the intervals and the belonging standard deviations
    
    Parameters
    ----------
    fluxes : np.array
        fluxes to extimate the error of.
    times : np.array
        times corresponding to fluxes.
    delta_t_points : float
        time-difference condition (delta_t_points - maximum difference between 2 data points
        of same cluster)
    timeit : bool, optional
        Specify wether to time the task ad return the information or not.
        The default is False
        

    Returns
    -------
    LC_errors : estimated errors of flux
    means : mean values of flux
    stabws : standard deviations of flux

    """
    
    import numpy as np

    #time execution
    if timeit:
        task = Time_stuff("lc_error")
        task.start_task()

    
    
    times_shifted = np.roll(times, 1)  #shifts array by one entry to the left, last element gets reintroduced at idx=0
    times_bool = times-times_shifted > delta_t_points 
    cond_fulfilled = np.where(times_bool)[0]    #returns an array of the indizes of the times, which fulfill condition in times_bool
    cond_fulfilled_shift = np.roll(cond_fulfilled,1)

    means  = np.array([]) #array to save mean values of intervals to
    stabws = np.array([]) #array to save stabws of intevals to
    LC_errors = np.empty_like(times)        #assign same errors to whole interval of LC
    
    for idxf, idxfs in zip(cond_fulfilled, cond_fulfilled_shift):
        
        if idxf == cond_fulfilled[0]:
            mean_iv  = np.mean(fluxes[0:idxf])
            stabw_iv = np.std(fluxes[0:idxf])
            LC_errors[0:idxf] = stabw_iv        #assign same errors to whole interval of LC         

        elif idxf == cond_fulfilled[-1]:
            mean_iv  = np.mean(fluxes[idxf:])
            stabw_iv = np.std(fluxes[idxf:])
            LC_errors[idxf:] = stabw_iv
        else:
            mean_iv  = np.mean(fluxes[idxfs:idxf])
            stabw_iv = np.std(fluxes[idxfs:idxf])
            LC_errors[idxfs:idxf] = stabw_iv    #assign same errors to whole interval of LC         
        
        means  = np.append(means,  mean_iv)
        stabws = np.append(stabws, stabw_iv) 
            
#        print("idxf: %i, idxfs: %i" %(idxf, idxfs))        

#    print(len(cond_fulfilled), len(means), len(stabws))
#    plt.figure()
#    plt.errorbar(cond_fulfilled, means, yerr=stabws, linestyle="", marker=".", color="r")
#    plt.show()
        
    #time execution
    if timeit:
        task.end_task()
    
    return LC_errors, means, stabws


#%%PHOEBE
#______________________________________________________________________________
#Class useful for working with PHOEBE
#TODO: get extract_data(), plot_data_PHOEBE inside this class
class PHOEBE_additional:

    def __init__(self):
        pass

#______________________________________________________________________________
#function to extract data from phoebe b. object
def extract_data(phoebes_b, distance, unit_distance, show_keys=False, timeit=False):
    """
    Function to extract data from PHOEBE bundle

    Parameters
    ----------
    phoebes_b : PHOEBEs default binary object
        data structure object of PHOEBE to be converted to dict.
    distance : float
        distance to observed object. If not known simply type 1
    unit_distance : str
        a string to specify the unit of distance.
    show_keys : bool, optional
        If true prints out all possible keys (twigs) of the created dictionary. The default is False.
    timeit : bool, optional
        Specify wether to time the task ad return the information or not.
        The default is False
        

    Returns
    -------
    dictionary of PHOEBEs default binary object with all the data-arrays in it.
    The structure is the following:
    binary = {
            "lc-dataset1":{
                "enabled":True
                "model1":{"qualifier1":np.array([values]), "qualifier2":np.array([...]), ... ,"qualifieri":np.array([...])},
                "model2":{...},
                :
                :
                :
                "modeli":{...}
                }, 
            "lc-dataset2":{"enabled":True, qualifier1:np.array([values]),...},
            :
            :
            :
            "lc-dataseti":{...},
            "orb-daraset1:{
                "model1":{
                    "component1":{"qualifier1":np.array([values]), "qualifier2":np.array([...]), ... ,"qualifieri":np.array([...])}
                    "component2":{"qualifier1":np.array([values]), "qualifier2":np.array([...]), ... ,"qualifieri":np.array([...])}
                    :
                    :
                    "componenti":{"qualifier1":np.array([values]), "qualifier2":np.array([...]), ... ,"qualifieri":np.array([...])}
                    },
                "model2":
                :
                :
                "modeli":
                }
        }
    
    Notes
    -----
    to add new dataset: add another if d[...:...] == "newdataset", and extract the date with phoebes twig-system
        

    """
    
    import numpy as np
    
    #time execution
    if timeit:
        task = Time_stuff("extract_data")
        task.start_task()
    
    
    #convert given distance to Solar Radii, to have correct units            
    if distance == 0:
        raise ValueError("distance can not be 0!!! Choose a estimate value for the distance to the object!!\nValue has te be entered in meters!!")

    
    R_Sun_m = 696342*10**3 #m
    R_Sun_Lj = 7.36*10**-8 #Lj
    R_Sun_pc = 2.25767*10**-8 #pc


    if unit_distance == "m":
        distance /= R_Sun_m
    elif unit_distance == "Lj":
        distance /= R_Sun_Lj
    elif unit_distance == "pc":
        distance /= R_Sun_pc
    elif unit_distance == "Sol_Rad":
        distance /= 1
    else:
        print("unit of distance has to be a string of: m, Lj, pc, Sol_Rad!!")
        return

    
    
    b = phoebes_b
    binary = {}     #create dictionary to store values according to phoebe twigs
    
    #extracts the given data of specified datasets
    #to add a new dataset add another "if d[... : ...] == "newdataset" ", and extract the date with phoebes twig-system
    for d in b.datasets:
        binary[d] = {}
        binary[d]["enabled"] = b.get_value(dataset=d, qualifier="enabled")
        binary[d]["binary"] = {}
#        print("______________")
#        print("\ndataset: %s"%(d))
        
        #only extract data saved under componet=binary, if enabled == True
        if binary[d]["enabled"]:
            for q in b[d]["binary"].qualifiers:
                binary[d]["binary"][q] = b.get_value(dataset=d, component="binary", qualifier=q)
#                print("dataset: %s, component: binary, qualifier: %s" %(d, q))
        
        for m in b[d].models:
            binary[d][m] = {}
#            print("dataset: %s, model: %s"%(d, m))
            
            #extract all available orbit datasets of the created models
            if d[0:3] == "orb":
                for c in b[d][m].components:
                    if c != "binary":
                        binary[d][m][c] = {}
                        binary[d][m][c]["phases"] = b.to_phase(b.get_value(dataset=d, model=m, component=c, qualifier="times"))
#                        print("dataset: %s, model: %s, component: %s"%(d,m,c))

#TODO: NOT QUITE SURE IF CONVERSION to [''] IS CORRECT
                        us = b.get_value(dataset=d, model=m, component=c, qualifier="us")   #xposition
                        vs = b.get_value(dataset=d, model=m, component=c, qualifier="vs")   #yposition
                        ws = b.get_value(dataset=d, model=m, component=c, qualifier="ws")   #zposition                        
                        theta_arcsec = np.arctan(us/distance)*(180/np.pi)*3600     #displacement in x-direction - arctan returns radians
                        phi_arcsec   = np.arctan(vs/distance)*(180/np.pi)*3600     #displacement in y-direction - arctan returns radians
                        alpha_arcsec = np.arctan(ws/distance)*(180/np.pi)*3600     #displacement in z-direction - arctan returns radians
                        binary[d][m][c]["theta"] = theta_arcsec
                        binary[d][m][c]["phi"] = phi_arcsec
                        binary[d][m][c]["alpha"] = alpha_arcsec
                        
                        for q in b[d][m][c].qualifiers:
                            binary[d][m][c][q] = b.get_value(dataset=d, model=m, component=c, qualifier=q) 
#                            print("dataset: %s, model: %s, component: %s, qualifier: %s"%(d,m,c,q))
            
            #extract all available lightcurve datasets of the created models
            if d[0:2] == "lc":
                binary[d][m]["phases"] = b.to_phase(b.get_value(dataset=d, model=m, qualifier="times"))
                for q in b[d][m].qualifiers:
                    binary[d][m][q] = b.get_value(dataset=d, model=m, qualifier=q) 
#                    print("dataset: %s, model: %s, qualifier: %s"%(d,m,q))
            
            if d[0:2] == "rv":
                for c in b[d][m].components:
                    binary[d][m][c] = {}
                    binary[d][m][c]["phases"] = b.to_phase(b.get_value(dataset=d, model=m, component=c, qualifier="times"))
#                    print("dataset: %s, model: %s, component: %s"%(d,m,c))

                    for q in b[d][m][c].qualifiers:
                        binary[d][m][c][q] = b.get_value(dataset=d, model=m, component=c, qualifier=q) 
#                        print("dataset: %s, model: %s, qualifier: %s"%(d,m,q))

                    
#    print(binary)
    
    #shows all available twigs, if set to true
    if show_keys:
        print("ALLOWED Keys/twigs")
        for k1 in binary.keys():
            print("\n_______twig %s__________\n" % (k1))
            for k2 in binary[k1].keys():
                if type(binary[k1][k2]) == bool:
                    print("binary[\"%s\"][\"%s\"]"%(k1,k2))
                
                elif k2 == "binary":
                    for k3 in binary[k1][k2].keys():
                        print("binary[\"%s\"][\"%s\"][\"%s\"]"%(k1,k2,k3))              
               
                else:
                    for k3 in binary[k1][k2].keys():
                        if type(binary[k1][k2][k3]) == np.ndarray:                    
                            print("binary[\"%s\"][\"%s\"][\"%s\"]"%(k1,k2,k3))
                        else:
                            for k4 in binary[k1][k2][k3].keys():
                                print("binary[\"%s\"][\"%s\"][\"%s\"][\"%s\"]"%(k1,k2,k3,k4))

    #time execution
    if timeit:
        task.end_task()   
        
    return binary

#______________________________________________________________________________
#function plot extracted data (data in binary-object)
#TODO: Add a variable to an array of color-names. Those will be used in order to plot each run    
def plot_data_PHOEBE(binary, lc_phase_time_choice="phases", rv_phase_time_choice="phases", orb_solrad_arcsec_choice = "arcsec", save_plot=False, save_path="", plot_lc=True, plot_orb=True, plot_rv=True, timeit=False):
    """
    creates plots of the extracted values
    to not show a plot, set plot_ to False
    to save plots to a defined location: set save_plot to True and define a save_path (string)
        the filename will be generated automatically
        if save_path is not defined, the current working directory will be used
        if not the whole path is given, the path will start from the current working directory
        save_path has to have a / at the end, otherwise it will be part of the name

    Parameters
    ----------
    binary : dict
        Object extracted from PHOEBE with extract_data function.
    lc_phase_time_choice : str, optional
        Used to specify wether to use time or phase for lc plotting. The default is "phases".
    rv_phase_time_choice : TYPE, optional
        Used to specify wether to use time or phase for rv plotting. The default is "phases".
    orb_solrad_arcsec_choice : TYPE, optional
        Used to specify wether to use solar radii or arcsec for orbit plotting. The default is "arcsec".
    save_plot : bool, optional
        If True, saves the plot to the specified path. The default is False.
    save_path : str, optional
        Used to specify the savepath of the created plots. The default is "".
    plot_lc : bool, optional
        If True, creates a plot of the LC. The default is True.
    plot_orb : TYPE, optional
        If True, creates a plot of the Orbits. The default is True.
    plot_rv : TYPE, optional
        If True, creates a plot of the RV. The default is True.
    timeit : bool, optional
        Specify wether to time the task ad return the information or not.
        The default is False
        
    Returns
    -------
    None.

    """
    

    import matplotlib.pyplot as plt
    import os
    import re
    
    #time execution
    if timeit:
        task = Time_stuff("plot_ax")
        task.start_task()
    
    #check if safe_path has the needed structure
    if (not save_path[-1] == "/") or (save_path == ""):
        print("ERROR: save_path has to have an \"/\" at the end!")
        return
    
    if not os.path.isdir(save_path):   #create new saveingdirectory for plots, if it does not exist
        os.makedirs(save_path)


    print("Started creating Plots")
    
        
    for nrd, d in enumerate(binary): 
        if not plot_lc:  binary[d]["enabled"] = False
        if not plot_orb: binary[d]["enabled"] = False
        if not plot_rv:  binary[d]["enabled"] = False
        
        #return message, if no plots are crated
        if (not plot_lc) and (not plot_orb) and (not plot_rv):
            print("No plots created.\nAll values binary[\"dataset\"][\"enabled\"] == False for all datasets")
            return
        
        if binary[d]["enabled"]:
            for nrm, m in enumerate(binary[d]):
                if m != "enabled" and m != "binary":
                    
                    #plot lightcurves
                    if d[0:2] == "lc":
                        
                        passband = re.findall("(?<=_).+", d)[0] #get which passband filter was used
                        
                        #figure
                        fignamelc = "lc%s"%(1000+10*nrd+nrm)
                        figlc = plt.figure(num=fignamelc, figsize=(10,10))
                        figlc.suptitle("lightcurve \nModel: %s, Passband: %s"%(m,passband), fontsize=18)
                        plt.xticks(fontsize=16); plt.yticks(fontsize=16)
                        plt.ylabel(r"flux $ \left[ \frac{W}{m^2} \right]$", fontsize=16)
                        
                        print("\nplotting %s(%s) of binary[%s][%s] in figure %s" % ("flux", lc_phase_time_choice, d,m, fignamelc))

                        #plot                            
                        fluxes = binary[d][m]["fluxes"]
                        #check wether to plot over phases or times                        
                        if lc_phase_time_choice == "times":
                            times = binary[d][m]["times"]                            
                            plt.xlabel(r"time [d]", fontsize=16)
                            plt.plot(times, fluxes, "o", figure=figlc, label=m+"_"+passband)
                        elif lc_phase_time_choice == "phases":
                            phases = binary[d][m]["phases"]                            
                            plt.xlabel(r"phases", fontsize=16)
                            plt.plot(phases, fluxes, "o", figure=figlc, label=m+"_"+passband)
                        else:
                            print("Neither keyword is correct\ Keyword has to be either \"times\" or \"phases\". Default is \"phases\"!")
                            return

                        plt.legend()
                    
                    #save
                    if save_plot:
                        filename = "plot_"+d+"_"+m+".png"
                        path = save_path + filename
                        plt.savefig(path)
                        print("plots saved to %s" %(path))
                        
                    #plot orbits    
                    if d[0:3] == "orb":
                        
                        #figure
                        fignameorb = "orb%s"%(2000+nrm) #create figure here, to show orbits of both components in one graph
                        figorb = plt.figure(num=fignameorb, figsize=(10,10))
                        figorb.suptitle("orbits \nModel:%s" %(m), fontsize=18)
                        plt.xticks(fontsize=16); plt.yticks(fontsize=16)

                        for nrc, c in enumerate(binary[d][m]):                            
                            print("\nplotting %s(%s) of binary[%s][%s][%s] in figure %s" % ("vs", "us", d,m,c, fignameorb))
                            
                            #plot
                            if orb_solrad_arcsec_choice == "solrad":
                                vs = binary[d][m][c]["vs"]
                                us = binary[d][m][c]["us"]
                                plt.xlabel(r"us (xposition) $ \left[R_{\odot} \right]$", fontsize=16)
                                plt.ylabel(r"vs (yposition) $ \left[R_{\odot} \right]$", fontsize=16)                                
                                plt.plot(us, vs, figure = figorb, label=c)
                            elif orb_solrad_arcsec_choice == "arcsec":
                                theta = binary[d][m][c]["theta"]
                                phi = binary[d][m][c]["phi"]
                                plt.xlabel(r"theta $ \left[ '' \right]$", fontsize=16)
                                plt.ylabel(r"phi $ \left[ '' \right]$", fontsize=16)                                
                                plt.plot(theta, phi, figure = figorb, label=c)
                            else:
                                print("Neither keyword is correct\ Keyword has to be either \"solrad\" or \"arcsec\". Default is \"arcsec\"!")
                            
                            plt.legend()
                            

                        #save
                        if save_plot:
                            filename = "plot_"+d+"_"+m+".png"
                            path = save_path + filename
                            plt.savefig(path)
                            print("plots saved to %s" %(path))
                    
                    #plot radial velocities        
                    if d[0:2] == "rv":
                        
                        #figure
                        fignamerv = "rv%s"%(3000+nrm) #create figure here, to show radial velocities of both components in one graph
                        figrv = plt.figure(num=fignamerv, figsize=(10,10))
                        figrv.suptitle("radial velocities \nModel:%s" %(m), fontsize=18)
                        plt.xticks(fontsize=16); plt.yticks(fontsize=16)
                        plt.ylabel(r"Radial Velocity $\left[ \frac{km}{s} \right]$", fontsize=16)

                        for nrc, c in enumerate(binary[d][m]):
                            
                            #plot
                            rvs = binary[d][m][c]["rvs"]
                            if rv_phase_time_choice == "times":
                                print("\nplotting %s(%s) of binary[%s][%s][%s] in figure %s" % ("rvs", "time", d,m,c, fignamerv))
                                times = binary[d][m][c]["times"]
                                plt.xlabel(r"time [d]", fontsize=16)
                                plt.plot(times, rvs, marker="o", linestyle="", figure=figrv, label=c)
 
#!!!!!to do: phase does not work propely!!!!!
                            elif rv_phase_time_choice == "phases":
                                print("\nplotting %s(%s) of binary[%s][%s][%s] in figure %s" % ("rvs", "phase", d,m,c, fignamerv))
                                phases = binary[d][m][c]["phases"]
                                plt.xlabel(r"phases", fontsize=16)
                                plt.plot(phases, rvs, marker="o", linestyle="", figure=figrv, label=c)
                             
                            else:
                                print("Neither keyword is correct\ Keyword has to be either \"times\" or \"phases\". Default is \"phases\"!")
                                return
                            
                            plt.legend()

                        #save
                        if save_plot:
                            filename = "plot_"+d+"_"+m+".png"
                            path = save_path + filename
                            plt.savefig(path)
                            print("plots saved to %s" %(path))

    #time execution
    if timeit:
        task.end_task()   

    
    plt.show()
    return



