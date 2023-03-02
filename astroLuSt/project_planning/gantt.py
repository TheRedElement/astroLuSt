
#______________________________________________________________________________
#Class for creating a GANTT-chart variation (for Mission Planning)
class GANTT:
    """
        - class to genereate a variation of a GANTT-chart
        
        Attributes
        ----------
            - time
                - array of the period of time where the project takes place
                - time has to be an array of datetime-objects!
            - starts
                - array of datetime-objects
                - contains the starting point relative to time.min()
                - i.e. the first task starts usually at start = 0
            - ends
                - array of datetime-objects
                - contains the starting point relative to time.min()
            - tasks
                - relative workload for the tasks to be done (dependent on time)
            - tasknames
                - nametag for each task
            - weights
                - array of weights weighting the importance of each task
            - percent_complete
                - array of percentages
                - defines how much of a specific task is completed

        Methods
        -------
            - sigmoid
                - returns the sigmoid of some input
            - task_func
                - fucntion that defines one task
                - is a combination of two sigmoids with opposite signs before 'x'
            - task
                - function to add a task to the project
            - make_graph
                - function to visualize all tasks combined
                    - includes a GANTT-variation
                    - includes a GANTT-graph
                - also returns the total workload for every timepoint
            - make_classic_gantt
                - function to create a Gantt-Graph in classical style

        Dependencies
        ------------
            - matplotlib
            - numpy
            - datetime

        Comments
        --------
            - 'whole_area' (argument of 'self.task', will be adjusted for each graph separately)
                - make sure you set 'whole_area' to the value you need, when passing a task to the class

    """

    def __init__(self, time):

        import numpy as np
        from datetime import datetime
        
        if any([type(t) != datetime for t in time]):
            raise TypeError("'time' has to be an array containing 'datetime' objects!")
        self.time = time
        self.starts = np.array([])
        self.ends = np.array([])
        self.tasks = np.array([])
        self.tasknames = []
        self.weights = np.ones(len(self.tasks))

        self.percent_complete = np.array([])
        self.percent_idxi = np.array([], dtype=int)

    def sigmoid(self, time, slope, shift):
        """
            - calculates the sigmoid function
        """
        import numpy as np
        Q1 = 1 + np.e**(slope*(-(time - shift)))
        return 1/Q1

    def task_func(self, time, start, end, start_slope, end_slope):
        """
            - function to calculate a specific tasks workload curve
            - combines two sigmoids with opposite signs before the exponent to do that
        
            Parameters
            ----------
                - time
                    - np.array
                    - the times (relative to the starting time) used for the tasks
                        - i.e. an array starting with 0
                - start
                    - float
                    - time at which the tasks starts
                    - relative to the zero point in time (i.e. time.min() = 0)
                - end
                    - float
                    - time at which the task ends
                    - relative to the zero point in time (i.e. time.min() = 0)
                - start_slope
                    - float
                    - how steep the starting phase should be
                - end_slope
                    - float
                    - how steep the ending phase (reflection phase) should be

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
        start_phase = self.sigmoid(time, start_slope, start)
        end_phase   = self.sigmoid(-time, end_slope, -end)
        task = start_phase + end_phase
        task -= task.min()
        task /= task.max()
        return task

    def task_integral(self, time, start, end, start_slope, end_slope, percent_complete, whole_area=True):
        """
            - function to estimate the index corresponding to the 'percent_completed'
                - To do this:
                    - calculates the intergal of the function used to define a task ('task_func()')
                    - i.e. 2 sigmoids with opposite signs befor the exponent
                - algebraic solution (latex formatting):

                    >>> \\begin{align}
                    >>>     \\frac{\ln(e^{s_1x} + e^{a_1s_1})}{s_1}
                    >>>         - \\frac{\ln(e^{s_2x} + e^{a_2s_2})}{s_2}
                    >>>         - \mathcal{C}
                    >>> \\end{align}

            Parameters
            ----------
                - time
                    - np.array
                    - the times (relative to the starting time) used for the tasks
                        - i.e. an array starting with 0
                - start
                    - float
                    - time at which the tasks starts
                    - relative to the zero point in time (i.e. time.min() = 0)
                - end
                    - float
                    - time at which the task ends
                    - relative to the zero point in time (i.e. time.min() = 0)
                - start_slope
                    - float
                    - how steep the starting phase should be
                - end_slope
                    - float
                    - how steep the ending phase (reflection phase) should be
                - percent_complete
                    - float, optional
                    - percentage describing how much of the task is completed
                    - number between 0 and 1
                    - the default is 0
                - whole_area
                    - bool, optional
                    - whether to consider the whole area beneath the curves as 100% of the task
                        - this will especially in low percentages NOT line up with the actual GANTT-chart (second subplot) 
                    - otherwise will consider the area between 'start' and 'end' as 100% of the task
                        - will be exactly aligned with the GANTT-chart
                        - but it might be that 0% already has some area colored in
                    - the default is True
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
        def ln1(bound):
            return np.log(np.e**(start_slope*bound) + np.e**(start_slope*start))
        def ln2(bound):
            return np.log(np.e**(end_slope*bound)   + np.e**(end_slope*end))

        #integral over whole interval
        if whole_area:
            #upper bound
            T1 = ln1(time.max())
            T2 = ln2(time.max())
            #upper bound
            T3 = ln1(time.min())
            T4 = ln2(time.min())
        else:
            #upper bound
            T1 = ln1(end)
            T2 = ln2(end)
            #lower bound
            T3 = ln1(start)
            T4 = ln2(start)
        int_whole_interval = T1/start_slope - T2/end_slope - (T3/start_slope - T4/end_slope)
        
        #ingetrals for all time-points
        T1_i = ln1(time)
        T2_i = ln2(time)
        int_times = T1_i/start_slope - T2_i/end_slope - (T3/start_slope - T4/end_slope)

        #get index of time that is the closest to percent_complete from the total time interval
        percent_idx = np.argmin(np.abs(int_times - int_whole_interval*percent_complete))

        return percent_idx

    def task(self, start, end, start_slope, end_slope, taskname=None, weight=1, percent_complete=0, whole_area=True, testplot=False):
        """
            - function to define a specific task
            - will add that task to 'self.tasks' as well

            Parameters
            ----------
                - start
                    - float
                    - time at which the tasks starts
                    - relative to the zero point in time (i.e. time.min() = 0)
                - end
                    - float
                    - time at which the task ends
                    - relative to the zero point in time (i.e. time.min() = 0)
                - start_slope
                    - float
                    - how steep the starting phase should be
                - end_slope
                    - float
                    - how steep the ending phase (reflection phase) should be
                - taskname
                    - str, optional
                    - name of the task added
                    - the default is None
                        - Will generate 'Task' and add a number one more than the current amount of tasks
                - weight
                    - float, optional
                    - weight to set the importance of the task with respect to the other tasks
                    - the default is 1
                - percent_complete
                    - float, optional
                    - percentage describing how much of the task is completed
                    - number between 0 and 1
                    - the default is 0
                - whole_area
                    - bool, optional
                    - whether to consider the whole area beneath the curves as 100% of the task
                        - this will especially in low percentages NOT line up with the actual GANTT-chart (second subplot) 
                    - otherwise will consider the area between 'start' and 'end' as 100% of the task
                        - will be exactly aligned with the GANTT-chart
                        - but it might be that 0% already has some area colored in
                    - the default is True
                - testplot
                    - bool, optional
                    - whether to show a testplot of the created task
                    - the default is False

            Raises
            ------
                - ValueError
                    - if 'percent_complete' bigger than 1 or smaller than 0 is passed

            Returns
            -------
                - task
                    - np.array
                    - an array of the percentages of workload over the whole task
            
            Dependencies
            ------------
                - matplotlib
                - numpy

            Comments
            --------


        """
        import matplotlib.pyplot as plt
        import numpy as np

        if percent_complete < 0 or percent_complete > 1:
            raise ValueError("'percent_complete has to be a float between 0 and 1!")

        #array of unit-timesteps relative to the starting date
        calc_time = np.arange(0, self.time.shape[0], 1)

        #update self.starts and self.ends
        
        if end > self.time.shape[0]-1:
            end_idx = self.time.shape[0]-1
        else:
            end_idx = end
        if start < 0:
            start_idx = 0
        else:
            start_idx = start
        self.starts = np.append(self.starts, self.time[int(start_idx)])
        self.ends   = np.append(self.ends,   self.time[int(end_idx)])

        #define task
        task = self.task_func(calc_time, start, end, start_slope, end_slope)

        if self.tasks.shape[0] == 0:
            self.tasks = np.append(self.tasks, task)
        else:
            self.tasks = np.vstack((self.tasks, task))
        
        #add weight
        self.weights = np.append(self.weights, weight)
        
        #add percentage that has been completed
        self.percent_complete = np.append(self.percent_complete, percent_complete)
        
        #add index of percentage that has been completed
        percent_idx = self.task_integral(calc_time, start, end, start_slope, end_slope, percent_complete, whole_area)
        self.percent_idxi = np.append(self.percent_idxi, percent_idx)
        
        #add task-name
        if taskname is None:
            taskname = f"Task {len(self.tasknames)+1:d}"
        self.tasknames.append(taskname)

        if testplot:
            fig = plt.figure(figsize=(16,9))
            ax = fig.add_subplot(111)
            fig.suptitle(f"Testplot for your {taskname}", fontsize=24)
            ax.plot(self.time, task, "-", label=taskname)
            ax.fill_between(self.time, task, where=(self.time < self.time[percent_idx]), alpha=.3, label=f"completion ({percent_complete*100}%)")
            ax.set_xlabel("Time [Your Unit]", fontsize=20)
            ax.set_ylabel("Relative Workload (Within Task) [-]", fontsize=20)
            ax.tick_params("both", labelsize=20)
            plt.tight_layout()
            plt.legend(fontsize=20)
            plt.show()

        return task


    def make_graph(
        self,
        projecttitle="Your Project", timeunit="Your Unit", today=None,
        colors=None, enumerate_tasks=True,
        show_totalwork=True, show_completion=True
        ):
        """
            - function to visualize the workload of a project w.r.t. the time
            - will create a plot inlcuding a clssical GANTT-plot and a variation

            Parameters
            ----------
                - projecttitle
                    - str, optional
                    - title of your project
                        - will be shown as title of plot created
                    - only relevant for the plot created
                    - the default is 'Your Project'
                - timeunit
                    - str, optional
                    - unit of 'time'
                    - only relevant for the plot created
                    - the default is 'Your Unit'
                - today
                    - float, optional
                    - current state
                    - will plot a vertical line at the current state in the plot created
                    - only relevant for the plot created
                    - the default is None (will not plot a line)
                - colors
                    - list, np.array, optional
                    - list of matplotlib colors or rgb-tupels
                    - the default is None
                        - will generate colors automatically
                - enumerate_tasks
                    - bool, optional
                    - whether to enumerate the tasks contained in the GANTT-instance
                        - will enumerate them in the order they got added to the GANTT-instance
                    - only relevant for the plot created
                    - the default is true
                - show_totalwork
                    - bool, optional
                    - whether to add a plot of the total work for each point in time
                    - only relevant for the plot created
                    - the default is True
                - show_completion
                    - bool, optional
                    - whether to show the completion of individual tasks
                    - only relevant for the plot created
                    - the default is True
                - testplot
                    - bool, optional
                    - whether to show a testplot of the created task
                    - the default is True
                
                Raises
                ------
                    - TypeError
                        - if 'colors' has wrong type
                    - ValueError
                        - if 'colors' has other length than 'self.tasks'

                Returns
                -------
                    - tasks_combined
                        - np.array
                        - combination of all tasks
                        - the maximum workload for a given point in time of all tasks combined is 100% 
                    - fig
                        - matplotlib figure object
                    - axs
                        - matplotlib axes object

                Dependencies
                ------------
                    - matplotlib
                    - numpy
                    - astroLuSt

                Comments
                --------
                    - 'whole_area' (argument of 'self.task', will be adjusted for each graph separately)
                        - make sure you set 'whole_area' to the value you need, when passing a task to the class

        """

        import numpy as np
        import matplotlib.pyplot as plt
        from astroLuSt.visualization.plotting import generate_colors

        if colors is None:
            colors = generate_colors(len(self.tasks)+2, cmap='nipy_spectral')[1:-1]
        elif type(colors) != (np.array and list):
            raise TypeError("'colors' has to be a list or an np.array!")
        elif len(colors) != len(self.task):
            raise ValueError("'colors' has to have the same length as 'self.tasks'!")
        
        if today is None:
            today = self.time.min()
        
        tasks_zeromin = self.tasks-self.tasks.min()
        
        try:
            tasks_zeromin.shape[1]
            weights = np.sum(tasks_zeromin, axis=0)
        except:
            weights = tasks_zeromin
        
        tasks_combined = tasks_zeromin/weights.max()



        #create figure
        fig = plt.figure(figsize=(16,9))
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)

        fig.suptitle(projecttitle, fontsize=24)


        #################
        #GANTT variation#
        #################

        #all tasks
        try:
            tasks_combined.shape[1]
        except:
            tasks_combined = np.array([tasks_combined])
        tasks_combined = (tasks_combined.T*self.weights).T    #weight all tasks

        for idx, (task, percent_idx, percent_complete, c) in enumerate(zip(tasks_combined, self.percent_idxi, self.percent_complete, colors)):
            ax1.plot(self.time, task, zorder=2+(1/(idx+1)), color=c, alpha=1, linestyle="-", linewidth=5)
            if show_completion:
                ax1.fill_between(self.time, task, where=(self.time < self.time[percent_idx]), zorder=1+(1/(idx+1)), color=c, alpha=.3)

        #total workload
        if show_totalwork:
            ax1.plot(self.time, np.sum(tasks_combined, axis=0), linestyle=":", color="k", alpha=.6, label="Total workload", zorder=0.9)
            ax1.fill_between(self.time, np.sum(tasks_combined, axis=0), where=(self.time>today), color="tab:grey", alpha=.2, label="TODO", zorder=0.9)
            if today is not None:
                ax1.fill_between(self.time, np.sum(tasks_combined, axis=0), where=(self.time<today), color="tab:green", alpha=.2, label="Finished", zorder=0.9)


        #######
        #GANTT#
        #######

        ax2.plot(self.time, np.ones_like(self.time), alpha=0)   #just needed to get correct xaxis-labels

        text_shift = self.time[1]-self.time[0]
        for idx, (tn, start, end, percent_complete, c) in enumerate(zip(self.tasknames[::-1], self.starts[::-1], self.ends[::-1], self.percent_complete[::-1], colors[::-1])):
            if enumerate_tasks:
                label = f"{len(self.tasks)-idx:d}. {tn}"
            else:
                label = f"{tn}"
            duration = end - start
            completion = duration*percent_complete
            ax2.barh(tn, completion, left=start, color=c, alpha=1.0, zorder=2)
            ax2.barh(tn, duration, left=start, color=c, alpha=0.5, zorder=1)
            ax2.text(end+text_shift, idx, f"{percent_complete*100}%", va="center", alpha=0.8, fontsize=18)
            ax2.text(start-text_shift, idx, label, va="center", ha="right", alpha=0.8, fontsize=18)
        
        #current point in time
        ax1.vlines(today, ymin=0, ymax=1, color="k", zorder=20, label="Today", linestyle="--", linewidth=2.5)
        ax2.vlines(today, ymin=-1, ymax=self.tasks.shape[0], color="k", zorder=20, label="Today", linestyle="--", linewidth=2.5)

        
        #make visually appealing
        ax2.set_ylim(-1, self.tasks.shape[0])
        ax2.minorticks_on()
        ax2.tick_params(axis='y', which='minor', left=False)
        ax2.xaxis.grid(which="both", zorder=0)
        ax2.get_yaxis().set_visible(False)
        
        #create legend
        ax1.legend(fontsize=16)

        #labelling
        ax1.set_xlabel("")
        ax1.set_ylabel("Relative Workload [-]", fontsize=20)
        ax1.set_xticklabels([])
        # ax1.set_xticks([])
        ax1.get_shared_x_axes().join(ax1, ax2)
        
        ax1.tick_params("both", labelsize=20)
        ax2.set_xlabel(f"Time [{timeunit}]", fontsize=20)
        ax2.tick_params("both", labelsize=20)


        axs = plt.gcf().get_axes()
        
        plt.tight_layout()

        return tasks_combined, fig, axs

    def make_classic_gantt(
        self,
        projecttitle="Your Project", timeunit="Your Unit", today=None,
        colors=None, enumerate_tasks=False,
        ):
        """
            - function to create a GANTT-graph in classical style
            
            Parameters
            ----------
                - projecttitle
                    - str, optional
                    - title of your project
                        - will be shown as title of plot created
                    - only relevant for the plot created
                    - the default is 'Your Project'
                - timeunit
                    - str, optional
                    - unit of 'time'
                    - only relevant for the plot created
                    - the default is 'Your Unit'
                - today
                    - float, optional
                    - current state
                    - will plot a vertical line at the current state in the plot created
                    - only relevant for the plot created
                    - the default is None (will not plot a line)
                - colors
                    - list, np.array, optional
                    - list of matplotlib colors or rgb-tupels
                    - the default is None
                        - will generate colors automatically
                - enumerate_tasks
                    - bool, optional
                    - whether to enumerate the tasks contained in the GANTT-instance
                        - will enumerate them in the order they got added to the GANTT-instance
                    - only relevant for the plot created
                    - the default is true
                
                Raises
                ------
                    - TypeError
                        - if 'colors' has wrong type
                    - ValueError
                        - if 'colors' has other length than 'self.tasks'

                Returns
                -------
                    - fig
                        - matplotlib figure object
                    - axs
                        - matplotlib axes object

                Dependencies
                ------------
                    - matplotlib
                    - numpy
                    - astroLuSt

                Comments
                --------

        """
        import numpy as np
        import matplotlib.pyplot as plt
        import astroLuSt.visualization.plotting_astroLuSt as alp

        #check shapes
        if colors is None:
            colors = alp.Plot_LuSt.color_generator(len(self.tasks), color2black_factor=.9, color2white_factor=.9)[0]
        elif type(colors) != (np.array and list):
            raise TypeError("'colors' has to be a list or an np.array!")
        elif len(colors) != len(self.task):
            raise ValueError("'colors' has to have the same length as 'self.tasks'!")

        #create plot
        fig = plt.figure(figsize=(16,9))
        ax1 = fig.add_subplot(111)
        fig.suptitle(projecttitle, fontsize=24)

        ax1.plot(self.time, np.ones_like(self.time), alpha=0)   #just needed to get correct xaxis-labels

        text_shift = self.time[1]-self.time[0]
        for idx, (tn, start, end, percent_complete, c) in enumerate(zip(self.tasknames[::-1], self.starts[::-1], self.ends[::-1], self.percent_complete[::-1], colors[::-1])):
            if enumerate_tasks:
                label = f"{len(self.tasks)-idx:d}. {tn}"
            else:
                label = f"{tn}"
            duration = end - start
            completion = duration*percent_complete
            ax1.barh(tn, completion, left=start, color=c, alpha=1.0, zorder=2)
            ax1.barh(tn, duration, left=start, color=c, alpha=0.5, zorder=1)
            ax1.text(end+text_shift, idx, f"{percent_complete*100}%", va="center", alpha=0.8, fontsize=18)
            ax1.text(start-text_shift, idx, label, va="center", ha="right", alpha=0.8, fontsize=18)
        
        #current point in time
        ax1.vlines(today, ymin=-1, ymax=self.tasks.shape[0], color="k", zorder=20, label="Today", linestyle="--", linewidth=2.5)

        #make visually appealing
        ax1.set_ylim(-1, self.tasks.shape[0])
        ax1.minorticks_on()
        ax1.tick_params(axis='y', which='minor', left=False)
        ax1.xaxis.grid(which="both", zorder=0)
        ax1.get_yaxis().set_visible(False)
        #labelling
        ax1.set_xlabel(f"Time [{timeunit}]", fontsize=20)
        ax1.tick_params("both", labelsize=20)

        ax1.legend(fontsize=16)

        axs = plt.gcf().get_axes()
        plt.tight_layout()


        return fig, axs

