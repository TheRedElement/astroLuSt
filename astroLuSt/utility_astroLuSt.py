

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
            - start_task
                - saves the point in time for a task as starting point
            - end_task
                - saves the point in time for a task as starting point
        
        Attributes
        ----------
            - task
                - str
                - name of the task
            - start
                - datetime object, optional
                - the point in time, at which the process started
                - the default is None, since it will get set by start_task
            - end
                - datetime object, optional
                - the point in time, at which the process ended
                - the default is None, since it will get set by end_task

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
            - Function to denote the starting point of a task.

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
            - Function to denote the endpoint of a task.
            - Also prints out the time needed for the execution.

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
                - Needs to be called after start_task
        """
        from datetime import datetime

        self.end = datetime.now()
        self.needed_time = (self.end - self.start)
        printing = self.end.strftime("%Y-%m-%d %H:%M:%S.%f")
        print(f"--> Time needed for {self.task}: {self.needed_time}")
        print(f"--> Finished {self.task} at time {printing}")
    
#______________________________________________________________________________
#Class for printing tables
class Table_LuSt:
    #TODO: Add method to add new column
    #TODO: Add method to add another header row
    #TODO: latex_template(): make so that if separator == " " the "&" are above each other
    """
        Class to quickly print nice tables and save them to a text-file if need be.
        
        Methods
        -------
            - add_row
                - adds a row to the Table_LuSt object
            - add_column TODO: Implement
                - adds a whole column to the Table_LuSt object
            - add header_row TODO: Implement
                - adds a row to the current header of the Table_LuSt object     
            - print_header
                - prints out the header of the table
            - print_rows
                - prints out the rows of the table
            - print_table
                - prints out the whole table
            - latex_template:
                - Prints a latex template of the table
                    - If this template is copied to a .tex document it should print a table similar to the one created with this class
        
        Attributes
        ----------
            - table_name
                - str, optional
                - name of the table when saved in file
                - the default is None
            - header
                - list, optional
                - list containing
                    - the entries for the header
                - the default is None
            - rows
                - list of lists, optional
                - list containing
                    - a list for each row containing
                        - the entries for each cell in that row
                - each row has to have the same length
                - the default is None
            - formatstrings
                - list of lists, optional
                - list containing
                    - a list for each row containing
                        - the formatstring for each cell in that row
                - has to be of same dimension as rows
                - the default is None
            - separators
                - list, optional
                - list containing
                    - the separators after each column
                    - allowed are '|', '||' and ' '
                - has to be of same length as header
                - the default is None
            - alignments
                - list, optional
                - list containing
                    - the alignment tag for each column
                    - allowed are 'l', 'c', 'r' (left-bound, centered, rightbound)
                - has te of same length as header
                - the default is None
            - newsections
                - list, optional
                - list containing
                    - '-', '=' or False
                    - The first two will print out the respective symbol as separator before the entry
                    - False will print nothing before the entry
                - basically used to start a new section with the current row
                - has to be of same length as rows
                - the default is None
                
        Dependencies
        ------------
            - re


        Comments
        --------
    """

    def __init__(self, table_name=None, header=None, rows=None, formatstrings=None, separators=None, alignments=None, newsections=None):
        if table_name is None:
            self.table_name = "Table"
        else:
            self.table_name = table_name
        if header is None:
            self.header = []
        else:
            self.header = header
        if rows is None:
            self.rows = []
        #check if all rows have the same length
        elif all(len(row) == len(next(iter(rows))) for row in iter(rows)):
            self.rows = rows
        else:
            raise ValueError("All lists in rows (all rows) have to have the same length!")
        if formatstrings is None:
            self.formatstrings = []
            for row in self.rows:
                formatstring = []
                for r in row:
                    #make sure the type is correct for f string
                    if type(r) == str:
                        formattype = "s"
                    else:
                        formattype = ".0f"
                    formatstring.append("%6"+formattype)
                self.formatstrings.append(formatstring)
        else:
            self.formatstrings = formatstrings
        if separators is None:
            self.separators = ["|"]*len(self.header)
        else:
            self.separators = separators
        if alignments is None:
            self.alignments = ["c"]*len(self.header)
        else:
            self.alignments = alignments        
        if newsections is None:
            self.newsections = [False]*len(self.rows)
        else:
            self.newsections = newsections

        #combined variables
        self.num_width = str(len(str(len(self.rows)))+2)   #length of the string of the number of rows +2
        self.tablewidth = None  #width of the output table
        self.to_return_header = "No header created yet"

        #attributes that can't be set
        self.to_return_rows = "No rows created yet"
        self.complete_table = "No table created yet. Call Table_Lust_object.print_table() to create a table."
        
        #convert alignments entries to f-string formatters
        self.aligns = ["c"] + self.alignments   #add "c" for row-number column
        self.aligns = [align if align != "l" else "<" for align in self.aligns]
        self.aligns = [align if align != "c" else "^" for align in self.aligns]
        self.aligns = [align if align != "r" else ">" for align in self.aligns]

        #check if the shapes are correct
        if len(self.separators) != len(self.header):
            raise ValueError(f"len(separators) (currently: {len(self.separators):d}) has to be len(header) (currently: {len(self.header):d}).\n"
                            "This is because it specifies the separators between two columns.\n"
                            "The first column is per default the rownumber")
        if len(self.alignments) != len(self.header):
            raise ValueError(f"len(alignments) (= {len(self.alignments):d}) has to be len(header) (= {len(self.header):d}).\n"
                            "The first column the rownumber and set to c per default")
        if len(self.newsections) != len(self.rows):
            raise ValueError("len(newsections) has to be len(rows).")

        #check if all types are correct
        if any((sect != "-") and (sect != "=") and (sect != False) for sect in iter(self.newsections)):
            raise TypeError("The entries of newsections have to be either '-' or '=' or False!")
        if any((alignment != "l") and (alignment != "c") and (alignment != "r") for alignment in iter(self.alignments)):
            raise TypeError("The entries of alignments have to be either 'l' or 'c' or 'r' (left-bound, centered, right-bound)!")
        if any((separator != "|") and (separator != "||") and (separator != " ") for separator in iter(self.separators)):
            raise TypeError("The entries of separators have to be either '|' or '||' or ' '!")

    def __repr__(self):
        return ("\n"
                f"Table_LuSt(\n"
                f"table_name = {self.table_name},\n"
                f"header = {self.header},\n"
                f"rows = {self.rows},\n" 
                f"formatstrings = {self.formatstrings},\n"
                f"separators = {self.separators},\n"
                f"alignments = {self.alignments},\n"
                f"newsections = {self.newsections}\n"
                ")")
        
    def add_row(self, row, fstring=None, new_sect=False):
        """
            - Function to add another row to the table.
            - Adding a row also results in adding a formatstring for this row.

            Parameters
            ----------
                - row
                    - list
                    - list containing
                        - the entries for each cell
                - fstring
                    - list, optional
                    - list containing
                        - the corresponding formatstrings for the entries
                    - the default is None
                - new_sect
                    - bool or str, optional
                    - allowed are '-', '=', False
                    - wether to start a new section with this row
                        - Starts a new section when set to true
                    - the default is False
            
            Raises
            ------
                - ValueError
                        - If the row and fstring are not of the same length

            Returns
            -------

            Dependencies
            ------------

            Comments
            --------
        """

        #check correct type of new_sect
        if any((sect != "-") and (sect != "=") and (sect != False) for sect in iter(self.newsections)):
            raise TypeError("The entries of newsections have to be either '-' or '=' or False!")

        #add row and corresponding attributes
        self.rows.append(row)
        self.newsections.append(new_sect)
        if fstring is None and len(self.formatstrings) == 0:
            fstring = [""]*len(row)
        elif fstring is None and len(self.formatstrings) != 0:
            fstring = self.formatstrings[0]
        self.formatstrings.append(fstring)

        if len(row) != len(fstring):
            raise ValueError("len(row) as to be equal to len(fstring)")

    def add_column(self, header, rows, formatstrings, separator="|", alignment="c"):
        """
            - Function to add another column to the table.

            Parameters
            ----------
                - header
                    - str
                    - header of new column
                - formatstrings
                    - list, optional
                    - list containing
                        - strings
                        - the corresponding formatstrings for the entries
                - separator
                    - str, optional
                    - the separator to use for the column
                    - the default is '|'
                - alignment
                    - str, optional
                    - the alignment of the row
                    - the default is 'c'
            
            Raises
            ------
                - ValueError
                        - If the number of entries in the column is unequal to the number of rows in the table
                        - If the number of entries in formatstrings is unequal to the number of rows in the table

            Returns
            -------

            Dependencies
            ------------

            Comments
            --------
                - NOT IMPLEMENTED YET (Has issues)  

        """

        raise Warning("NOT IMPLEMENTED YET!")        
             
        # if len(self.rows) != len(rows):
        #     raise ValueError("rows (len = %i) has to be the same length as self.rows (len = %i)!"%(len(rows), len(self.rows)))
        # if len(self.formatstrings) != len(formatstrings):
        #     raise ValueError("formatstrings (len = %i) has to be the same length as self.formatstrings (len = %i)!"%(len(formatstrings), len(self.formatstrings)))
        
        # self.header.append(header)
        # self.separators.append(separator)
        # self.alignments.append(alignment)
        # for ridx, fidx in zip(range(len(self.rows)), range(len(self.formatstrings))):
        #     self.rows[ridx].append(rows[ridx])
        #     self.formatstrings[fidx].append(formatstrings[fidx])
                
    
    def add_header_row(self):
        raise Warning("NOT IMPLEMENTED YET!")

    def print_header(self, print_it=True, hide_rownumbers=False):
        """
            - Function to print out the header of the created table.

            Parameters
            ----------       
                - print_it
                    - bool, optional
                    - wether to print the header in the bash or not
                    - the default is True
                - hide_rownumbers
                    - bool, optional
                    - whether to show a column containing the rownumbers or not
                    - the default is False

            Raises
            ------

            Returns
            -------
                - to_return
                    - str
                    - a string of the header
                    - could be written to a file
                    - used in print_table() when writing to a file

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
        if len(self.separators) == 0:
            self.separators = ["|"]*len(self.header)
        
        seps = self.separators + ["|"]  #append some string to sparators since the last one will get omitted anyways

        #initialize header with a rownumber column (if wished)
        if hide_rownumbers:
            header_print = ""
        else:
            header_print = "{:"+self.aligns[0] + self.num_width +"}" + seps[0]
            header_print = header_print.format("#")
        #fill in the rest of the header and print it
        for fs, sep, align in zip(self.formatstrings[0], seps[1:], self.aligns[1:]):
            fs_digit = re.findall(r"\d+", fs)[0]
            header_print += "{:"+align +fs_digit+ "s}"+ sep
        to_print = header_print[:-1].format(*self.header)

        if print_it:
            print(to_print)
        self.tablewidth = len(to_print)  #total width the table will have 

        to_return = to_print + "\n"

        #set as attribute to access for latex formatting
        self.to_return_header = to_return   
        
        return to_return

    def print_rows(self, print_it=True, hide_rownumbers=False):
        """
            - Function to print out the rows of the table.

            Parameters
            ----------
                - print_it
                    - bool, optional
                    - wether to print the rows in the bash or not
                    - the default is True
                - hide_rownumbers
                    - bool, optional
                    - wehter to show a column containing the rownumbers or not
                    - the default is False
           
            Raises
            ------

            Returns
            -------
                - to_return
                    - str
                    - a string of the table-body (all rows)
                    - could be written to a file
                    - used in print_table() when writing to a file

            Dependencies
            ------------
            
            Comments
            --------
        """
        #initialize return
        to_return = ""

        #fill formatstring list, if not enough formatstrings were provided
        if len(self.rows) < len(self.formatstrings):
            diff = abs(len(self.rows) - len(self.formatstrings))
            to_add = [self.formatstrings[-1]]*diff
            for ta in to_add:
                self.formatstrings.insert(0, ta)
        row_num = 1 #keep track of the rownumber to enumerate table
        
        #print rows
        for row, fstring, new_sect in zip(self.rows, self.formatstrings, self.newsections):

            seps = self.separators + ["|"]   #append empty string to sparators since the last one will get omitted anyways

            #initialize row with its rownumber (if wished)
            if hide_rownumbers:
                row_print = ""
            else:
                row_print = "{:"+self.aligns[0] + self.num_width + "}" + seps[0]
                row_print = row_print.format(row_num)
            #add row-entries
            for fs, sep, align in zip(fstring, seps[1:], self.aligns[1:]):
                row_print += "{:"+ align + fs[1:] + "}" + sep
            row_num += 1

            #start a new section if specified
            if new_sect != False:
                if print_it:
                    print(new_sect*self.tablewidth)
                to_return += (new_sect*self.tablewidth + "\n")
            
            #print new row
            if print_it:
                print(row_print[:-1].format(*row))
            to_return += row_print[:-1].format(*row) + "\n"

        #set as attribute to access for latex formatting
        self.to_return_rows = to_return

        return to_return

    def print_table(self, save=False, writingtype="w", print_it=True, hide_rownumbers=False):
        """
            - Function to combine print_header and print_rows to display a nice table

            Parameters
            ----------
            - save
                - str or bool, optional
                - allowed is any string or False
                    - if set to False, the table will be printed in the shell
                    - if set to some string, a file with the respective name will be saved in addition to printing it in the shell
                - wether to save the created table to a file or just display it in the shell
                - the default is False
            - writingtype
                - str, optional
                - the type of writing one wants to execute for saving the table
                - allowed are the standard strings for open()
                    - "w" for writing to a file (a file will be created if not existent)
                    - "a" for appending to an existing file
                    - "x" for creating a new file (will raise an error if the file exists)
                - the default is "w"
            - print_it
                - bool, optional
                - wether to print the table in the bash or not
                - the default is True
            - hide_rownumbers
                - bool, optional
                - wehter to show a column containing the rownumbers or not
                - the default is False

            Raises
            ------
            - ValueError
                - if wrong arguments are passed as save

            Returns
            -------

            Dependencies
            ------------
                
            Comments
            --------
        """
        #display (for print_it==True) and save (for type(save)==str) 
        head = Table_LuSt.print_header(self, print_it=print_it, hide_rownumbers=hide_rownumbers)
        if print_it:
            print("="*self.tablewidth)
        rows = Table_LuSt.print_rows(self, print_it=print_it, hide_rownumbers=hide_rownumbers)
        if print_it:
            print("-"*self.tablewidth)

        if type(save) == str:
            outfile = open(save, writingtype)
            outfile.write("\n")
            outfile.write(self.table_name + "\n")
            outfile.write(len(self.table_name)*"-" + "\n")
            outfile.write(head)
            outfile.write("="*self.tablewidth + "\n")
            outfile.write(rows)
            outfile.write("-"*self.tablewidth + "\n\n")
            outfile.close()

        #construct one string of the complete table (needed for latex_template)
        self.complete_table = "="*self.tablewidth + "\n" + head + "-"*self.tablewidth +"\n" + rows + "-"*self.tablewidth +"\n"
        
    def latex_template(self, save=False, writingtype="w", print_it=False, print_latex=True, hide_rownumbers=False):
        #TODO: make so that if separator == " " the "&" are above each other
        """
            - Function to create a template that can be inserted into your latex document and will result in a nice table.

            Parameters
            ----------
                - save
                    - str or bool, optional
                    - allowed is any string or False
                        - if set to False, the table will be printed in the shell
                        - if set to some string, a file with the respective name will be saved in addition to printing it in the shell
                    - wether to save the created table to a file or just display it in the shell
                    - the default is False
                - writingtype
                    - str, optional
                    - the type of writing one wants to execute for saving the table
                    - allowed are the standard strings for open()
                        - "w" for writing to a file (a file will be created if not existent)
                        - "a" for appending to an existing file
                        - "x" for creating a new file (will raise an error if the file exists)
                    - the default is "w"
                - print_it
                    - bool
                    - wether to print the headr in the bash or not
                - print_latex
                    - bool
                    - wether to print the latex code in the bash or not
                - hide_rownumbers
                    - bool, optional
                    - wehter to show a column containing the rownumbers or not
                    - the default is False
        
            Raises
            ------
                - ValueError
                    - if wrong arguments are passed as save

            Returns
            -------

            Dependencies
            ------------ 
                - re

            Comments
            --------
        """        
        import re

        #calculate current table
        Table_LuSt.print_table(self, save=save, print_it=print_it)

        #set up table environment
        latex_table = "\\begin{table}[]"+"\n"
        latex_table += (4*" " +"\centering\n")

        #set up tabular (show/hide rownumbers according to specification)
        if hide_rownumbers:
            tabulator = 4*" " + "\\begin{tabular}{"
            for sep, align in zip(self.separators, self.alignments):
                tabulator += sep+align
            tabulator = re.sub("\|", "", tabulator, 1)
        else:
            tabulator = 4*" " + "\\begin{tabular}{c"
            for sep, align in zip(self.separators, self.alignments):
                tabulator += sep+align
        tabulator += "}\n"
        latex_table += tabulator

        if hide_rownumbers:
            complete_table = re.sub("(?<=\n)\s+[\d\#]\s\|*", "", self.complete_table)
        else:
            complete_table = self.complete_table

        #change sectionseparators to \hline
        hline = re.sub("-{%i,}"%(self.tablewidth), r"\\hline", complete_table)
        hlinehline = re.sub("={%i,}"%(self.tablewidth), r"\\hline\\hline", hline)

        #chang column separators to &
        andpercent = re.sub(r"(?<=\w)[^\S\n]+(?=\w+)", " & ", hlinehline)  #substitute whitespace separators
        andpercent = re.sub(r"\|+", " & ", andpercent)                     #substitute all other separators


        #add latex newline while keeping "\n" for printing (plus ignore lines with \hline)
        newlines = re.sub(r"(?<!\\hline)\n", r" \\\\ \n", andpercent)
        
        #add proper indentation
        indent = re.sub(r"\n", r"\n"+8*" ", newlines)

        #make sure that # becomes \#
        if hide_rownumbers:
            hashtag = re.sub(r"\ #", r"", indent)
        else:
            hashtag = re.sub(r"\ #", r"\#", indent)

        #add tablebody to latex_table
        latex_table += (8*" " + hashtag + "\n")

        #end tabular
        latex_table += 4*" " + "\\end{tabular}\n"
        #add caption
        latex_table += 4*" " + "\\caption{ADD YOUR CAPTION}\n"
        #add label
        latex_table += 4*" " + "\\label{tab:ADD SOME LABEL}\n"
        #end table
        latex_table += "\\end{table}"

        latex_table = re.sub(r"(?<=\n)\ +\n", "", latex_table)

        #print and write to file according to specification
        if print_latex:
            print(latex_table)        
        if type(save) == str:
            outfile = open(save, writingtype)
            outfile.write("\n")
            outfile.write(self.table_name + "\n")
            outfile.write(len(self.table_name)*"-" + "\n")
            outfile.write(latex_table)
            outfile.write("\n\n")
            outfile.close()

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
        import astroLuSt.plotting_astroLuSt as alp

        if colors is None:
            colors = alp.Plot_LuSt.color_generator(len(self.tasks), color2black_factor=.9, color2white_factor=.9)[0]
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
        import astroLuSt.plotting_astroLuSt as alp

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


class MindMap:
    """
        - class to create a very primitive mindmap using python code

        Attributes
        ----------
            - node_contents
                - list of strings
                - text corresponding to the node
            - node_levels
                - list of int
                - levels corresponding to the node
                - a root is defined as node_level == 0
            - hide_branch
                - list of bools
                - whether to hide all the following nodes
                - NOT IMPLEMENTED YET
            - edge_froms
                - list of strings
                - origins of all edges between nodes
                - the entries are entries of 'node_contents'
            - edge_tos
                - list of strings
                - ends of all edges between nodes
                - the entries are entries of 'node_contents'
            - edge_weights
                - list of float
                - weight for each edge
                - determines size of edge in graph
            - edge_types
                - list of str
                - which line-style to use when drawing the edge
                
        Methods
        -------
            - get_nodes_edges
                - function to get dicts of nodes and edges
            - check_keys
                - function to check if correct keys are provided to relationships and carac
            - generate_node_colors
                - function to generate colors for plotting the nodes
            - get_fibonacci_disc
                - function to generate a fibonacci disc
                - used to generate node positions when plotting
            - sector_mask
                - function to mask an array
                - mask will be a sector of a circle
            - generate_node_positions
                - function to generate node positions
                    - used if no positions are provided to draw_MM
            - draw_MM
                - function to create an image of the MM
            - change_wallpaper
                - function to change a windows desktop wallpaper to a image provided by its absolute path
            - save
                - function to save a created MindMap to a text file
                - NOT IMPLEMENTED YET
            - load
                - function to load a MindMap saved as a text file
                - the file has to follow a specific structure (see documentation of load())

        Dependencies
        ------------
            - matplotlib
            - numpy
            - re
            - inspect
    """

    def __init__(
        self,
        node_contents=None, node_levels=None, hide_branch=None,  #node attributes
        edge_froms=None, edge_tos=None, edge_weights=None, edge_types=None, #edge attributes
        ):

        import numpy as np
        
        #nodes
        if node_contents is None:
            self.node_contents = np.array([], dtype=str)
        else:
            self.node_contents = np.array(node_contents)
        if node_levels is None:
            self.node_levels = np.zeros_like(self.node_contents, dtype=np.int16)
        else:
            self.node_levels = np.array(node_levels)
        if hide_branch is None:
            self.hide_branch = np.zeros_like(self.node_contents, dtype=bool)
        else:
            self.hide_branch = np.array(hide_branch, dtype=bool)

        #edges
        if edge_froms is None:
            self.edge_froms = np.array([], dtype=str)
        else:
            self.edge_froms = np.array(edge_froms)
        if edge_tos is None:
            self.edge_tos = np.empty_like(self.edge_froms, dtype=str)
        else:
            self.edge_tos = np.array(edge_tos)
        if edge_weights is None:
            self.edge_weights = np.ones_like(self.edge_froms, dtype=np.float16)
        else:
            self.edge_weights = np.array(edge_weights)
        if edge_types is None:
            self.edge_types = np.empty_like(self.edge_froms, dtype=str)
        else:
            self.edge_types = np.array(edge_types)

        self.nodes = {}
        self.edges = {}
        
        #check shapes and update self.nodes and self.edges
        self.get_nodes_edges()


    def get_nodes_edges(self, fix_shapes=False):
        """
            - function to get dicts of nodes and edges
        """
        import inspect
        import numpy as np
        
        
        #check shapes
        attributes = inspect.getfullargspec(self.__init__).args[1:]
        node_attributes = attributes[:int(attributes.index("edge_froms"))]
        edge_attributes = attributes[int(attributes.index("edge_froms")):]


        for na in node_attributes:
            #check if shapes are correct (exclude node_colors, becasue it is a tuple of 2 RGB values)
            if eval("self."+na+".shape") != self.node_contents.shape:
                if not fix_shapes:
                    raise ValueError(f"All node-attributes have to have the same shape. Shape of {na} {eval('self.'+na+'.shape')} does not match the other shapes {self.node_contents.shape}")

                #fix shapes by filling them with empties
                exec(f"self.{na} = np.empty_like(self.node_contents, dtype=self.{na}.dtype)")
            self.nodes[na] = eval("self."+na)

        for ea in edge_attributes:
            #check if shapes are correct
            if eval("self."+ea+".shape") != self.edge_froms.shape:
                if not fix_shapes:
                    raise ValueError(f"All edge-attributes have to have the same shape. Shape of {ea} {eval('self.'+ea+'.shape')} does not match the other shapes {self.edge_froms.shape}")
                
                #fix shapes by filling them with empties
                exec(f"self.{ea} = np.empty_like(self.edge_froms, dtype=self.{ea}.dtype)")
            self.edges[ea] = eval("self."+ea)


    def __repr__(self):

        return ("MindMap(\n"
                f"node_contents  = {self.node_contents},\n"
                f"node_levels    = {self.node_levels},\n"
                f"hide_branch    = {self.hide_branch},\n"
                f"edge_froms     = {self.edge_froms},\n"
                f"edge_tos       = {self.edge_tos},\n"
                f"edge_weights   = {self.edge_weights},\n"
                f"edge_typse     = {self.edge_types},\n"
                ")\n")


    #Error handling
    def check_keys(self, relationships={"from":[], "to":{}}, carac={"ID":[], "type":[]}):
        """
            - function to check if correct keys are provided to relationships and carac
        """
        #check if correct keys provided
        if ("from" or "to") not in relationships.keys():
            raise KeyError(f"""{relationships.keys()} are not valid keys for 'relationships'!
            'relationships' has to be of the following structure: {'{'}'from':[...], 'to':[...]{'}'} !""")
        if ("ID" or "type") not in carac.keys():
            raise KeyError(f"""{carac.keys()} are not valid keys for 'carac'!
            'carac' has to be of the following structure: {'{'}'ID':[...], 'type':[...]{'}'} !""")
        else:
            pass

    def generate_node_colors(self, cmap="jet"):
        """
            - function to generate node colors according to the node level

            Parameters
            ----------
                - cmap
                    - str, optional
                    - matplotlib colormap name
                    - the default is 'jet'
             Raises
             ------

             Returns
             -------
                - node_colors
                    - np.array
                    - contains rgb-tripels mapped to each node
            
            Dependencies
            ------------
                - matplotlib
                - numpy
        """
        # import astroLuSt.plotting_astroLuSt as alp
        import matplotlib.pyplot as plt
        import numpy as np

        #initialize with color-generator colors
        node_colors = np.empty((self.node_contents.shape[0], 3))
        if len(node_colors) > 0:
            # level_colors = alp.Plot_LuSt.color_generator(ncolors=self.node_levels.max()+1, color2black_factor=.8, color2white_factor=1)[0]
            level_colors = eval(f"plt.cm.{cmap}(np.linspace(0,1,self.node_levels.max()+1))[:,:-1]")
            for lvl in np.unique(self.node_levels):
                node_colors[(lvl == self.node_levels)] = level_colors[lvl]
        
        return node_colors

    def get_fibonacci_disc(self, n, spacing=1):
        """
            - function to generate a fibonacci disc
            - http://blog.marmakoide.org/?p=1

            Parameters
            ----------
                - n
                    - int
                    - number of points to create
                - radial_offset
                    - float, optional
                    - offset in radial direction
                    - the default is 0
                - spacing
                    - float, optional
                    - amount of space between the points
                    - the default is 1

            Raises
            ------

            Returns
            -------
                - coords
                    - np.array
                    - 2d-array containing coordinates of the generated datapoints
            
            Dependencies
            ------------
                - numpy

            Comments
            --------

        """
        
        import numpy as np

        r = np.sqrt(np.arange(0, n, 1)/(n)).reshape((n, 1)) #radius of each point

        phi_golden = np.pi * (3 - np.sqrt(5)) #golden ratio
        phi = phi_golden * np.arange(0, n, 1) #polar angle

        #convert from polar coords to karthesian
        coords = np.zeros((n,2)) + r
        coords[:,0] *= np.cos(phi)
        coords[:,1] *= np.sin(phi)
        coords *= spacing

        return coords

    def sector_mask(self, coords, n_roots, center=[0,0], radius=1, start_angle=0, end_angle=None, testplot=False):
        """
            - function to divide 'coords' into 'n_roots' equally large partitions
                - each of the partitions is a sector of a circle with radius 'radius'
                - the circles origin is located at 'center'
            
            Parameters
            ----------
                - coords
                    - np.array
                    - 2d-array containing x- and y value tuples
                - n_roots
                    - int
                    - number of partitions to generate
                - center
                    - tuple, list, np.array, optional
                    - location of the origin of the circle, which is used to partition the data
                    - the default is [0,0]
                - radius
                    - float, optional
                    - radius of the circle, which is used to partition the data
                    - the default is 1
                - start_angle
                    - float, optional
                    - the angle to start partitioning at
                    - the default is 0
                - end_angle
                    - float, optional
                    - the angle to end partitioning at
                    - the default is None
                        - will lead to using a full circle (i.e. 2*pi)
                -testplot
                    - bool, optional
                    - whether to show a testplot
                    - the default is False                
            
            Raises
            ------

            Returns
            -------
                - coords_masks
                    - list
                        - contains 'n_roots' boolean masks
                - angle_range
                    - np.array
                    - range of angles used to partition the data
            
            Dependencies
            ------------
                - numpy
                - matplotlib
            
            Comments
            --------

        """
        import numpy as np
        import matplotlib.pyplot as plt
        
        if end_angle is None:
            end_angle = 2*np.pi


        #angles of partitions
        angle_range = np.linspace(start_angle, end_angle, n_roots+1)
        
        x = coords[:,0]
        y = coords[:,1]

        coords_masks = []
        for tmax, tmin in zip(-angle_range[:-1]+0.5*np.pi, -angle_range[1:]+0.5*np.pi):
            
            #carthesian to polar
            r2 = (x-center[0])**2 + (y-center[1])**2
            theta = np.arctan2(x-center[0], y-center[1]) - tmin
            theta %= 2*np.pi #wrap angles between 0 and 2*pi

            #radius mask
            rad_mask = (r2 <= radius**2)
            #angular mask
            angle_mask = (theta <= (tmax-tmin))

            coords_masks.append(rad_mask*angle_mask)
        
        if testplot:
            plt.figure(figsize=(10,10))
            for cm in coords_masks:
                plt.scatter(coords[cm][:,0], coords[cm][:,1])
            plt.show()


        return coords_masks, angle_range

    def generate_node_positions(self, 
        section_separation=0.1, spacing=1,
        center=[0,0], start_angle=0, end_angle=None,
        correction_exponent=2,
        ):
        """
            - function to generate some default node-positions
            - will generate positions consisting of datapoints on a fibonacci disc with radius 1 around 'center'
            - will generate sectors corresponding to the number of root nodes (nodes with 'node_level' == 0)

            Parameters
            ----------
                - section_separation
                    - float, optional
                    - measure of how much each sector shall be separated from the other ones
                    - the default is 0.1
                - spacing
                    - float, optional
                    - amount of space between the points
                    - the default is 1
                - center
                    - tuple, list, np.array, optional
                    - location of the origin of the circle, which is used to partition the data
                    - the default is [0,0]
                - start_angle
                    - float, optional
                    - the angle to start partitioning the fibonacci disc at
                    - the default is 0
                - end_angle
                    - float, optional
                    - the angle to end partitioning the fibonacci disc at
                    - the default is None
                        - will lead to using a full circle (i.e. 2*pi)
                - correction_exponenet
                    - float, optional
                    - exponent to generate more datapoints
                    - needed in case one branch has a lot of nodes
                        - thus, even partitioning will lead to an IndexError
                    - the default is 2

            Raises
            ------

            Returns
            -------
                - node_positions
                    - np.array
                    - 2d
                    - coordinates for each node in the MindMap
            
            Dependencies
            ------------
                - numpy
            
            Comments
            --------
        """

        import numpy as np

        n_roots = np.count_nonzero((self.node_levels == 0))
        n_nodes = self.node_contents.shape[0]

        #generate coordinates
        coords = self.get_fibonacci_disc(int(n_nodes**correction_exponent), spacing=spacing)
        divisions, angle_range = self.sector_mask(
            coords, n_roots, center=center,
            radius=spacing, start_angle=start_angle, end_angle=end_angle,
            testplot=False)

        #generate shifts to separate sections
        shifts = section_separation*np.array([
            np.cos(angle_range),
            np.sin(angle_range)
            ]).T

        #initiate node positions
        node_positions = np.zeros(shape=(self.node_contents.shape[0],2))
        
        min_level = self.node_levels.min()
        if min_level != 0:
            print("Make sure that at least one of your node_leves == 0. If this is not the case, the plotting might lead to unwanted results.")

        #set root nodes (used as starting point for main branches)
        for d, s, root, in zip(divisions, shifts, self.node_contents[(self.node_levels==min_level)]):

            coords[d] = coords[d]+s #update coordinates for each sector with the sector separation
            

            distance_sort = np.argsort((coords[d][:,0]**2 + coords[d][:,1]**2)) #sort each sector by distance to (0,0)

            root_coords = coords[d][distance_sort[0]]   #coordinates for the root note - point closest to (0,0) of the respective sector
            node_positions[(self.node_contents == root)] = root_coords      #set node_position for root to root_coords


        #set coordinates for all other nodes
        #iterate until deepest element is reached
        max_depth = self.node_levels.max()
        d = min_level
        while d <= max_depth:
            cur_ids = self.node_contents[(self.node_levels==d)]

            #iterate over all elements in the current depth
            for idx, cur_id in enumerate(cur_ids):
                from_to_idx = (self.edge_froms == cur_id)

                #iterate over all edges from current depth to next depth
                for idx_sub, (ef, et) in enumerate(zip(self.edge_froms[from_to_idx], self.edge_tos[from_to_idx])):
                    mask_sub, angle_range = self.sector_mask(
                        coords, 1, center=node_positions[(self.node_contents == ef)][0],
                        radius=spacing, start_angle=0, end_angle=2*np.pi,
                        testplot=False
                    )
                    distance_sort_sub = np.argsort((coords[mask_sub[0]][:,0] - node_positions[(self.node_contents == ef)][0,0])**2 + (coords[mask_sub[0]][:,1] - node_positions[(self.node_contents == ef)][0,1])**2 )
                    

                    #check if shortest distance is already taken
                    #if so, check the next shortest
                    #repeat until one distance is not taken
                    checkidx = 0
                    while coords[mask_sub[0]][distance_sort_sub[checkidx]] in node_positions:
                        checkidx += 1
                    node_positions[(self.node_contents == et)] = coords[mask_sub[0]][distance_sort_sub[checkidx]]

            d+=1    #proceed to next depth
        
        return node_positions


    #MM operations
    def draw_MM(self,
        node_positions=None, node_sizes=1, node_colors="jet", node_shape="o", node_borderwidth=0, node_bordercolor="k",     #Nodes
        edge_colors="k", edge_styles="-",
        font_size=12, font_color="k",
        plt_style="default",
        section_separation=0.1, spacing=1,
        start_angle=0, end_angle=None,
        correction_exponent=2,
        xmargin=0.15, ymargin=0.15, pad=1.08,
        figsize=(16,9), ax_pos=111,
        save=False, dpi=180):               
        """
            - function to show the created Mind-Map

            Parameters
            ----------
                - node_sizes
                    - list, int, optional
                    - list of sizes for each node
                    - the default is None
                        - will outogenereate a size according to the specified "level" of a node
                        - L0 > L1 > L2 > ...
                - node_colors
                    - list, str, optional
                    - list of colors for the nodes
                    - str of a matplotlib colormap-name
                    - the default is 'jet'
                        - Colors generated according to a matplotlib colormap
                        - will generate colors, according to the "level" of each node
                        - L1: Color1, L2: Color2, ...
                - node_shape
                    - str, optional
                    - defines shape of the nodes
                    - uses matplotlib markers
                    - the default is "o"
                - node_borderwidth
                    - float, optional
                    - defines the width of the border surrounding a node
                    - the default is 0, i.e. no border
                - node_bordercolor
                    - str, rgb-triple optional
                    - defines color of the border surrounding a node
                    - the default is black
                - edge_colors
                    - list, str, optional
                    - list of colors for all the edges
                    - the default is "black"
                - edge_styles
                    - list, str, optional
                    - list of the styles to use for each edge
                    - uses matplotlib linestyle strings
                    - the default is a solid line
                - font_size
                    - int, optional
                    - font size of the IDs (content) of each node
                    - the default is 12
                - font_color
                    - str, optional
                    - color of the IDs (content) of each node
                    - the default is "black"
                - plt_style
                    - str, optional
                    - which style of matplotlib to use for plotting
                    - the default is "default"
                - section_separation
                    - float, optional
                    - measure of how much each branch shall be separated from the other ones
                    - the default is 0.1
                - spacing
                    - float, optional
                    - amount of space between the nodes
                    - the default is 1
                - center
                    - tuple, list, np.array, optional
                    - location of the origin of the circle, which is used to partition the datapoints
                    - the default is [0,0]
                - start_angle
                    - float, optional
                    - the angle to start partitioning the fibonacci disc at
                    - the default is 0
                - end_angle
                    - float, optional
                    - the angle to end partitioning the fibonacci disc at
                    - the default is None
                        - will lead to using a full circle (i.e. 2*pi)
                - correction_exponenet
                    - float, optional
                    - exponent to generate more datapoints
                    - needed in case one branch has a lot of nodes
                        - thus, even partitioning will lead to an IndexError
                    - the default is 2
                - xmargin
                    - float, optional
                    - some value to specify the distance of the data towards the borders
                    - the default is 0.15
                - ymargin
                    - float, optional
                    - some value to specify the distance of the data towards the borders
                    - the default is 0.15
                - pad
                    - int, optional
                    - padding in plt.tight_layout()
                    - high values will push the datapoints closer together
                    - low values will pull the datapoints further apart
                    - the default 1.08
                - figsize
                    - tuple, optional
                    - size of the figure
                    - the default is (16,9)
                - ax_pos
                    - int, optional
                    - matplotlib axis position specifier
                    - position of the axis used for the MindMap
                    - useful if you want to include something else in the figure as well
                    - the default is '111'
                        - will fill the whole figure
                - save
                    - str, bool, optional
                    - whether to save the produced output
                        - will be saved if a string is provided
                        - will be saved to that respective location
                    - the default is False
                - dpi
                    - int, optional
                    - the resolution to use for saving the image
                    - the default is 180

            Raises
            ------
                - IndexError
                    - in case too little nodes for a particular branch have been generated
                    - also gives the hint of increasing 'correction_exponen'

            Returns
            -------
                - fig
                    - matplotlib figure object
                - ax
                    - matplotlib axes object

            Dependencies
            ------------
                - matplotlib

        """
        import matplotlib.pyplot as plt

        #change background        
        plt.style.use(plt_style)

        #generate node_positions
        if node_positions is None:
            try:
                node_positions = self.generate_node_positions(
                    section_separation=section_separation, spacing=spacing,
                    center=[0,0], start_angle=start_angle, end_angle=end_angle,
                    correction_exponent=correction_exponent,
                )
            except IndexError as i:
                print(f"ORIGINAL ERROR: {i}")
                raise IndexError("Try to increase 'correction_exponent' in order to ensure not falling out of bounds.")
        if type(node_colors) == str:
            #treat node_colors differently, since it can be an array of RGB tripels
            node_colors = self.generate_node_colors(node_colors)


        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(ax_pos)
        ax.scatter(
            node_positions[:,0], node_positions[:,1],
            s=node_sizes*5000/(self.node_levels+1), c=node_colors, marker=node_shape,
            linewidth=node_borderwidth, edgecolor=node_bordercolor,
            zorder=1
            )
        for nps, content in zip(node_positions, self.node_contents):
            ax.text(
                nps[0], nps[1], content,
                horizontalalignment="center", verticalalignment="center", multialignment="center",
                fontsize=font_size, color=font_color,
                )
        for ef, et in zip(self.edge_froms, self.edge_tos):
            np_from = node_positions[(ef == self.node_contents)]
            np_to   = node_positions[(et == self.node_contents)]
            ax.plot((np_from[0,0], np_to[0,0]), (np_from[0,1], np_to[0,1]), linestyle=edge_styles, color=edge_colors, zorder=0)
        
        #set axis margins (to push datapints around in the frame)
        ax.margins(xmargin, ymargin)

        #hide axes
        plt.axis("off")
        plt.tight_layout(pad=pad)

        #save
        if type(save) == str:
            plt.savefig(save, dpi=dpi, bbox_inches='tight', pad_inches=0)

        return fig, ax

    def change_wallpaper(self, absolute_path):
        """
            - function to change the windows-wallpaper to your created mindmap

            Parameters
            ----------
                - absolute_path
                    - str
                    - the ABSOLUTE path to the source file

            Raises
            ------

            Returns
            -------

            Dependencies
            ------------
                - ctypes
            
            Comments
            --------

        """
        import ctypes
            
        #change wallpaper
        try:
            ctypes.windll.user32.SystemParametersInfoW(20, 0, absolute_path , 0)
        except Exception as e:
            print(f"ORIGINAL ERROR: {e}")
            raise ValueError("Probaly you have to change the path to your list of TODOs")
        
        pass


    #saving and loading
    def save(self, path="MM_temp.txt"):
        #TODO: Implement - use code from generate_node_positions?
        #TODO: Make sure to leave one empty line at the start of the document!
        """
            - function to save the MindMap as an indented list in a text-file
        """
        import numpy as np

        max_depth = self.node_levels.max()
        min_depth = self.node_levels.min()
        print()


        roots = self.node_contents[(self.node_levels==min_depth)]
        for root in roots:
            print(root)
            connection_idxs = (self.edge_froms==root)

            while np.any(connection_idxs):
                for c_sub in self.edge_tos[connection_idxs]:
                    print("\t"*self.node_levels[(self.node_contents==c_sub)][0], c_sub)
                    connection_idxs_sub = (self.edge_froms==c_sub)

                    for c_2sub in self.edge_tos[connection_idxs_sub]:
                        print("\t"*self.node_levels[(self.node_contents==c_2sub)][0], c_2sub)

                    connection_idxs = (self.edge_froms==c_sub)


        raise NotImplementedError("NOT IMPLEMENTED YET!")
        pass

    def load(self, path):
        """
            - function to load MindMap from an indented bullet-list in a md-file
            - each node has to be lead by '- '
            - an example would be: \n
                >>> - Parent1  \n
                >>>     - Child11 \n
                >>>         - Child111 \n
                >>>     - Child12 \n
                >>>         - Child121 \n
                >>>         - Child122 \n
                >>> - Parent2 \n
                >>>     - Child21 \n
                >>>     - ... \n
            
            Parameters
            ----------
                - path
                    - str
                        - the path to the source file
                        - has to be of above mentioned structure

            Raises
            ------

            Returns
            -------

            Dependencies
            ------------
                - numpy
                - re
            
            Comments
            --------
        """
        import re
        import numpy as np

        # raise Warning("Not implement yet!")
        infile = open(path)
        content = infile.read()
        infile.close()

        entries = re.findall(r"(?<=- \b).+", content)
        indents = re.findall(r"^[^\S\n]*(?=-)", content, flags=re.M)

        #add some element which will be ignored
        #only needed to ensure that the last actual element gets used in the MindMap
        entries_ = np.append(entries, "IGNORE")
        indents_ = np.append(indents, "")

        #get self.carac ('ID's and 'type's)
        self.node_contents = np.array([e for e in entries_])
        self.node_levels = np.array([len(i)//4 for i in indents_[:len(entries_)]])

        max_depth = self.node_levels.max()                      #deepest element
        d = 0                                                   #initiate at root node to get into loop
        #iterate until the deepest element is reached
        while d <= max_depth:
            cur_ids = self.node_contents[(self.node_levels==d)]
            sub_ids = self.node_contents[(self.node_levels==(d+1))]

            #iterate over all elements in the current depth (get parents)
            for cur_id, next_id in zip(cur_ids[:-1], cur_ids[1:]):
                next_idx = int(np.where(self.node_contents == next_id)[0])
                cur_idx = int(np.where(self.node_contents == cur_id)[0])
                
                #iterate over all elements in the next depth (get children)
                for sub_id in sub_ids:
                    sub_idx = int(np.where(self.node_contents == sub_id)[0])

                    #check if the children are before the next and after the previous parent
                    #i.e.: check if the children belong to the respective layer
                    if cur_idx < sub_idx and sub_idx < next_idx:
                        
                        #append to the respective data-structures
                        self.edge_froms = np.append(self.edge_froms, cur_id)
                        self.edge_tos = np.append(self.edge_tos, sub_id)
            
            #proceed in next depth
            d += 1
        
        #discard "IGNORE"-element
        self.node_contents = self.node_contents[:-1]
        self.node_levels = self.node_levels[:-1]

        #update attributes
        self.get_nodes_edges(fix_shapes=True)

        pass