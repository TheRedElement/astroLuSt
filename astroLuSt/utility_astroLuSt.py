

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
        if today is not None:
            ax1.vlines(today, ymin=-1, ymax=self.tasks.shape[0], color="red", zorder=20, label="Today", linestyle="--", linewidth=2.5)
            ax1.annotate('TODAY', xy=(today+2*text_shift, -0.8), color="k", fontsize=18)

        #make visually appealing
        ax1.set_ylim(-1, self.tasks.shape[0])
        ax1.minorticks_on()
        ax1.tick_params(axis='y', which='minor', left=False)
        ax1.xaxis.grid(which="both", zorder=0)
        ax1.get_yaxis().set_visible(False)
        #labelling
        ax1.set_xlabel(f"Time [{timeunit}]", fontsize=20)
        ax1.tick_params("both", labelsize=20)

        axs = plt.gcf().get_axes()
        plt.tight_layout()


        return fig, axs
