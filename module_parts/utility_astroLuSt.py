

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
    
#______________________________________________________________________________
#Class for printing tables
#TODO: Add method to output latex template for table
#TODO: Add choice of separator?
#TODO: Add something like add_section()
#TODO: Add check that formatstrings matches rows for passing the formatstrings directly or initiate correctly
#TODO: Add save to file option in plot_table
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
            - formatstrings
                --> list of lists, optional
                --> list containing
                    ~~> a list for each row containing
                        + the formatstring for each cell in that row
                --> has to be of same dimension as rows
                --> the default is None
            - separators
                --> list, optional
                --> list containing
                    ~~> the separators after each column
                    ~~> Allowed are '|', '||' and ''
                --> has to be of same length as header
                --> the default is None
            - sections
                #TODO
                --> list, optional
                --> list containing 
                

        Dependencies
        ------------
            - re


        Comments
        --------
    """

    def __init__(self, header=None, rows=None, formatstrings=None, separators=None, sections=None):
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
        else:
            self.formatstrings = formatstrings
        if separators is None:
            self.separators = ["|"]*len(self.header)
        else:
            self.separators = separators
        if sections is None:
            self.sections = [""]*len(self.rows)
        else:
            self.sections = sections

        self.num_width = str(len(str(len(self.rows)))+2)   #length of the string of the number of rows +2
        self.tablewidth = None  #width of the output table

    #check if the shapes are correct
        if len(self.separators) != len(self.header):
            raise ValueError("len(separators) has to be len(header).\n"
                            "This is because it specifies the separators between two columns.\n"
                            "The first column is per default the rownumber")
        if len(self.sections) != len(self.rows):
            raise ValueError("len(sections) has to be len(rows).")

    #check if all types are correct
        if any((sect != "-") and (sect != "=") and (sect != "") for sect in iter(self.sections)):
            raise TypeError("The entries of sections have to be either '-' or '=' or ''!")

    def __repr__(self):
        return ("\n"
                f"Table_LuSt(\n"
                f"rows = {self.rows},\n" 
                f"header = {self.header},\n"
                f"formatstrings = {self.formatstrings}),\n"
                f"separators = {self.separators}")
        
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
        if fstring is None and len(self.formatstrings) == 0:
            fstring = [""]*len(row)
        elif fstring is None and len(self.formatstrings) != 0:
            fstring = self.formatstrings[0]
        self.formatstrings.append(fstring)

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
        if len(self.separators) == 0:
            self.separators = ["|"]*len(self.header)
        
        seps = self.separators + ["|"]  #append empty string to sparators since the last one will get omitted anyways
        
        #initialize header with a rownumber column
        header_print = "{:^"+self.num_width+"}" + seps[0]
        header_print = header_print.format("#")
        #fill in the rest of the header and print it
        for fs, sep in zip(self.formatstrings[0], seps[1:]):
            fs_digit = re.findall(r"\d+", fs)[0]
            header_print += "{:^"+fs_digit+"s}"+ sep
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
        if len(self.rows) < len(self.formatstrings):
            diff = abs(len(self.rows) - len(self.formatstrings))
            to_add = [self.formatstrings[-1]]*diff
            for ta in to_add:
                self.formatstrings.insert(0, ta)
        row_num = 1 #keep track of the rownumber to enumerate table
        
        #print rows
        for row, fstring in zip(self.rows, self.formatstrings):

            seps = self.separators + ["|"]   #append empty string to sparators since the last one will get omitted anyways

            #initialize row with its rownumber
            row_print = "{:^"+self.num_width+"}" + seps[0]
            row_print = row_print.format(row_num)
            #add row-entries
            for fs, sep in zip(fstring, seps[1:]):
                row_print += "{:^"+fs[1:]+"}" + sep
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


