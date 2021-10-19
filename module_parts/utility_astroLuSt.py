

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
