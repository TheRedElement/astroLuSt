

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

