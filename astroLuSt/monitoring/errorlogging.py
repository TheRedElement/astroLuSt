
#%%imports
import traceback
import re
import pandas as pd

#%%definitions
class LogErrors:
    """
        - class to log errors that occured and got caught via try - except statements

        Attributes
        ----------
            - verbose
                - int, optional
                - verbosity level
                - the default is 0

        Infered Attributes
        ------------------
            - df_errorlog
                - pd.DataFrame
                - dataframe to log all the caught errors
        
        Methods
        -------
            - print_exc()
            - save_log()
        
        Dependencies
        ------------
            - pandas
            - re
            - traceback
            
        Comments
        --------
    """

    def __init__(self,
        verbose:int=1,
        ) -> None:
        self.verbose = verbose

        self.df_errorlog = pd.DataFrame(
            columns=['exception', 'prefix', 'suffix', 'file', 'line', 'problem line', 'error msg', 'time']
        )
        pass

    def __repr__(self) -> str:
        
        return (
            f'ExecptionFormatting()'
        )
    
    def log_exc(self,
        e:Exception,
        prefix:str='', suffix:str='',
        verbose:int=None,
        ) -> None:
        """
            - method to log a formatted version of the caught exception e
            - will print the formatted version if requested

            Parameters
            ----------
                - e
                    - Exception
                    - exception that got caught via try - except
                - prefix
                    - str, optional
                    - something to print before the caught exception
                    - the default is ''
                - suffix
                    - str, optional
                    - something to print after the caught exception
                    - the default is ''
                - verbose
                    - int, optional
                    - verbosity level
                    - will override self.verbose if passed
                    - the default is None
                        - falls back to self.verbose
            
            Raises
            ------
            
            Returns
            -------

            Comments
            --------
        """
        
        #initialize
        if verbose is None:
            verbose = self.verbose

        format_exc = traceback.format_exc()

        if verbose > 0:
            print(prefix)
            print(format_exc)
            print(suffix)


        return
    
    def exc2df(self,
        e:Exception,
        prefix:str='', suffix:str='',
        ) -> None:
        """
            - method to store a caught exception e to a pandas DataFrame

            Parameters
            ----------
                - e
                    - Exception
                    - exception that got caught via try - except
                - prefix
                    - str, optional
                    - something to print before the caught exception
                    - the default is ''
                - suffix
                    - str, optional
                    - something to print after the caught exception
                    - the default is ''
            
            Raises
            ------
            
            Returns
            -------

            Comments
            --------
        """        
        format_exc = traceback.format_exc()

        files = re.findall(r'(?<=File ").+(?=")', format_exc)        
        lines = re.findall(r'(?<=line )\d+(?=,)', format_exc)
        problem_lines = re.findall(r'(?<=<module>\n).+', format_exc)
        error_msgs = re.findall(r'\w+Error: [\w ]+', format_exc)

        df_temp = pd.DataFrame({
            'exception':format_exc*len(files),
            'prefix':prefix*len(files),
            'suffix':suffix*len(files),
            'file':files,
            'line':lines,
            'problem line':problem_lines,
            'error msg':error_msgs*len(files),
            'time':[pd.Timestamp.now()]*len(files)
        })        

        self.df_errorlog = pd.concat([self.df_errorlog, df_temp])

        return

    def save_log(self,
        filename:str,
        **kwargs
        ) -> None:
        """
            - method to save the current errorlog dataframe to a csv file

            Parameters
            ----------
                - filename
                    - str, optional
                    - path to the file to save the errorlog to
                - **kwargs
                    - kwargs passed to pd.DataFrame.to_csv()
            
            Raises
            ------
            
            Returns
            -------
            
            Comments
            --------
        """

        self.df_errorlog.to_csv(filename, **kwargs)

        return

