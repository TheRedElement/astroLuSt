#%%imports
import matplotlib.pyplot as plt
import numpy as np
import random
import string
from typing import Union, Callable

from astroLuSt.preprocessing.dataseries_manipulation import periodize

#%%definitions
class GenUniqueStrings:
    """
        - class to generate unique strings from a given set of characters to choose from
        - roughly follows scipy.stats distribution conventions


        Attributes
        ----------
            - n
                - int, optional
                - lengths of the strings to generate (not counting 'suffix' and 'prefix')
                - the default is 1
            - char_choices
                - int, list, optional
                - iterable providing the set of characters to choose from
                - the default is None
                    - will generate using uppercase letters and numbers
            - prefix
                - str, optional
                - a prefix to put in front of every generated string
                - the default is None
            - suffix
                - str, optional
                - a suffix to put at the end of every generated string
                - the default is None

        Methods
        -------
            - rvs()

        Dependencies
        ------------
            - numpy
            - random
            - string
            - typing

        Comments
        --------

    """
    def __init__(self,
        n:int=1,
        char_choices:Union[list,str]=None,
        prefix:str=None,
        suffix:str=None,
        ) -> None:

        self.n = n
        if char_choices is None: self.char_choices = string.ascii_uppercase+string.digits
        else:                    self.char_choices = char_choices
        if prefix is None: self.prefix = ''
        else:              self.prefix = prefix
        if suffix is None: self.suffix = ''
        else:              self.suffix = suffix

        return
    
    def __repr__(self) -> str:

        return (
            f'GenUniqueStrings(\n'
            f'    n={self.n},\n'
            f'    char_choices={self.char_choices},\n'
            f'    prefix={self.prefix},\n'
            f'    suffix={self.suffix},\n'
            f')'
        )
    
    def rvs(self,
        shape:Union[int, tuple]=None,
        random_state:int=None,
        ):
        """
            - method similar to the scipy.stats rvs() method
            - rvs ... random variates
            - will generate an array of size 'size' containing randomly generated samples
            
            Parameters
            ----------
                - shape
                    - int, tuple optional
                    - number of samples to generate
                    - the default is None
                        - will generate a single sample
                - random_state
                    - int, optional
                    - seed of the random number generator
                    - provide any integer for reproducible results
                    - the default is None
                        - non-reproducible results

            Raises
            ------

            Returns
            -------
                - output
                    - np.ndarray, str
                    - array containing the generated strings
                    - if only one sample got generated and 'shape' is of type int a single string will be returned
            
            Comments
            --------
        """

        #new instance of RNG
        randstate = random.SystemRandom(random_state)

        if shape is None:
            shape = 1
        # else:
        n_samples = int(np.prod(shape))

        output = np.array([])
        for idx in range(n_samples):
            uid = ''.join(randstate.choices(self.char_choices, k=self.n))
            output = np.append(output, self.prefix + uid + self.suffix)

        output = output.reshape(shape)

        #if a single sample got generated, and an integer was passed just return the string
        if n_samples == 1 and isinstance(shape, int):
            return output[0]
        #otherwise return an np.ndarray
        else:
            return output
    

class GeneratePeriodicSignals:

    def __init__(self,
        npoints:np.ndarray=None,
        periods:np.ndarray=None
    ) -> None:
      
        self.npoints = npoints
        self.periods = periods

        pass

    def __repr__(self) -> str:
        return (
            f'GeneratePeriodicSignals(\n'
            f'    npoints={repr(self.npoints)},\n'
            f'    periods={repr(self.periods)},\n'
            f')'
        )


    def compositesin(self,
      phase:np.ndarray, periods, amplitudes
      ):

      y = 0
      for n in range(periods.shape[1]):
        shift = np.random.randint(0, self.npoints)
        y_ = amplitudes[:,n].reshape(-1,1)*np.roll(np.sin(phase*2*np.pi*periods[:,n].reshape(-1,1)), shift=shift)
        y += y_

      return y

    def generate(self,
      func:Callable=None,
      noise:float=0.05,
      ):

      if func is None:
        # func = self.simplesin
        func = self.compositesin

      periods = np.random.rand(self.nsamples,2)*4
      amplitudes = np.random.rand(self.nsamples,2)*1

      x_out = np.array([np.linspace(0, 1, self.npoints) for i in range(self.nsamples)])
      y_out = func(x_out, periods, amplitudes)
      y_out += np.random.randn(self.npoints)*noise #add noise

      return x_out, y_out, periods

    def generate_one(self,
        npoints:int,
        period:float,
        ):
        xy = []

        x = np.linspace(0, 1, 10)
        y = x.copy()**2

        repetitions = npoints//x.shape[0]
        # print(repetitions)

        xp, yp = periodize(y=y, period=period, repetitions=repetitions, x=x, testplot=False)

        # print(xp, yp)

        # xy = np.array([xp, yp]).T

        return xp, yp

    def rvs(self,
        shape:tuple=None,
        random_state:int=None,    
        ):

        #initilaize shape if not passed
        if shape is None: shape = (1,10)
        
        
        if self.periods is None:
            periods = np.ones(shape[0])
        elif isinstance(self.periods, (int, float)):
            periods = np.zeros(shape[0]) + self.periods
        else:
            periods = self.periods
        if self.npoints is None:
            npoints = np.zeros(shape[0]) + shape[1]
        elif isinstance(self.npoints, int):
            npoints = np.zeros(shape[1]) + self.npoints
        else:
            npoints = self.npoints

        print(periods, npoints)

        x_out = []
        y_out = []
        for n, p in zip(npoints, periods):
            xp, yp = self.generate_one(npoints=n, period=p)
            x_out.append(xp)
            y_out.append(yp)
        
        fig = plt.figure()
        for xi, yi in zip(x_out, y_out):
            print(xi.shape, yi.shape)
            plt.scatter(xi, yi)
        plt.show()
        return
