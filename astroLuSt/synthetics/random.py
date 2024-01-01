#%%imports
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
import random
from scipy.signal import sawtooth
from scipy.stats import norm
import string
from typing import Union, Tuple

from astroLuSt.preprocessing.dataseries_manipulation import periodize

#%%definitions
class GenUniqueStrings:
    """
        - class to generate unique strings from a given set of characters to choose from
        - roughly follows scipy.stats distribution conventions


        Attributes
        ----------
            - `n`
                - int, optional
                - lengths of the strings to generate (not counting `suffix` and `prefix`)
                - the default is 1
            - `char_choices`
                - int, list, optional
                - iterable providing the set of characters to choose from
                - the default is `None`
                    - will generate using uppercase letters and numbers
            - `prefix`
                - str, optional
                - a prefix to put in front of every generated string
                - the default is `None`
            - `suffix`
                - str, optional
                - a suffix to put at the end of every generated string
                - the default is `None`

        Methods
        -------
            - `rvs()`

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
            f'    n={repr(self.n)},\n'
            f'    char_choices={repr(self.char_choices)},\n'
            f'    prefix={repr(self.prefix)},\n'
            f'    suffix={repr(self.suffix)},\n'
            f')'
        )
    
    def rvs(self,
        shape:Union[int, tuple]=None,
        random_state:int=None,
        ):
        """
            - method similar to the `scipy.stats` `rvs()` method
            - rvs ... random variates
            - will generate an array of size `shape` containing randomly generated samples
            
            Parameters
            ----------
                - `shape`
                    - int, tuple optional
                    - number of samples to generate
                    - the default is `None`
                        - will generate a single sample
                - `random_state`
                    - int, optional
                    - seed of the random number generator
                    - provide any integer for reproducible results
                    - the default is `None`
                        - non-reproducible results

            Raises
            ------

            Returns
            -------
                - `output`
                    - np.ndarray, str
                    - array containing the generated strings
                    - if only one sample got generated and `shape` is of type int a single string will be returned
            
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
    """
        - class to generate a unique series of random periodic signals

        Attributes
        ----------
            - `npoints`
                - np.ndarray, int, optional
                - number of points per dataseries
                - if `None`
                    - will default to `shape[1]` of the parameter `shape` of `self.rvs()`
                - if int
                    - will use this many datapoints for all dataseries
                - the default is `None`
            - `periods`
                - np.ndarray, int, float, optional
                - periods of the individual generated dataseries
                    - will generate as many unique periodized dataseries as elements in `periods`
                - if `None`
                    - will default to `shape[0]` of the parameter `shape` of `self.rvs()`
                - if int or float
                    - will use this period for all dataseries
                - the default is `None`
            - `amplitudes`
                - TODO                
            - `x_offsets`
                - np.ndarray, int, optional
                - has to be of same length as `periods`
                - offsets to add to the x-values of the generated periodic dataseries
                - if an np.ndarray
                    - will be interpreted as the offset per generated dataseries
                - if an int
                    - will use that offset for all generated dataseries
                - the default is `None`
            - `choices`
                - np.ndarray, int, optional
                - has to be of dtype object
                - contains either callables (functions) or np.ndarrays
                    - callables will be evaluated on the parameter `x` of `self.rvs()`
                    - np.ndarrays will be returned and periodized as they are
                - these choices will be used to randomly generate dataseries
                - the default is `None`
                    - will use an array containing the following methods
                        - `sine()`
                        - `cosin()`
                        - `tangent()`
                        - `polynomial()`
                        - `gaussian()`
                        - `random()`

        Methods
        -------
            - `sine()`
            - `cosin()`
            - `tangent()`
            - `polynomial()`
            - `gaussian()`
            - `random()`
            - `select_choice()`
            - `generate_one()`
            - `rvs()`
            - `plot_result()`

        Dependencies
        ------------
            - matplotlib
            - numpy
            - scipy
            - typing
    """

    def __init__(self,
        npoints:Union[np.ndarray,int]=None,
        periods:Union[np.ndarray,int,float]=None,
        amplitudes:Union[np.ndarray,int,float]=None,
        x_offsets:Union[np.ndarray,int]=None,
        choices:np.ndarray=None,
        verbose:int=0,
    ) -> None:
      
        if npoints is None:                     self.npoints    = 0
        else:                                   self.npoints    = npoints
        if periods is None:                     self.periods    = 1
        else:                                   self.periods    = periods
        if amplitudes is None:                  self.amplitudes = 1
        else:                                   self.amplitudes = amplitudes
        if x_offsets is None:                   self.x_offsets = 0
        else:                                   self.x_offsets = x_offsets
        self.x_offsets                          = x_offsets
        self.verbose                            = verbose
        

        #initialize choices
        self.passed_choices = choices
        if choices is None:
            self.choices = np.array([
                self.sine,
                self.cosine,
                self.tangent,
                self.sawtooth,
                self.polynomial,
                self.gaussian,
                self.random,
            ], dtype=object)
        

        pass

    def __repr__(self) -> str:
        choices2print = np.array([c.__name__ for c in self.choices])
        return (
            f'GeneratePeriodicSignals(\n'
            f'    npoints={repr(self.npoints)},\n'
            f'    periods={repr(self.periods)},\n'
            f'    x_offsets={repr(self.x_offsets)},\n'
            # f'    choices={choices2print},\n'
            f'    choices={repr(self.passed_choices)},\n'
            f')'
        )

    def sine(self,
        x:np.ndarray,
        **kwargs,
        ) -> np.ndarray:
        """
            - method to calculate a sine function in phase space

            Parameters
            ----------
                - `x`
                    - np.ndarray
                    - input values to evaluate on
                    - phases for the purpose of this class
                - `**kwargs`
                    - kwargs to pass to the function
                    - used for consistency across methods
            
            Raises
            ------

            Returns
            -------
                - `sin`
                    - np.ndarray
                    - evaluation of `x`

            Comments
            --------

        """
        sin = np.sin(x*2*np.pi)
        return sin
    
    def cosine(self,
        x:np.ndarray,
        **kwargs,
        ) -> np.ndarray:
        """
            - method to calculate a cosine function in phase space

            Parameters
            ----------
                - `x`
                    - np.ndarray
                    - input values to evaluate on
                    - phases for the purpose of this class
                - `**kwargs`
                    - kwargs to pass to the function
                    - used for consistency across methods
            
            Raises
            ------

            Returns
            -------
                - `cos`
                    - np.ndarray
                    - evaluation of `x`

            Comments
            --------

        """
        cos = np.cos(x*2*np.pi)
        return cos
    
    def tangent(self,
        x:np.ndarray,
        **kwargs,
        ) -> np.ndarray:
        """
            - method to calculate a tangent function in phase space

            Parameters
            ----------
                - `x`
                    - np.ndarray
                    - input values to evaluate on
                    - phases for the purpose of this class
                - `**kwargs`
                    - kwargs to pass to the function
                    - used for consistency across methods
            
            Raises
            ------

            Returns
            -------
                - `tan`
                    - np.ndarray
                    - evaluation of `x`

            Comments
            --------

        """        
        tan = np.tan(x*2*np.pi)
        return tan 

    def sawtooth(self,
        x:np.ndarray,
        **kwargs,
        ) -> np.ndarray:
        """
            - method to calculate a sawtooth function in phase space

            Parameters
            ----------
                - `x`
                    - np.ndarray
                    - input values to evaluate on
                    - phases for the purpose of this class
                - `**kwargs`
                    - kwargs to pass to the function
                    - used for consistency across methods
            
            Raises
            ------

            Returns
            -------
                - `st`
                    - np.ndarray
                    - evaluation of `x`

            Comments
            --------

        """            
        st = sawtooth(x*2*np.pi)
        return st

    def polynomial(self,
        x:np.ndarray,
        **kwargs,
        ) -> np.ndarray:
        """
            - method to evaluate a polynomial function

            Parameters
            ----------
                - `x`
                    - np.ndarray
                    - input values to evaluate on
                - `**kwargs`
                    - kwargs to pass to the function
                        - will use `kwargs['p']`
                            - polynomial coefficients
                    - used for consistency across methods
            
            Raises
            ------

            Returns
            -------
                - `poly`
                    - np.ndarray
                    - evaluation of `x`

            Comments
            --------

        """
        poly = np.polyval(x=x, p=kwargs['p'])
        return poly

    def gaussian(self,
        x:np.ndarray,
        **kwargs
        ) -> np.ndarray:
        """
            - method to evaluate a gaussian function

            Parameters
            ----------
                - `x`
                    - np.ndarray
                    - input values to evaluate on
                - `**kwargs`
                    - kwargs to pass to the function
                        - will use `kwargs['loc']`
                            - mean of the gaussian
                        - will use `kwargs['scale']`
                            - standard deviation of the gaussian
                        - will use `kwargs['amp']`
                            - amplitude of the gaussian
                            - i.e. mixing coefficient
                    - used for consistency across methods
            
            Raises
            ------

            Returns
            -------
                - `gauss`
                    - np.ndarray
                    - evaluation of `x`

            Comments
            --------

        """        
        gauss = norm.pdf(x, loc=kwargs['loc'], scale=kwargs['scale']) * kwargs['amp']
        return gauss

    def random(self,
        x:np.ndarray,
        **kwargs
        ) -> np.ndarray:
        """
            - method to generate a random array of the same shape as `x`

            Parameters
            ----------
                - `x`
                    - np.ndarray
                    - input values to evaluate on
                - `**kwargs`
                    - kwargs to pass to the function
                        - will use `kwargs['amp']`
                            - amplitude of the noise
                    - used for consistency across methods
            
            Raises
            ------

            Returns
            -------
                - `randarray`
                    - np.ndarray
                    - evaluation of `x`

            Comments
            --------

        """        
        randarray = np.random.randn(x.shape[0])*kwargs['amp']
        return randarray

    def select_choice(self,
        choices:np.ndarray, x:np.ndarray,
        func_kwargs:dict=None
        ) -> np.ndarray:
        """
            - method to randomly select from `choices` and evaluate on `x` or return directly
                - will evaluate if a callable is chosen
                - will return directly if an array is chosen

            Parameters
            ----------
                - `choices`
                    - np.ndarray
                    - has to be of dtype object
                    - available options for the generation of the base-signal
                        -  base-signal will be generated
                            - by evaluating on `x`
                                - i.e. by calling `choice(x, **kwargs)`
                            - by returning choice directly
                                - if `choice` happens to be a np.ndarray
                    - this signal will then be periodized
                - `x`
                    - np.ndarray
                    - input values to evaluate `choice` on
                - `func_kwargs`
                    - dict, optional
                    - function kwargs passed to choice
                        - i.e. `chioce(x, **kwargs)` will be called
                    - the default is `None`
                        - will be initilized with following dict
                            
                            ```python
                            >>> func_kwargs = {
                            >>>     'p':[np.random.randint(1,5)],
                            >>>     'amp':np.random.uniform(0.1,5), 'loc':np.random.uniform(-1,1), 'scale':np.random.randn(),
                            >>> }
                            ```

                        - `p` will be used by `self.polynomial()`
                        - `amp` will be used by `self.normal()` and `self.random()`
                        - `loc` will be used by `self.normal()`
                        - `scale` will be used by `self.normal()`
            
            Raises
            ------

            Returns
            -------
                - `choice`
                    - np.ndarray
                    - the random choice evaluated on `x`
                        if choice is a callable
                        - i.e. `choice(x, **kwargs)`
                    - the choice directly
                        - if choice is a np.ndarray

            Comments
            --------
        """

        if func_kwargs is None:
            func_kwargs = {
                'p':[np.random.randint(1,5)],
                'amp':np.random.uniform(0.1,5), 'loc':np.random.uniform(-1,1), 'scale':np.random.uniform(0.1,1),
            }
        if 'p' not in func_kwargs:      func_kwargs['p'] = [np.random.randint(1,5)]
        if 'amp' not in func_kwargs:    func_kwargs['amp'] = np.random.uniform(0.1,5)
        if 'loc' not in func_kwargs:    func_kwargs['loc'] = np.random.uniform(-1,1)
        if 'scale' not in func_kwargs:  func_kwargs['scale'] = np.random.uniform(0.1,1)

        choice = np.random.choice(choices, size=None)

        if callable(choice):
            choice = choice(x, **func_kwargs)

        return choice

    def generate_one(self,
        y:np.ndarray,
        npoints:int,
        period:float,
        ) -> Tuple[np.ndarray,np.ndarray]:
        """
            - method to generate one random periodic signal of lenght npoints with period 'period'

            Parameters
            ----------
                - `y`
                    - np.ndarray
                    - y-values of the signal to periodize
                - `npoints`
                    - int
                    - number of datapoints the periodic data-series shall have
                - `period`
                    - float
                    - period the generated periodic signal shall have
            
            Raises
            ------

            Returns
            -------
                - `xp`
                    - np.ndarray
                    - x-values of the periodic data-series
                - `yp`
                    - np.ndarray
                    - y-values of the periodic data-series

            Comments
            -------- 
        """

        #calculate how often the signal has to repeat to get the desired amount of datapoints
        repetitions = npoints/y.shape[0]

        #periodize the input signal
        xp, yp = periodize(y=y, period=period, repetitions=repetitions, testplot=False)

        return xp, yp

    def rvs(self,
        shape:tuple=None,
        choices:np.ndarray=None,
        x_min:np.ndarray=None, x_max:np.ndarray=None,
        noise_level_y:float=0.1,
        noise_level_x:float=0.1,
        random_state:int=None,
        verbose:int=None,
        func_kwargs:dict=None
        # ) -> Tuple[np.ndarray[np.ndarray], np.ndarray[np.ndarray]]: #works for python >= 3.9
        ) -> Tuple[np.ndarray, np.ndarray]:

        #default parameters
        if shape is None:       shape       = (1,100)
        if choices is None:     choices     = self.choices
        if x_min is None:       x_min       = np.zeros(shape[0])
        if x_max is None:       x_max       = np.ones(shape[0])
        if verbose is None:     verbose     = self.verbose

        #from attributes
        if isinstance(self.periods, (int,float)):   periods     = np.array([[self.periods]]   *shape[0])
        else:                                       periods     = self.periods
        if isinstance(self.amplitudes, (int,float)):amplitudes  = np.array([[self.amplitudes]]*shape[0])
        else:                                       amplitudes  = self.amplitudes
        if isinstance(self.x_offsets, (int,float)): x_offsets   = np.array([[self.x_offsets]]*shape[0])
        else:                                       x_offsets   = self.x_offsets
        if isinstance(self.npoints, int):           npoints     = np.array([self.npoints]     *shape[0])
        else:
            npoints = self.npoints
            if len(np.unique(self.npoints)) > 1:
                shape[1] = None
            else:
                shape[1] = self.npoints[0]

        #update `shape` accordingly
        shape = (len(periods), shape[1])

        #check all shapes
        if (len(periods) != shape[0]) \
            or (len(amplitudes) != shape[0]) \
            or (len(x_offsets) != shape[0]) \
            or (len(x_min) != shape[0]) \
            or (len(x_max) != shape[0]):
            raise ValueError((
                f'`shape[0]` has to be the same as'
                f' `len(periods)`, `len(amplitudes)`, `len(x_offsets)`, `len(x_min)`, `len(x_max)`.'
                f' The respective values are:'
                f' {shape[0]=}, {len(periods)=}, {len(amplitudes)=}, {len(x_offsets)=}, {len(x_min)=}, {len(x_max)=}!'
            ))

        #initialize output lists
        x_gen = []
        y_gen = []

        ##individual samples
        for xn, xx, n, p, a, xo in zip(x_min, x_max, npoints, periods, amplitudes, x_offsets):

            #init individual sample
            x = np.linspace(xn, xx, n)
            x += np.random.randn(x.shape[0]) * noise_level_x
            y =  np.random.randn(x.shape[0]) * noise_level_y    #init with noise

            #composite parts with different periods and amplitudes
            for pi, ai, xoi in zip(p, a, xo):
                
                #convert to phases for superposition
                phases = (x-xoi)/pi
                #choose function/array to generate from
                yi = self.select_choice(choices=choices, x=phases, func_kwargs=func_kwargs)
                y += ai*yi

            #append to output
            x_gen += [x]
            y_gen += [y]


        return x_gen, y_gen, periods

    def rvs_(self,
        shape:tuple=None,
        choices:np.ndarray=None,
        x:np.ndarray=None,
        noise_level_y:float=0.1,
        noise_level_x:float=0.1,
        random_state:int=None,
        func_kwargs:dict=None 
        # ) -> Tuple[np.ndarray[np.ndarray], np.ndarray[np.ndarray]]: #works for python >= 3.9
        ) -> Tuple[np.ndarray, np.ndarray]:
        """
            - method similar to the `scipy.stats` `rvs()` method
            - rvs ... random variates
            - will generate an array of the same length as `self.periods` containing randomly generated samples
                - each generated dataseries can have a different length encoded in `self.npoints`
            
            Parameters
            ----------
                - `shape`
                    - int, tuple, optional
                    - shape of the generated list of arrays
                    - only used if one of `self.periods` and `self.npoints` are `None`
                        - in the case that `self.periods` is `None` will
                            - generate `shape[0]` dataseries with period of 1
                        - in the case that `self.npoints` is `None` will
                            - generate all timeseries with `shape[1]` points
                    - the default is `None`
                        - will use `self.periods` and `self.npoints` to infer the shapes
                        - if `self.period` and `self.npoints` is None as well
                            - will default to `(1,10)`
                                - `periods = 1`
                                - `npoints = 10`
                                - i.e. 1 sample with 10 datapoints
                - `choices`
                    - np.ndarray, optional
                    - has to be of dtype object
                    - contains either callables (functions) or np.ndarrays
                        - callables will be evaluated on `x`
                        - np.ndarrays will be returned and periodized as they are
                    - these choices will be used to randomly generate dataseries
                    - the default is `None`
                        - will fallback to `self.choices`
                - `x`
                    - np.ndarray, optional
                    - x-values of the timeseries in phase-space
                    - i.e. before generating the periodic timeseries the chosen choice out of choices will be evaluated on `x`
                        - `choices(x)` will be called
                    - the default is `None`
                        - will use `np.linspace(0,1,10,endpoint=False)`
                - `noise_level_x`
                    - float, optional
                    - scale of the noise added to the periodized signal in x-direction
                    - the default is 0.1
                - `noise_level_y`
                    - float, optional
                    - scale of the noise added to the periodized signal in y-direction
                    - the default is 0.1
                - `random_state`
                    - int, optional
                    - seed of the random number generator
                    - provide any integer for reproducible results
                    - the default is `None`
                        - non-reproducible results
                - `func_kwargs`
                    - dict, optional
                    - kwargs passed to `self.select_choice()`
                    - contains kwargs for all choices in choices
                    - the default is `None`
                        - will autogenerate kwargs needed to use the default `self.choices`

            Raises
            ------

            Returns
            -------
                - `x_gen`
                    - np.ndarray
                        - contains np.ndarrays
                            - can have different lengths
                    - each entry contains the x-values of one generated periodic signal (entry in `y_gen`)
                - `y_gen`
                    - np.ndarray
                        - contains np.ndarrays
                            - can have different lengths
                    - each entry contains the y-values of one generated periodic signal
                        - the entry of `x_gen` contains the corresponding x-values
            
            Comments
            --------
                - make sure that whichever callable you pass within choices can accept `**kwargs`
        """

        #initilaize

        ##shape if nothing is passed to __init__()
        if shape is None: shape = (1,10)


        if choices is None:
            choices = self.choices
        
        ##periods adopted to shape
        if self.periods is None:
            periods = np.ones(shape[0])
        ##same period accross samples
        elif isinstance(self.periods, (int, float)):
            periods = np.zeros(shape[0]) + self.periods
        ##varying period accross samples
        else:
            periods = self.periods
            shape = (periods.shape[0],shape[1])
        
        ##amplitudes adopted to shape
        if self.amplitudes is None:
            amplitudes = np.ones(shape[0])
        ##same period accross samples
        elif isinstance(self.amplitudes, (int, float)):
            amplitudes = np.zeros(shape[0]) + self.amplitudes
        ##varying period accross samples
        else:
            amplitudes = self.amplitudes

        ##npoints adopted to shape
        if self.npoints is None:
            npoints = np.zeros(shape[0]) + shape[1]
        ##same number of points accross samples
        elif isinstance(self.npoints, int):
            npoints = np.zeros(shape[0]) + self.npoints
            shape = (shape[0],self.npoints)
        ##varying number of points accross samples
        else:
            npoints = self.npoints
            if len(np.unique(npoints)) > 1: shape = (shape[0],None)         #shape not inferable (variable npoints per dataseries)
            else:                           shape = (shape[0],npoints[0])   #shape inferable (npoints the same accross dataseries)
        
        ##no offset
        if self.x_offsets is None:
            x_offsets = np.zeros(shape[0])
        ##same offset accross samples
        elif isinstance(self.x_offsets, (int, float)):
            x_offsets = np.zeros(shape[0]) + self.x_offsets
        ##varying offset accross samples
        else:
            x_offsets = self.x_offsets

        ##default x-values (phases) to generate on
        if x is None:
            x = np.linspace(0,1,10,endpoint=False)

        #reshape offset to work with inhomogeneous arrays
        if shape[1] is None:
            x_offsets = x_offsets.flatten()
        else:
            x_offsets = x_offsets.reshape(-1,1)

        #initialize output lists/arrays
        x_gen = []
        y_gen = []

        #generate random periodic signals
        for n, p, a in zip(npoints, periods, amplitudes):
            
            #generate individual parts of the signal (will be superpositioned)
            ##init output
            yp = np.zeros((1,int(n)))
            for pi, ai in zip(p,a):
                #choose function/array to generate from
                y = self.select_choice(choices=choices, x=x, func_kwargs=func_kwargs)

                #generate one periodized signal
                xp, ypi = self.generate_one(y=y, npoints=n, period=pi)
                yp += ai*ypi #scale by amplitude
            #add noise
            yp += np.random.randn(*yp.shape)*noise_level_y
            xp += np.random.randn(*xp.shape)*noise_level_x
            
            #append to output
            x_gen.append(xp.flatten())
            y_gen.append(yp.flatten())

        #add offset, transform to array
        x_gen = np.array(x_gen, dtype=object) + x_offsets
        y_gen = np.array(y_gen, dtype=object)
        self.x_gen = x_gen
        self.y_gen = y_gen
        
        return x_gen, y_gen

    def plot_result(self,
        x_gen:np.ndarray, y_gen:np.ndarray,
        periods:np.ndarray=None,
        fig_kwargs:dict=None, plot_kwargs:dict=None,
        ) -> Tuple[Figure,plt.Axes]:
        """
            - method to create a testplot visualizing the generated dataseries

            Parameters
            ----------
                - `x_gen`
                    - np.ndarray
                    - TODO
                - `y_gen`
                    - np.ndarray
                    - TODO
                - `p_gen`
                    - np.ndarray
                    - TODO
                - `fig_kwargs`
                    - dict, optional
                    - kwargs to pass to `plt.fig()`
                    - the default is `None`
                        - will initialize with empty dict (`fig_kwargs = {}`)
                - `plot_kwargs`
                    - dict, optional
                    - kwargs to pass to `ax.plot()`
                    - the default is `None`
                        - will initialize with `{'marker':'o'}`

            Raises
            ------

            Returns
            -------
                - `fig`
                    - matplotlib Figure
                    - created matplotlib figure
                - `axs`
                    - `plt.Axs`
                    - axes corresponding to `fig`

            Comments
            --------
        """

        if fig_kwargs is None:
            fig_kwargs = {}
        if plot_kwargs is None:
            plot_kwargs = {'marker':'o'}

        fig = plt.figure(**fig_kwargs)
        ax1 = fig.add_subplot(111)
        for xi, yi in zip(x_gen, y_gen):
            ax1.plot(xi, yi, **plot_kwargs)
        
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')

        axs = fig.axes   

        return fig, axs

