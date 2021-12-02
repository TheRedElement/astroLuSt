# astroLuSt

A collection of functions useful for dataprocessing especially in astronomy.

## Installation

To install astroLuSt simply call
```shell
python -m pip install git+https://github.com/TheRedElement/astroLuSt
```
from your console.

## Files

The different parts of the module are saved in *./astroLuSt*.
The current version consists of the following parts:
- __data_astroLuSt.py__
    - classes and functions useful for data-processing
- __PHOEBE_astroLuSt.py__
    - classes and functions useful for working with PHOEBE
- __plotting_astroLuSt.py__
    - classes and functions useful for plotting
- __utility_astroLuSt.py__
    - classes and functions for random convenient stuff

The structure of the module is the following:

```
astroLuSt
|-- files
|   `-- colorcodes
|-- astroLuSt
|   |-- __init__.py
|   |-- data_astroLuSt.py
|   |-- PHOEBE_astroLuSt.py
|   |-- plotting_astroLuSt.py
|   `-- utility_astroLuSt.py
|-- astroLuSt_example_usage.ipynb
|-- README.md
`-- setup.py
```

## Dependencies

The current dependencies are
- numpy
- matplotlib
- scipy
- re
- datetime
- copy

## Examples

To get a feel on how the different functions and classes behave I provided some information in __astroLuSt_example_usage.ipynb__.
