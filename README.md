# astroLuSt

A collection of functions useful for dataanlysis especially in astronomy.

## files

The different parts of the module are saved in module_parts.
The current version consists of the following parts:
    - data_astroLuSt.py
        --> classes and functions useful for data-processing
    - PHOEBE_astroLuSt.py
        --> classes and functions useful for working with PHOEBE
    - plotting_astroLuSt.py
        --> classes and functions useful for plotting
    - utility_astroLuSt.py
        --> classes and functions for random convenient stuff

## Data_LuSt

This class is designed to provide useful functions especially for time series analysis.
It contains the following methods:
    - linspace_def
        --> function to create an array of datapoints spaced 
            with some defined distribution
            ~~> the distribution is a superposition of 
                gaussians
            ~~> the gaussians are defined by the user through 
                providing the respective centers and widths
            ~~> the datapoints are returned for a range
                defined by the user
        --> has the option to prived a testplot for
            visualizing the created distribution
    - lc_error
        --> function to estimate the errors of a lightcurve 
            given a time-difference condition
        --> useful for e.g. space photoetry data
        --> works by basically clustering the dataseries
    - pdm
        --> Not implemented yet, but on the TODO-list
    - fold
        --> function to fold a timeseries on a provided period
        --> returns the timeseries in phase space
            ~~> from -0.5 to 0.5
        --> can in theory also be used to convert an array of
            times to its phase equivalent
    - periodic_shift
        --> function to shift a dataseries by shift
            considering periodic boundaries
    - phase2time
        --> function to convert a given array of phases into
        its corresponding time equivalent
    - phase_binning
        --> function to execute binning in phase on some 
            dataseries
        --> allows for definition of high-resolution areas by
            using linspace_def
    - sigma_clipping
        --> function to execute sigma-clipping on some  
            dataseries
        --> one needs to provide the intervals to consider and
            the reference_phases

## Plot_LuSt

This class provides an efficient way of creating multipanel plots, as well as some other functions useful for plotting.
It contains the following methods:
    - plot_ax
        --> a utility function for easily creating
            multipanel-plots
        --> makes use of matplotlib
        --> returns the figure as well as axes object to allow
            further manipulations outside of the function
    - hexcolor_extract
        --> a function that creates a dictionary of hex-codes
            for colors in more or less spectral order
        --> especially useful, for not choosing the same color
            twice in one plot
        --> depends on ./files/colorcodes.txt
    - color_generator
        --> a function to create a defined number of as
            different colors as possible
        --> Not implemented yet, but on the TODO-list

## Time_stuff

Class to time the execution of tasks
It contains the following methods:
    - start_task
        --> start the timer for the task
    - end_task
        --> end the timer for the task
        --> print the result

## Table_LuSt
        
Class to create table in the bash from a given list of rows and a header.
Has some additional methods for additional functionality:
    - add_row
        --> adds a row to the existing Table_LuSt-object
    - add_header_row
        --> Not implemented yet, but on the TODO-list
    - print_header
        --> prints out the current header of the table
    - print_rows
        --> prints out the current body (rows) of the table
    - print_table
        --> prints the table in the bash
        --> also allows for saving the output into an external
            file
    - latex_template
        --> creates a template that will reproduce the
            printed table once transferred to a .tex-file
        --> also allows for saving the output into an external
            file
