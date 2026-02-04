# code-PhD-marijn

Scripts for calculations, plots and data analysis of my research. 
These involve typical things in AMO labs like MOT imaging, and specifically analysis on atoms in optical tweezers.
Hopefully, the scripts can serve as a starting point if you want to do similar things. 

* Scripts have been tested in a conda environment (version 25.3.1) with python version 3.9.21.
* The working directory should be set to the root folder `code-PhD-marijn` for the imports to work. 
* Plots will be saved under `output` folder. 
* `Modules` and `utils` folders contain functions/classes that are imported in the individual `scripts`. To do this, run in root folder the following, to allow python to find the modules and utils content:

`pip install -e .` 
