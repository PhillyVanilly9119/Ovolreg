# Ovolreg - Optical Coherence Tomography volume registration library

Simple Python library to register consecutively acquired OCT volumes for averaging and enhancing contrast

## Purpose
The project currently works as a python scripting environment, without a graphical userinterface - basically ony

## Getting started
To get started install anaconda https://www.anaconda.com/products/distribution and clone this repository onto your local machine.
1. for Windows/Linux: navigate to the directory into which you've cloned this repository: 
~\ cd <full_path>
2. create the virtual anaconda env: 


## Core functions
- Currently the registration of volume data works with n consecutively acquired volumes on a slice- i.e. B-scan-basis.
- 3D registrations, based on the SimpleITK library (see webpage: https://simpleitk.org/ and their repository https://github.com/InsightSoftwareConsortium/SimpleITK, as well as Jupyter Notebook Tutorials (for the python wrapper/API): https://github.com/InsightSoftwareConsortium/SimpleITK-Notebooks/tree/master/Python) and are planned for the future.
