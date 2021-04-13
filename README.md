[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/guiwitz/microfilm/master?urlpath=lab)
# Microfilm

This package is a collection of tools to handle time-lapse microscopy datasets. The ```dataset``` submodule in particular allows to import series of tiff files, multi-page tiffs, h5 and Nikon ND2 files and to handle them through a common set of functions. The ```microplot``` modules allows to easily create figures with 2d plots of multi-channel images for which color maps (LUTs) can be easily selected.

## Installation

You can install this package directly from Github using: 

```
pip install git+https://github.com/guiwitz/microfilm.git@master#egg=microfilm
```

To test the package via the Jupyter interface and the notebooks available [here](notebooks) you can create a conda environment using the [environment.yml](binder/environment.yml) file:

```
conda env create -f environment.yml
```