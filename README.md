[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/guiwitz/microfilm/master?urlpath=lab)
# Microfilm

This package is a collection of tools to handle 2D time-lapse microscopy images. Its focus is primarily on plotting such datasets. In particular it makes it easy to represents multi-channel datasets in *composite* mode where the color maps of multiple channels are combined into a single image. It also allows to easily complete such figures with standard annotations like labels and scale bars. In addition, these figures can be turned into animations if time-lapse data are provided. Animations are either interactive when run in a Jupyter notebook, or save in standard movie formats (mp4, gif etc.)

Following the model of packages like [seaborn](https://seaborn.pydata.org/index.html), ```microfilm``` offers tight integration with Matplotlib. Complete access is given to the structures like axis and figures underlying the ```microfilm``` objects, allowing thus for the creation of arbitrarily complex plots for users familiar with Matplotlib.

## Installation

You can install this package directly from Github using: 

```
pip install git+https://github.com/guiwitz/microfilm.git@master#egg=microfilm
```

To test the package via the Jupyter interface and the notebooks available [here](notebooks) you can create a conda environment using the [environment.yml](binder/environment.yml) file:

```
conda env create -f environment.yml
```

## Simple example

It is very easy to create a ready-to-use plot of a multi-channel image dataset. In the following code snippet, we load a Numpy array, reshape it (input should by CTXY) and generate the figure below in a single line:

```python
import numpy as np
import skimage.io
from microfilm.microplot import microshow

image = skimage.io.imread('../demodata/coli_nucl_ori_ter.tif')
image_t10 = image[:,10]
microim = microshow(images=image_t10, fig_scaling=5,
                 cmaps=['pure_blue','pure_red', 'pure_green'],
                 unit='um', scalebar_units=2, unit_per_pix=0.065, scale_text_centered=True, scale_font_size=20,
                 label_text='A', label_font_size=30)

microim.ax.figure.savefig('../illustrations/composite.png', bbox_inches = 'tight', pad_inches = 0, dpi=600)
```

<img src="/illustrations/composite.png" alt="image" width="400">


It is then straightforward to extend a simple image into an animation as both objects take the same options. Additionally, a time-stamp can be added to the animation. This code generates the movie visible below:

```python
import numpy as np
import skimage.io
from microfilm.microanim import Microanim

image = skimage.io.imread('../demodata/coli_nucl_ori_ter.tif')
microanim = Microanim(data=image, cmaps=['pure_blue','pure_red', 'pure_green'], fig_scaling=5,
                      unit='um', scalebar_units=2, unit_per_pix=0.065,
                      scale_text_centered=True, scale_font_size=20,)
microanim.microim.add_label('A', label_font_size=30)
microanim.add_time_stamp('T', 10, location='lower left', timestamp_size=20)
```

<img src="/illustrations/composite_movie.mp4" alt="movie" width="400">

## Features

In addition to plotting and animation, ```microfilm``` currently provides two other modules: ```dataset```to handle time-lapse data of various formats and ```splitmasks``` which provides tools to analyze the evolution of intensity over time in complex regions. More documentation on those modules comes soon.

## Authors

This package has been created by Guillaume Witz, Micorscopy Imaging Center and Science IT Support, Bern University.
