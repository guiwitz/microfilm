[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/guiwitz/microfilm/master?urlpath=lab)
# Microfilm

This package is a collection of tools to handle and plot 2D time-lapse microscopy datasets based on the Matplotlib library. In particular it makes it possible to create in a single line of code plots of multi-channel datasets in *composite* mode, completed with standard annotations like a figure label and scale bar. In addition, one can animate such time-lapse plots interactively in Jupyter notebooks and to export the movies in standard formats (mp4, gif etc.) Following the model of packages like seaborn, ```microfilm``` offers tight integration with Matplotlib, giving full access to the underlying structures like axis and figures, allowing thus for the creation of arbitrarily complex plots.

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

image = skimage.io.imread('../microfilm/dataset/tests/Sample/mitosis.tif')
im_proj = image.max(axis=1)[0]

anim = microshow(
    images=im_proj, fig_scaling=5,
    cmaps=['pure_blue','pure_red'], limits=[[0,20000],[0,18000]],
    unit='um', scalebar_units=20, unit_per_pix=0.5,scale_text_centered=True, 
    scale_font_size=20, label_text='A', label_font_size=30)
```

<img src="/illustrations/composite.png" alt="image" width="400">


It is then straightforward to extend a simple image into an animation as both objects take the same options. Additionally, a time-stamp can be added to the animation. This code generates the movie visible below:

```python
import numpy as np
import skimage.io
from microfilm.microanim import Microanim

im_proj = np.moveaxis(image.max(axis=1),0,1)

# create animation
microanim = Microanim(
    data=im_proj, cmaps=['pure_blue','pure_red'],
    unit='um', scalebar_units=10, unit_per_pix=0.5, fig_scaling=5)
# add label
microanim.microim.add_label('A', label_font_size=30)
# add timestamp
microanim.add_time_stamp('S', 5, location='lower left', timestamp_size=20)
# save animation as gif
microanim.save_movie('../illustrations/composite_movie.gif')
```

<img src="/illustrations/composite_movie.gif" alt="movie" width="400">
