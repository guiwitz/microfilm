from typing import Iterable
import warnings

import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from matplotlib.colors import ListedColormap
from skimage.color import hsv2rgb
from skimage.exposure import rescale_intensity
from cmap import Colormap

def cmaps_def(cmap_name, num_colors=256, flip_map=False):
    """
    Return a colormap defined by its name
    
    Parameters
    ----------
    cmap_name: str
        cmap colormap, {'pure_red', 'pure_green', 'pure_blue', 'pure_magenta',
        'pure_cyan', 'pure_yellow'} or Matplotlib colormap
    num_colors: int
        number of steps in color scale
    flip_map: bool
        invert colormap

    Returns
    -------
    cmap: Matplotlib colormap

    """

    try:
        cmap = Colormap(cmap_name)
        cmap = cmap.to_matplotlib(N=num_colors)
    except ValueError:
        if cmap_name == 'pure_red':
            cmap = ListedColormap(np.c_[np.linspace(0,1,num_colors), np.zeros(num_colors), np.zeros(num_colors)])
        elif cmap_name == 'pure_green':
            cmap = ListedColormap(np.c_[np.zeros(num_colors), np.linspace(0,1,num_colors), np.zeros(num_colors)])
        elif cmap_name == 'pure_blue':
            cmap = ListedColormap(np.c_[np.zeros(num_colors), np.zeros(num_colors), np.linspace(0,1,num_colors)])
        elif cmap_name == 'pure_cyan':
            cmap = ListedColormap(np.c_[np.zeros(num_colors), np.linspace(0,1,num_colors), np.linspace(0,1,num_colors)])
        elif cmap_name == 'pure_magenta':
            cmap = ListedColormap(np.c_[np.linspace(0,1,num_colors), np.zeros(num_colors), np.linspace(0,1,num_colors)])
        elif cmap_name == 'pure_yellow':
            cmap = ListedColormap(np.c_[np.linspace(0,1,num_colors), np.linspace(0,1,num_colors), np.zeros(num_colors)])
        elif cmap_name == 'segmentation':
            cmap = random_cmap(num_colors=num_colors)
        elif cmap_name == 'ran_gradient':
            cmap = random_grandient_cmap(num_colors=num_colors)
        else:
            raise Exception(f"Your colormap {cmap_name} doesn't exist either in Matplotlib or microfilm.")
                
    if flip_map:
        cmap = cmap.reversed()
    
    return cmap
    
def color_translate(cmap_name):

    color_dict = {
        'pure_red': 'red',
        'pure_green': 'green',
        'pure_blue': 'blue',
        'pure_magenta': 'magenta',
        'pure_cyan': 'cyan',
        'pure_yellow': 'yellow',
        'gray': 'gray'
    }
    if cmap_name in color_dict.keys():
        return color_dict[cmap_name]
    else:
        warnings.warn(f"No appropriate color found for your colormap. Using black.'")
        return 'black'

def random_cmap(alpha=0.5, num_colors=256):
    """Create random colormap for segmentation"""
    
    colmat = np.random.rand(num_colors,4)
    colmat[:,-1] = alpha
    colmat[0,-1] = 0
    cmap = matplotlib.colors.ListedColormap(colmat)

    return cmap

def random_grandient_cmap(num_colors=25, seed=42):
    """Create a colormap as the gradient of a given random color"""

    rgb = hsv2rgb([np.random.random(1)[0], 0.95, 0.95])

    cmap = ListedColormap(np.c_[
        np.linspace(0,rgb[0],num_colors),
        np.linspace(0,rgb[1],num_colors), 
        np.linspace(0,rgb[2],num_colors)
        ])
    return cmap

def colorify_by_cmap(image, cmap, rescale_type='min_max', limits=None):
    """
    Directly use an existing colormap cmap to colorize an image

    Parameters
    ----------
    image: 2d array
        image to convert to RGB
    cmap: Matplotlib cmap
        colormap to use for coloring
    rescale_type: str
        'min_max': between extrema values of image
        'dtype': full range of image dtype
        'zero_max': between zero and image max
        'limits': between limits given by parameter limits
    limits: list
        [min, max] limits to use for rescaling

    Returns
    -------
    image_colored: array
        3D RGB float array
    min_max: tuple
        actual min and max values used for rescaling

    """

    image, min_max = rescale_image(image, rescale_type=rescale_type, limits=limits)
                
    image_colored = cmap(image)
    
    return image_colored, min_max

def colorify_by_name(image, cmap_name, flip_map=False, rescale_type='min_max', limits=None, num_colors=256):
    """
    Return 2D image as 3D RGB stack colored with a given colormap.
    
    Parameters
    ----------
    image: 2d array
        image to convert to RGB
    cmap_name: str
        {'pure_red', 'pure_green', 'pure_blue', 'pure_magenta',
        'pure_cyan', 'pure_yellow'} or Matplotlib colormap
    flip_map: bool
        invert colormap
    rescale_type: str
        'min_max': between extrema values of image
        'dtype': full range of image dtype
        'zero_max': between zero and image max
        'limits': between limits given by parameter limits
    limits: list
        [min, max] limits to use for rescaling
    num_colors: int
        number of steps in color scale

    Returns
    -------
    image_colored: array
        3D RGB float array
    cmap: Matplotlib colormap object
        Generated colormap from name
    min_max: tuple
        actual min and max values used for rescaling

    """
    
    image, min_max = rescale_image(image, rescale_type=rescale_type, limits=limits)
    
    cmap = cmaps_def(cmap_name, num_colors=num_colors, flip_map=flip_map)
            
    image_colored = cmap(image)
    
    return image_colored, cmap, min_max


def colorify_by_hex(image, cmap_hex='#ff6600', flip_map=False, rescale_type='min_max',
                           limits=None, num_colors=256):
    """
    Return 2D image as 3D RGB stack colored with a hex color.
    
    Parameters
    ----------
    image: 2d array
        image to convert to RGB
    cmap_hex: str
        hex string defining color, default '#ff6600' 
    flip_map: bool
        invert colormap
    rescale_type: str
        'min_max': between extrema values of image
        'dtype': full range of image dtype
        'zero_max': between zero and image max
        'limits': between limits given by parameter limits
    limits: list
        [min, max] limits to use for rescaling
    num_colors: int
        number of steps in color scale
    
    Returns
    -------
    image_colored: array
        3D RGB float array
    cmap: Matplotlib colormap object
        Generated colormap from name
    min_max: tuple
        actual min and max values used for rescaling

    """
    
    image, min_max = rescale_image(image, rescale_type=rescale_type, limits=limits)
    chosen_col = np.array(list(int(cmap_hex[i:i+2], 16) for i in (1, 3, 5)))/(num_colors-1)
    new_col_scale = np.c_[np.linspace(0,chosen_col[0],num_colors),
                          np.linspace(0,chosen_col[1],num_colors),
                          np.linspace(0,chosen_col[2],num_colors)]
    cmap = ListedColormap(new_col_scale)
    if flip_map:
        cmap = cmap.reversed()
    image_colored = cmap(image)
    
    return image_colored, cmap, min_max

def rescale_image(image, rescale_type='min_max', limits=None):
    """
    Rescale the image between 0-1 according to a rescaling type.
    
    Parameters
    ----------
    image: 2d array
        image to scale
    rescale_type: str
        'min_max': between extrema values of image
        'dtype': full range of image dtype
        'zero_max': between zero and image max
        'limits': between limits given by parameter limits
    limits: list
        [min, max] limits to use for rescaling
        
    Returns
    -------
    image_rescaled: 2d array
    min_max: tuple
        actual min and max values used for rescaling
        
    """
    
    min_max = (image.min(), image.max())
    if image.min() == image.max():#all pixels have same value
        image_rescaled = np.ones(image.shape, dtype=np.float64)
    if rescale_type == 'min_max':
        image_rescaled = rescale_intensity(image, in_range='image', out_range=(0,1))
    elif rescale_type == 'dtype':
        image_rescaled = rescale_intensity(image, in_range='dtype', out_range=(0,1))
    elif rescale_type == 'zero_max':
        image_rescaled = rescale_intensity(image, in_range=(0, image.max()), out_range=(0,1))
        min_max = (0, image.max())
    elif rescale_type == 'limits':
        if limits is None:
            raise Exception(f"You need to provide explicit intensity limits of the form [min, max]")
        image_rescaled = rescale_intensity(image, in_range=(limits[0], limits[1]), out_range=(0,1))
        min_max = (limits[0], limits[1])
    return image_rescaled, min_max


def check_rescale_type(rescale_type, limits):
    """Adjust rescale_type depending on its own value and that of limits"""

    # if limits provides use those otherwise default to min_max
    if limits is not None:
        if rescale_type is None:
            rescale_type = 'limits'
        elif rescale_type != 'limits':
            warnings.warn(f"You gave explicit limits but are not using 'limits'\
                for rescale_type. rescale_type is ignored and set to 'limits'")
    else:
        if rescale_type is None:
            rescale_type = 'min_max'
        elif rescale_type == 'limits':
            rescale_type = 'min_max'
            warnings.warn(f"You set rescale_type to 'limits'\
                but did not provide such limits. Defaulting to 'min_max'")

    return rescale_type


def combine_image(images, proj_type='max', alpha=0.5):
    """
    Combine a list of 3D RGB arrays into a single RGB image.
    The combination is done via maximum or a sum projection
    or via alpha blending.
    
    Parameters
    ----------
    images: list of arrays
        list of 3d rgb(a) arrays
    proj_type: str
        projection type of color combination
        max: maximum
        sum: sum projection, restricted to dtype range
        alpha: alpha blending
    alpha: float
        transparency in range [0,1] of overlayed image(s) for
        proj_type == alpha

    Returns
    -------
    im_combined: array
        3D RGB array
    """
    
    if proj_type == 'max':
        im_combined = np.max(np.stack(images,axis = 3),axis = 3)
    elif proj_type == 'sum':
        im_combined = np.sum(np.stack(images,axis = 3),axis = 3)
        im_combined[im_combined > 1] = 1
    elif proj_type == 'alpha':

        # take first image and successively overlay all next ones
        # keep already transparent values and replace opaque by alpha
        # taken from https://en.wikipedia.org/wiki/Alpha_compositing
        im_base = images[0].copy()
        for i in range(1, len(images)):
            alpha_a = images[i][:,:,3][:,:, np.newaxis]
            alpha_a[alpha_a > 0] = alpha
            alpha_b = im_base[:,:,3][:,:, np.newaxis]
            alpha_0 = alpha_a + alpha_b * (1 - alpha_a)
            im_combined = np.ones_like(images[0])
            im_combined[:,:,0:3] = (images[i][:,:,0:3] * alpha_a + im_base[:,:,0:3] * alpha_b * (1 - alpha_a)) / alpha_0
            im_base = im_combined
    
    else:
        raise Exception(f"Your projection type {proj_type} is not implemented.")
    
    return im_combined


def multichannel_to_rgb(images, cmaps=None, flip_map=False, rescale_type='min_max',
                        limits=None, num_colors=256, proj_type='max', alpha=0.5, cmap_objects=None):
    """
    Convert a list of images to a single RGB image. Options can be passed
    as lists, one per channel, or as single element in which case the same value is used
    for all channel.
    
    Parameters
    ----------
    images: list or array
        list of 2d arrays or DxMxN array where D<4
    cmaps: list of str / Matplotlib colormaps
        colormap as names (e.g. 'pure_red' as specified in cmaps_def) or
        directly as Matplotlib colormaps (e.g. as returned by cmaps_def)
    flip_map: bool or list of bool
        invert colormap or not
    rescale_type: str or list of str
        'min_max': between extrema values of image
        'dtype': full range of image dtype
        'zero_max': between zero and image max
        'limits': between limits given by parameter limits
    limits: list or list of lists
        [min, max] limits to use for rescaling
    num_colors: int
        number of steps in color scale
    proj_type: str
        projection type of color combination
        max: maximum
        sum: sum projection, restricted to dtype range
        alpha: alpha blending
    alpha: float
        transparency in range [0,1] of overlayed image for
        proj_type == alpha
    cmap_objects: list
        list of Matplotlib cmaps, one per channel to use for coloring
        if provided, no colormaps are computed and cmap names are ignored
        
    Returns
    -------
    converted: 2d array
        multi-channel RGB image
    cmap_objects: list
        list of Matplotlib cmaps generated for each channel
    cmaps: list of str / Matplotlib colormaps
        colormap as names (e.g. 'pure_red' as specified in cmaps_def) or
        directly as Matplotlib colormaps (e.g. as returned by cmaps_def).
        If input cmaps are provided, the same ones are returned. Otherwise
        a list of default colormaps is returned as list of strings.
    image_min_max: list of tuples
        actual (min, max) values used for rescaling for each image
        
    """
    
    # checks
    images = check_input(images)

    if not isinstance(rescale_type, list):
            rescale_type = [rescale_type for i in range(len(images))]
    if (limits is None) or (not any(isinstance(i, Iterable) for i in limits)):
        limits = [limits for i in range(len(images))]

    # if colormaps are provided, use them, otherwise compute them
    if cmap_objects is not None:
        colorified, image_min_max = zip(*[colorify_by_cmap(
            im, cmap=cmap_objects[ind],
            rescale_type=rescale_type[ind],
            limits=limits[ind]) for ind, im in enumerate(images)
        ])
        converted = combine_image(colorified, proj_type=proj_type, alpha=alpha)

    else:
        if cmaps is not None:
            if len(images) != len(cmaps):
                raise Exception(f"You have {len(images)} images but only provided {len(cmaps)} color maps.")    
        
        # if no colormap is provided use true RGB for d<4
        if cmaps is None:
            if len(images) == 1:
                cmaps = ['gray']
            elif len(images) < 4:
                cmaps = ['pure_cyan','pure_magenta','pure_yellow']
            else:
                cmaps = ['ran_gradient' for x in images]
            
        if not isinstance(flip_map, list):
            flip_map = [flip_map for i in range(len(images))]
        
        colorified = []
        cmap_objects = []
        image_min_max = []
        for ind, im in enumerate(images):
            if isinstance(cmaps[ind], str):
                col_by_name = colorify_by_name(
                    im, cmap_name=cmaps[ind],
                    flip_map=flip_map[ind],
                    rescale_type=rescale_type[ind],
                    limits=limits[ind],
                    num_colors=num_colors)
                colorified.append(col_by_name[0])
                cmap_objects.append(col_by_name[1])
                image_min_max.append(col_by_name[2])
            else:
                col_by_cmap, min_max = colorify_by_cmap(
                    im, cmap=cmaps[ind],
                    rescale_type=rescale_type[ind],
                    limits=limits[ind])
                colorified.append(col_by_cmap)
                cmap_objects.append(cmaps[ind])
                image_min_max.append(min_max)

        converted = combine_image(colorified, proj_type=proj_type, alpha=alpha)
    
    return converted, cmap_objects, cmaps, image_min_max


def check_input(images, is_volume=False):
    """Converts input, either 2D (3D if is_volume) array, list of 2D (3D) arrays or 3D (4D) array of size DxNxM (DxZxNxM)
    where D<4 to list of 2D (3D) arrays."""
    
    ndim = 2
    if is_volume:
        ndim = 3

    if isinstance(images, np.ndarray):
        if images.ndim == ndim:
            images = [images]
        elif images.ndim == ndim+1:
            images = [x for x in images]
    elif isinstance(images, list):
        if not isinstance(images[0], np.ndarray):
            raise Exception(f"You need to pass a list of 2D (or 3D if is_volume is True) arrays.")
        elif images[0].ndim !=ndim:
            raise Exception(f"Array should be {ndim}D. You passed {images[0].ndim}D array.")
    
    return images

def project_volume(images, proj_type):
    """
    Project a list of 3D images to a list of 2D images
    
    Parameters
    ----------
    images: list of arrays
        list of 3d arrays of shape ZxNxM
    proj_type: str
        projection type for volume
        max: maximum
        min: minimum
        sum: sum
        mean: mean

    Returns
    -------
    images_proj : list of 2D arrays (float32)
        projected arrays
    """
    
    images_proj = [im.astype(np.float32) for im in images]

    if proj_type == 'max':
        images_proj = [np.max(im, axis=0) for im in images_proj]
        return images_proj
    if proj_type == 'min':
        images_proj =  [np.min(im, axis=0) for im in images_proj]
        return images_proj
    elif proj_type == 'sum':
        images_proj =  [np.sum(im, axis=0) for im in images_proj]
        return images_proj
    elif proj_type == 'mean':
        images_proj =  [np.sum(im, axis=0)/len(im) for im in images_proj]
        return images_proj
    else:
        raise Exception(f"Your projection type {proj_type} is not implemented.")



