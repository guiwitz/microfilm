import warnings

import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from matplotlib.colors import ListedColormap


def cmaps_def(cmap_name, num_colors=256, flip_map=False):
    """
    Return a colormap defined by its name
    
    Parameters
    ----------
    cmap_name: str
        Matplotlib colormap or 'pure_red', 'pure_green', 'pure_blue'
        'pure_magenta', 'pure_cyan', 'pure_yellow'
    num_colors: int
        number of steps in color scale
    flip_map: bool
        invert colormap

    Returns
    -------
    cmap: Matplotlib colormap

    """

    if cmap_name in plt.colormaps():
        cmap = plt.get_cmap(cmap_name, num_colors)
    elif cmap_name == 'pure_red':
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


def colorify_by_name(image, cmap_name, flip_map=False, rescale_type='min_max', limits=None, num_colors=256):
    """
    Return 2D image as 3D RGB stack colored with a given colormap.
    
    Parameters
    ----------
    image: 2d array
        image to convert to RGB
    cmap_name: str
        Matplotlib colormap or 'pure_red', 'pure_green', 'pure_blue'
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

    """
    
    image = rescale_image(image, rescale_type=rescale_type, limits=limits)
    
    cmap = cmaps_def(cmap_name, num_colors=num_colors, flip_map=flip_map)
            
    image_colored = cmap(image)
    
    return image_colored


def colorify_by_hex(image, cmap_hex='#ff6600', flip_map=False, rescale_type='min_max',
                           limits=None, num_colors=256):
    """
    Return 2D image as 3D RGB stack colored with a hex color.
    
    Parameters
    ----------
    image: 2d array
        image to convert to RGB
    cmap_name: str
        Matplotlib colormap or 'pure_red', 'pure_green', 'pure_blue'
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

    """
    
    image = rescale_image(image, rescale_type=rescale_type, limits=limits)
    chosen_col = np.array(list(int(cmap_hex[i:i+2], 16) for i in (1, 3, 5)))/(num_colors-1)
    new_col_scale = np.c_[np.linspace(0,chosen_col[0],num_colors),
                          np.linspace(0,chosen_col[1],num_colors),
                          np.linspace(0,chosen_col[2],num_colors)]
    cmap = ListedColormap(new_col_scale)
    if flip_map:
        cmap = cmap.reversed()
    image_colored = cmap(image)
    
    return image_colored

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
        
    """
    
    if not np.issubdtype(image.dtype, np.unsignedinteger):
        raise Exception(f"Image should be unsigned integer but yours is {image.dtype}")   

    max_of_dtype = np.iinfo(image.dtype).max
    image = image.astype(np.float64)
    
    if not np.any(image > 0):#blank image
        image_rescaled = image
    elif image.min() == image.max():#all pixels have same value
        image_rescaled = np.ones_like(image)
    elif rescale_type == 'min_max':
        min_val = np.min(image)#np.min(image[image>0])
        image_rescaled = (image-min_val)/(image.max()-min_val)
        image_rescaled[image_rescaled<0] = 0
    elif rescale_type == 'dtype':
        image_rescaled = image / max_of_dtype
    elif rescale_type == 'zero_max':
        image_rescaled = image / image.max()
    elif rescale_type == 'limits':
        if limits is None:
            raise Exception(f"You need to provide explicit intensity limits of the form [min, max]")
        image_rescaled = (image - limits[0]) / (limits[1] - limits[0])
        image_rescaled[image_rescaled<0] = 0
        image_rescaled[image_rescaled>1] = 1
    
    return image_rescaled


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


def combine_image(images, proj_type='max'):
    """
    Combine a list of 3D RGB arrays by max projection
    
    Parameters
    ----------
    images: list of arrays
        list of 2d arrays
    proj_type: str
        projection type of color combination
        max: maximum
        sum: sum projection, restricted to dtype range

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
    else:
        raise Exception(f"Your projection type {proj_type} is not implemented.")
    
    return im_combined


def multichannel_to_rgb(images, cmaps=None, flip_map=False, rescale_type='min_max',
                        limits=None, num_colors=256, proj_type='max'):
    """
    Convert a list of images to a single RGB image. Options can be passed
    as lists, one per channel, or as single element in which case the same value is used
    for all channel.
    
    Parameters
    ----------
    images: list or array
        list of 2d arrays or DxMxN array where D<4
    cmaps: list of str
        colormap names
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
        
    Returns
    -------
    image_rescaled: 2d array
        
    """
    
    images = check_input(images)
    
    if cmaps is not None:
        if len(images) != len(cmaps):
            raise Exception(f"You have {len(images)} images but only provided {len(cmaps)} color maps.")    
     
    # if no colormap is provided use true RGB
    if cmaps is None:
        cmaps = ['pure_red','pure_green','pure_blue']
        
    if not isinstance(flip_map, list):
        flip_map = [flip_map for i in range(len(images))]
    if not isinstance(rescale_type, list):
        rescale_type = [rescale_type for i in range(len(images))]
    if (limits is None) or (not any(isinstance(i, list) for i in limits)):
        limits = [limits for i in range(len(images))]
    
    converted = combine_image(
        [colorify_by_name(
            im, cmap_name=cmaps[ind],
            flip_map=flip_map[ind],
            rescale_type=rescale_type[ind],
            limits=limits[ind],
            num_colors=num_colors) for ind, im in enumerate(images)
        ], proj_type=proj_type)
    
    return converted


def check_input(images):
    """Converts input, either 2D array, list of 2D arrays or 3D array of size DxNxM where D<4
    to list of 2D arrays."""
    
    if isinstance(images, np.ndarray):
        if len(images.shape)==2:
            images = [images]
        elif len(images.shape)==3:
            if images.shape[0]>3:
                warnings.warn(f"Only the three first channels are considered. You have {images.shape[0]} channels.")
            images = [images[x] for x in range(np.min([3,images.shape[0]]))]
    elif isinstance(images, list):
        if not isinstance(images[0], np.ndarray):
            raise Exception(f"You need to pass a list of 2D arrays.")
        elif len(images[0].shape) !=2:
            raise Exception(f"Are should be 2D. You passed {len(images[0].shape)}D array.")
    
    return images

