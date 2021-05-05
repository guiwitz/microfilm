import warnings
import inspect

import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle


def colormap_from_name(image, cmap_name, flip_map=False, rescale_type='min_max', limits=None, num_colors=256):
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
    """
    
    image = rescale_image(image, rescale_type=rescale_type, limits=limits)
    
    try:
        cmap = plt.get_cmap(cmap_name, num_colors)
    except ValueError:
        if cmap_name == 'pure_red':
            cmap = ListedColormap(np.c_[np.linspace(0,1,num_colors), np.zeros(num_colors), np.zeros(num_colors)])
        if cmap_name == 'pure_green':
            cmap = ListedColormap(np.c_[np.zeros(num_colors), np.linspace(0,1,num_colors), np.zeros(num_colors)])
        if cmap_name == 'pure_blue':
            cmap = ListedColormap(np.c_[np.zeros(num_colors), np.zeros(num_colors), np.linspace(0,1,num_colors)])
        if cmap_name == 'segmentation':
            cmap = random_cmap(num_colors=num_colors)
    if flip_map:
            cmap = cmap.reversed()
            
    image_colored = cmap(image)
    
    return image_colored
    
def colormap_from_colorhex(image, cmap_hex='#ff6600', flip_map=False, rescale_type='min_max',
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

def random_cmap(alpha=0.5, num_colors=256):
    """Create random colormap for segmentation"""
    
    colmat = np.random.rand(num_colors,4)
    colmat[:,-1] = alpha
    colmat[0,-1] = 0
    cmap = matplotlib.colors.ListedColormap(colmat)
    return cmap


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
    
    if rescale_type == 'min_max':
        image_rescaled = (image-image.min())/(image.max()-image.min())
    elif rescale_type == 'dtype':
        image_rescaled = image / max_of_dtype
    elif rescale_type == 'zero_max':
        image_rescaled = image / image.max()
    elif rescale_type == 'limits':
        if limits is None:
            raise Exception(f"You need to provide explicit intensity limits of the form [min, max]")
        image_rescaled = image.astype(np.float)
        image_rescaled = image_rescaled - limits[0]
        image_rescaled[image_rescaled<0]=0
        image_rescaled[image_rescaled>limits[1]] = limits[1]
        image_rescaled = image_rescaled / limits[1]
    
    return image_rescaled

def combine_image(images):
    """Combine a list of 3D RGB arrays by max projection"""
    
    im_combined = np.max(np.stack(images,axis = 3),axis = 3)
    
    return im_combined

def multichannel_to_rgb(images, cmaps=None, flip_map=False, rescale_type='min_max',
                        limits=None, num_colors=256):
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
        [colormap_from_name(
            im, cmap_name=cmaps[ind],
            flip_map=flip_map[ind],
            rescale_type=rescale_type[ind],
            limits=limits[ind],
            num_colors=num_colors) for ind, im in enumerate(images)
        ])
    
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
            
    
def microshow(images, cmaps=None, flip_map=False, rescale_type='min_max', limits=None, num_colors=256,
              scalebar=False, height_pixels=3, unit_per_pix=None, scalebar_units=None, unit=None,
              scale_ypos=0.05, scale_color='white', scale_font_size=12,
              scale_text_centered=False, ax=None, fig_scaling=3
             ):
    """
    Plot image
    
    Parameters
    ----------
    images: list or array
        list of 2d arrays or DxMxN array D<4
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
    scalebar: bool
        add scale bar or not
    height_pixels: int
        height of scale bar
    unit_per_pix: float
        pixel scaling (e.g. 25um per pixel)
    scalebar_units: float
        size of scale bar in true units
    unit: str
        name of the scale unit
    scale_y_pos: float
        y position of scale bar (0-1)
    scale_color: str
        color of scale bar
    scale_font_size: int
        size of text, set to None for no text
    scale_text_centered: bool
        center text above scale bar
    ax: Matplotlib axis
        provide existing axis
    fig_scaling: int
        control figure scaling
            
    Returns
    -------
    Microimage object
    
    """
    
    images = check_input(images)
    converted = multichannel_to_rgb(images, cmaps=cmaps, flip_map=flip_map,
                                    rescale_type=rescale_type, limits=limits, num_colors=num_colors)
    
    height = images[0].shape[0]
    width = images[0].shape[1]
    if height > width:
        height_scaled = fig_scaling
        width_scaled = fig_scaling*width / height
    else:
        width_scaled = fig_scaling
        height_scaled = fig_scaling*height / width

    '''size_fact = 5
    fig = plt.figure()
    fig.set_size_inches(size_fact*width / height, size_fact, forward=False)
    ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
    fig.add_axes(ax)'''
    
    
    returnfig = False
    if ax is None:
        
        fig = plt.figure(frameon=False)
        fig.set_size_inches(width_scaled, height_scaled, forward=False)
        ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
        ax.set_axis_off()
        fig.add_axes(ax)
        #plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
        #plt.margins(0,0)

        #fig, ax = plt.subplots()
        returnfig = True
    else:
        fig = ax.figure
    ax.imshow(converted, interpolation='nearest')
    ax.set_axis_off()
    
    microim = Microimage(ax)
    
    if scalebar:
        
        image_width = images[0].shape[1]
        pixelsize = scalebar_units / unit_per_pix
        scale_width = pixelsize / image_width
        microim.add_scalebar(unit, scalebar_units, unit_per_pix, height_pixels=height_pixels,
                             scale_ypos=scale_ypos, scale_color=scale_color,
                             scale_font_size=scale_font_size, scale_text_centered=scale_text_centered)
        
    microim = Microimage(ax)  
    return microim
    

class Microimage:
    def __init__(self, ax):
        
        self.ax = ax        
    
    def add_scalebar(self, unit, scalebar_units, unit_per_pix, height_pixels=3, scale_ypos=0.05,
        scale_color='white', scale_font_size=12, scale_text_centered=False):
        """
        Add scalebar to an image.

        Parameters
        ----------
        unit: str
            name of the scale unit
        scalebar_units: float
            size of scale bar in true units
        unit_per_pix: float
            pixel scaling (e.g. 25um per pixel)
        height_pixels: int
            height of scale bar
        scale_y_pos: float
            y position of scale bar (0-1)
        scale_color: str
            color of scale bar
        scale_font_size: int
            size of text, set to None for no text
        scale_text_centered: bool
            center text above scale bar
            
        """
        
        if len(self.ax.get_images())==0:
            raise Exception(f"You need to have an image in your plot to add a scale bar.\
                Create your Microimage object using the microshow() function.")

        if (unit is None) or (scalebar_units is None) or (unit_per_pix is None):
            raise Exception(f"You need to provide a unit (unit), scale (unit_per_pix) and size of your scale bar (scalebar_units)")

        height_pixels /= self.ax.get_images()[0].get_array().shape[0]

        pixelsize = scalebar_units / unit_per_pix
        image_width = self.ax.get_images()[0].get_array().shape[1]
        scale_width = pixelsize / image_width

        if unit =='um':
            scale_text = f'{scalebar_units} $\mu$m'
        else:
            scale_text = f'{scalebar_units} {unit}'
        
        bar_pad = 0.05
        scale_bar = Rectangle((1-scale_width-bar_pad, scale_ypos), width=scale_width, height=height_pixels,
                              transform=self.ax.transAxes, facecolor=scale_color)
        
        if scale_font_size is not None:
            
            
            scale_text = self.ax.text(x=0, y=scale_ypos+height_pixels+0.02, s=scale_text,
                         transform=self.ax.transAxes, fontdict={'color':scale_color, 'size':scale_font_size})
            text_start = 1-scale_width-bar_pad
            
            # trick https://stackoverflow.com/questions/5320205/matplotlib-text-dimensions
            r = self.ax.figure.canvas.get_renderer()
            ax_width = self.ax.get_tightbbox(r).width
            text_width = scale_text.get_window_extent(renderer=r).width / ax_width
            
            if scale_text_centered:
                bar_middle = 1-0.5*scale_width-bar_pad
                text_start = bar_middle - 0.5*text_width
            
            # check if scale text outside image
            if text_start + text_width > 0.98:
                shift = text_start + text_width - 0.98
                scale_bar = Rectangle((1-scale_width-bar_pad-shift, scale_ypos), width=scale_width, 
                                      height=height_pixels, transform=self.ax.transAxes, facecolor=scale_color)
                text_start -= shift

            scale_text.set_x(text_start)
        self.ax.add_patch(scale_bar)
                
    def add_label(self, label_text, location='upper left', label_color='white',
                 label_font_size=15):
        """
        Add a figure label to an image.

        Parameters
        ----------
        label_text: str
            image label
        location: str or list
            position of the label on the image, can be
            'upper left', 'upper right', 'lower left', 'lower right' or
            a list with xy coordinates [xpos, ypos] where 0 < xpos, ypos < 1
        label_color: str
            color of label
        label_font_size: int
            size of label

        """
        
        r = self.ax.figure.canvas.get_renderer()
        label_text = self.ax.text(x=0.05, y=0.05, s=label_text,
                         transform=self.ax.transAxes, fontdict={'color':label_color, 'size':label_font_size})

        # label seems far from top but accomodates e.g. accents/trema
        image_height = self.ax.get_tightbbox(r).height
        image_width = self.ax.get_tightbbox(r).width

        text_height = label_text.get_window_extent(renderer=r).height / image_height
        text_width = label_text.get_window_extent(renderer=r).width / image_width
        
        if isinstance(location, list):
            if len(location) !=2:
                raise Exception(f"You need to provide a pair of xy positions as a list [xpos, ypos].")
            label_text.set_y(y=location[0])
            label_text.set_x(x=location[1])
                
        if location == 'upper left':
  
            label_text.set_y(y=1-text_height)
            label_text.set_x(x=0.01)
        
        elif location == 'upper right':
  
            label_text.set_y(y=1-text_height)
            label_text.set_x(x=1-text_width-0.01)
            
        elif location == 'lower left':
            
            label_text.set_y(y=0.01)
            label_text.set_x(x=0.01)
            
        elif location == 'lower right':
  
            label_text.set_y(y=0.01)
            label_text.set_x(x=1-text_width-0.01)

        return label_text