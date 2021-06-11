import inspect

from matplotlib.pyplot import savefig
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from . import colorify

   
def microshow(images=None, cmaps=None, flip_map=False, rescale_type=None, limits=None, num_colors=256,
              proj_type='max', height_pixels=3, unit_per_pix=None, scalebar_units=None, unit=None,
              scale_ypos=0.05, scale_color='white', scale_font_size=12, scale_text_centered=False,
              ax=None, fig_scaling=3, label_text=None, label_location='upper left',
              label_color='white', label_font_size=15, microim=None
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
    proj_type: str
        projection type of color combination
        max: maximum
        sum: sum projection, restricted to dtype range
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
    label_text: str
        image label
    label_location: str or list
        position of the label on the image, can be
        'upper left', 'upper right', 'lower left', 'lower right' or
        a list with xy coordinates [xpos, ypos] where 0 < xpos, ypos < 1
    label_color: str
        color of label
    label_font_size: int
        size of label
            
    Returns
    -------
    Microimage object
    
    """

    if microim is None:
        if images is None:
            raise Exception(f"You need to provide at least images")
 
        microim = Microimage(images=images, cmaps=cmaps, flip_map=flip_map, rescale_type=rescale_type,
        limits=limits, num_colors=num_colors, proj_type=proj_type, height_pixels=height_pixels, 
        unit_per_pix=unit_per_pix, scalebar_units=scalebar_units, unit=unit,
        scale_ypos=scale_ypos, scale_color=scale_color, scale_font_size=scale_font_size,
        scale_text_centered=scale_text_centered, ax=ax, fig_scaling=fig_scaling, label_text=label_text,
        label_location=label_location, label_color=label_color, label_font_size=label_font_size
        )
    
    microim.images = colorify.check_input(microim.images)

    microim.rescale_type = colorify.check_rescale_type(microim.rescale_type, microim.limits)

    converted = colorify.multichannel_to_rgb(microim.images, cmaps=microim.cmaps, flip_map=microim.flip_map,
                                    rescale_type=microim.rescale_type, limits=microim.limits,
                                    num_colors=microim.num_colors, proj_type=microim.proj_type)
    
    height = microim.images[0].shape[0]
    width = microim.images[0].shape[1]
    if height > width:
        height_scaled = microim.fig_scaling
        width_scaled = microim.fig_scaling*width / height
    else:
        width_scaled = microim.fig_scaling
        height_scaled = microim.fig_scaling*height / width
    
    if microim.ax is None:
        # trick https://stackoverflow.com/a/63187965
        microim.fig = plt.figure(frameon=False)
        microim.fig.set_size_inches(width_scaled, height_scaled, forward=False)
        microim.ax = plt.Axes(microim.fig, [0.0, 0.0, 1.0, 1.0])
        microim.ax.set_axis_off()
        microim.fig.add_axes(microim.ax)
    else:
        microim.fig = microim.ax.figure
    microim.ax.imshow(converted, interpolation='nearest')
    microim.ax.set_axis_off()
    
    if microim.unit is not None:
    
        image_width = microim.images[0].shape[1]
        pixelsize = microim.scalebar_units / microim.unit_per_pix
        scale_width = pixelsize / image_width
        microim.add_scalebar(microim.unit, microim.scalebar_units, microim.unit_per_pix,
                             height_pixels=microim.height_pixels, scale_ypos=microim.scale_ypos,
                             scale_color=microim.scale_color, scale_font_size=microim.scale_font_size,
                             scale_text_centered=microim.scale_text_centered)
    
    if microim.label_text is not None:
        if len(microim.label_text) > 0:
            for key in microim.label_text:
                if key != 'time_stamp':
                    microim.add_label(label_text=microim.label_text[key],
                    label_name=key, label_location=microim.label_location[key],
                    label_color=microim.label_color[key], label_font_size=microim.label_font_size[key])

    return microim
    

class Microimage:
    """
    Class implementing the plot object. It is usually created by
    calling the microshow function but can also be used directly.
    
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
    proj_type: str
        projection type of color combination
        max: maximum
        sum: sum projection, restricted to dtype range
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
    label_text: str
        image label
    label_location: str or list
        position of the label on the image, can be
        'upper left', 'upper right', 'lower left', 'lower right' or
        a list with xy coordinates [xpos, ypos] where 0 < xpos, ypos < 1
    label_color: str
        color of label
    label_font_size: int
        size of label

    """

    def __init__(self, images, cmaps=None, flip_map=False, rescale_type=None, limits=None, num_colors=256,
              proj_type='max', height_pixels=3, unit_per_pix=None, scalebar_units=None, unit=None,
              scale_ypos=0.05, scale_color='white', scale_font_size=12, scale_text_centered=False,
              ax=None, fig_scaling=3, label_text=None, label_location='upper left',
              label_color='white', label_font_size=15
             ):
        
        self.__dict__.update(locals())
        del self.self

        # if labels are provided convert them to dict from if necessary
        if isinstance(self.label_text, dict):
            self.label_text = label_text
            self.label_location = label_location
            self.label_color = label_color
            self.label_font_size = label_font_size

        elif self.label_text is not None:
            self.label_text = {'label': label_text}
            self.label_location = {'label': label_location}
            self.label_color = {'label': label_color}
            self.label_font_size = {'label': label_font_size}
        else:
            self.label_text = None
            self.label_location = None
            self.label_color = None
            self.label_font_size = None

    def update(self, ax=None, copy=False):
        """
        Update the Microimage axis or create a new copy of the object
        with a new axis and figure.

        Parameters
        ----------
        ax: Matplotlib axis
            Matplotlib axis to use for plot
        copy : bool
            create a new figure for plot, by default False

        Returns
        -------
        microim: if a new plot object is created (copy==True), the
                object is returned
        """
        
        if copy is False:
            self.ax = ax
            if self.ax is not None:
                self.fig = self.ax.figure
            microshow(microim=self)
        else:
            params = self.__dict__
            params_copy = params.copy()
            param_needed = list(inspect.signature(microshow).parameters)
            for k in params.keys():
                if k not in param_needed:
                    params_copy.pop(k, None)

            params_copy['ax'] = ax
            new_microim = microshow(**params_copy, microim=None)
            return new_microim

    def savefig(self, *args, **kwargs):

        self.fig.savefig(*args, **kwargs)
    savefig.__doc__ = plt.savefig.__doc__

    
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

        self.unit = unit
        self.scalebar_units = scalebar_units
        self.unit_per_pix = unit_per_pix
        self.height_pixels = height_pixels
        self.scale_ypos = scale_ypos
        self.scale_color = scale_color
        self.scale_font_size = scale_font_size
        self.scale_text_centered = scale_text_centered
        
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
                
    def add_label(self, label_text, label_name='default', label_location='upper left', label_color='white',
                 label_font_size=15):
        """
        Add a figure label to an image.

        Parameters
        ----------
        label_text: str
            image label
        label_location: str or list
            position of the label on the image, can be
            'upper left', 'upper right', 'lower left', 'lower right' or
            a list with xy coordinates [xpos, ypos] where 0 < xpos, ypos < 1
        label_color: str
            color of label
        label_font_size: int
            size of label

        """
        
        if self.label_text is None:
            self.label_text = {}
            self.label_location = {}
            self.label_color = {}
            self.label_font_size = {}

        self.label_text[label_name] = label_text
        self.label_location[label_name] = label_location
        self.label_color[label_name] = label_color
        self.label_font_size[label_name] = label_font_size

        r = self.ax.figure.canvas.get_renderer()
        label_text = self.ax.text(x=0.05, y=0.05, s=label_text,
                         transform=self.ax.transAxes, fontdict={'color':label_color, 'size':label_font_size})

        # label seems far from top but accomodates e.g. accents/trema
        image_height = self.ax.get_tightbbox(r).height
        image_width = self.ax.get_tightbbox(r).width

        text_height = label_text.get_window_extent(renderer=r).height / image_height
        text_width = label_text.get_window_extent(renderer=r).width / image_width
        
        if isinstance(label_location, list):
            if len(label_location) !=2:
                raise Exception(f"You need to provide a pair of xy positions as a list [xpos, ypos].")
            label_text.set_y(y=label_location[0])
            label_text.set_x(x=label_location[1])
                
        if label_location == 'upper left':
  
            label_text.set_y(y=1-text_height)
            label_text.set_x(x=0.01)
        
        elif label_location == 'upper right':
  
            label_text.set_y(y=1-text_height)
            label_text.set_x(x=1-text_width-0.01)
            
        elif label_location == 'lower left':
            
            label_text.set_y(y=0.01)
            label_text.set_x(x=0.01)
            
        elif label_location == 'lower right':
  
            label_text.set_y(y=0.01)
            label_text.set_x(x=1-text_width-0.01)

        return label_text

class Micropanel:
    """
    Class implementing a panel object of multiple Microimage
    objects.

    Parameters
    ----------
    rows: int
        number of panel rows
    cols: int
        number of panel columns
    fig_kwargs: parameters normally passed to plt.subplots()

    Attributes
    ----------
    fig: Matplotlib figure object
    ax: list
        list of Matplotlib axis objects
    microplots: list
        list of Microimage objects

    """
    
    def __init__(self, rows, cols, **fig_kwargs):

        self.microplots = []
        self.fig, self.ax = plt.subplots(rows, cols, **fig_kwargs)

    def add_element(self, pos, microim):
        if isinstance(pos, list):
            microim.update(self.ax[pos[0], pos[1]])
        else:
            microim.update(self.ax[pos])
        
        self.microplots.append(microim)

    def savefig(self, *args, **kwargs):

        self.fig.savefig(*args, **kwargs)
    savefig.__doc__ = plt.savefig.__doc__