import inspect

from matplotlib.pyplot import figure, savefig, text
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib as mpl
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
import numpy as np
from matplotlib_scalebar.scalebar import ScaleBar

from . import colorify

   
def microshow(
    images=None, cmaps=None, flip_map=False, rescale_type=None, limits=None, 
    num_colors=256, proj_type='max', alpha=0.5, volume_proj=None, channel_names=None,
    channel_label_show=False, channel_label_type='title', channel_label_size=0.1,
    channel_label_line_space=0.1, scalebar_thickness=0.02, scalebar_unit_per_pix=None,
    scalebar_size_in_units=None, unit=None, scalebar_location='lower right', scalebar_color='white',
    scalebar_font_size=12, scalebar_kwargs=None, scalebar_font_properties=None,
    ax=None, fig_scaling=3, dpi=72, label_text=None, label_location='upper left',
    label_color='white', label_font_size=15, label_kwargs={}, cmap_objects=None,
    show_colorbar=False, show_axis=False, microim=None
    ):
    """
    Plot image
    
    Parameters
    ----------
    images: list or array
        list of 2d arrays or DxMxN array D<4
    cmaps: list of str / Matplotlib colormaps or single str
        colormap can be provided as names (e.g. 'pure_red' as specified
        in cmaps_def) or directly as Matplotlib colormaps (e.g. as returned by cmaps_def)
        for single channel images, you can pass a single str instead of a list
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
        alpha: alpa blending
    alpha: float
        transparency in range [0,1] of overlayed image for
        proj_type == alpha
    volume_proj: str
        projection type for volume images
        None: no projection
        'max': maximum projection
        'sum': sum projection, restricted to dtype range
        'mean': mean projection
    channel_names: list
        list of channel names
    channel_label_show: bool
    channel_label_type: str
        'title', 'in_fig'
    channel_label_size: float
        relative font size for channel labels
    channel_label_line_space: float
        space between channel labels as fraction of channel_label_size
    scalebar_thickness: float
        fraction height of scale bar
    scalebar_unit_per_pix: float
        pixel scaling (e.g. 25um per pixel)
    scalebar_size_in_units: float
        size of scale bar in true units
    unit: str
        name of the scale unit
    scale_location: str
       upper right, lower left etc.
    scalebar_color: str
        color of scale bar
    scalebar_font_size: int
        size of text, set to None for no text
    scalebar_kwargs: dict
        additional options for scalebar formatting passed
    scalebar_font_properties: dict
        font properties for scalebar text
    ax: Matplotlib axis
        provide existing axis
    fig_scaling: int
        control figure scaling
    dpi: int
        dots per inches passed to plt.figure
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
    label_kwargs: dict
        additional options for label formatting passed
        to Matplotlib text object
    cmap_objects: list
        list of Matplotlib colormaps to use for coloring
        if provided, the cmap names are ignored
    show_colorbar: bool
        show colorbar
    show_axis: bool
        show plot axis
    microim: Microimage object
        object to re-use
            
    Returns
    -------
    Microimage object
    
    """

    if microim is None:
        if images is None:
            raise Exception(f"You need to provide at least images")
 
        microim = Microimage(
            images=images, cmaps=cmaps, flip_map=flip_map, rescale_type=rescale_type,
            limits=limits, num_colors=num_colors, proj_type=proj_type, alpha=alpha,
            volume_proj=volume_proj, channel_names=channel_names, channel_label_show=channel_label_show,
            channel_label_type=channel_label_type, channel_label_size=channel_label_size,
            channel_label_line_space=channel_label_line_space, scalebar_thickness=scalebar_thickness,
            scalebar_unit_per_pix=scalebar_unit_per_pix, scalebar_size_in_units=scalebar_size_in_units,
            unit=unit, scalebar_location=scalebar_location, scalebar_color=scalebar_color,
            scalebar_font_size=scalebar_font_size, scalebar_kwargs=scalebar_kwargs,
            scalebar_font_properties=scalebar_font_properties, ax=ax, fig_scaling=fig_scaling,
            dpi=dpi, label_text=label_text, label_location=label_location,
            label_color=label_color, label_font_size=label_font_size, label_kwargs=label_kwargs,
            cmap_objects=cmap_objects, show_colorbar=show_colorbar, show_axis=show_axis
        )
    
    microim.rescale_type = colorify.check_rescale_type(microim.rescale_type, microim.limits)
    converted, cmap_objects, cmaps, image_min_max = colorify.multichannel_to_rgb(
        microim.images, cmaps=microim.cmaps, flip_map=microim.flip_map,
        rescale_type=microim.rescale_type, limits=microim.limits,
        num_colors=microim.num_colors, proj_type=microim.proj_type,
        alpha=microim.alpha, cmap_objects=microim.cmap_objects
        )
    microim.cmap_objects = cmap_objects
    microim.cmaps = cmaps
    microim.image_min_max = image_min_max
    
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
        microim.fig = plt.figure(frameon=False, dpi=microim.dpi)
        microim.fig.set_size_inches(width_scaled, height_scaled, forward=False)
        microim.ax = plt.Axes(microim.fig, [0.0, 0.0, 1.0, 1.0])
        if not show_axis:
             microim.ax.set_axis_off()
        microim.fig.add_axes(microim.ax)
    else:
        microim.fig = microim.ax.figure
    microim.ax.imshow(converted, interpolation='nearest')
    if not show_axis:
        microim.ax.set_axis_off()
 
    if microim.channel_names is None:
        microim.channel_names = channel_names
    if microim.channel_label_show:
        microim.add_channel_labels(microim.channel_names, microim.channel_label_size)
    
    if microim.unit is not None:
    
        microim.add_scalebar(
            unit=microim.unit,
            scalebar_size_in_units=microim.scalebar_size_in_units, 
            scalebar_unit_per_pix=microim.scalebar_unit_per_pix,
            scalebar_thickness=microim.scalebar_thickness,
            scalebar_location=microim.scalebar_location,
            scalebar_color=microim.scalebar_color,
            scalebar_font_size=microim.scalebar_font_size,
            scalebar_kwargs=microim.scalebar_kwargs,
            scalebar_font_properties=microim.scalebar_font_properties
            )
    
    if microim.label_text is not None:
        if len(microim.label_text) > 0:
            for key in microim.label_text:
                if key != 'time_stamp':
                    microim.add_label(label_text=microim.label_text[key],
                    label_name=key, label_location=microim.label_location[key],
                    label_color=microim.label_color[key], label_font_size=microim.label_font_size[key],
                    label_kwargs=microim.label_kwargs[key])

    if microim.show_colorbar:
        microim.add_colorbar()

    return microim

def volshow(images, volume_proj='mean', **kwargs):
    """
    Wrapper for microshow for volume images.
    """
    return microshow(images, volume_proj=volume_proj, **kwargs)

volshow.__doc__ = microshow.__doc__
docstr = volshow.__doc__
docstr = docstr.splitlines()
docstr[1] = '   Plot volume'
volshow.__doc__ = '\n'.join(docstr)    

class Microimage:
    """
    Class implementing the plot object. It is usually created by
    calling the microshow function but can also be used directly.
    
    Parameters
    ----------
    images: list or array
        list of 2d arrays or DxMxN array D<4
    cmaps: list of str / Matplotlib colormaps or single str
        colormap can be provided as names (e.g. 'pure_red' as specified
        in cmaps_def) or directly as Matplotlib colormaps (e.g. as returned by cmaps_def)
        for single channel images, you can pass a single str instead of a list
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
    alpha: float
        transparency in range [0,1] of overlayed image for
        proj_type == alpha
    volume_proj: str
        projection type for volume images
        None: no projection
        'max': maximum projection
        'sum': sum projection, restricted to dtype range
        'mean': mean projection
    channel_names: list
        list of channel names
    channel_label_show: bool
    channel_label_type: str
        'title', 'in_fig'
    channel_label_size: float
        relative font size of channel label
    channel_label_line_space: float
        space between channel labels as fraction of channel_label_size
    scalebar_thickness: float
        fraction of height of scale bar
    scalebar_unit_per_pix: float
        pixel scaling (e.g. 25um per pixel)
    scalebar_size_in_units: float
        size of scale bar in true units
    unit: str
        name of the scale unit
    scalebar_location: str
       upper right, lower right etc.
    scalebar_color: str
        color of scale bar
    scalebar_font_size: int
        size of text, set to None for no text
    scalebar_kwargs: dict
        additional keyword arguments for scalebar
    scalebar_font_properties: dict
        font properties for scalebar text
    ax: Matplotlib axis
        provide existing axis
    fig_scaling: int
        control figure scaling
    dpi: int
        dots per inches passed to plt.figure
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
    label_kwargs: dict
        additional options for label formatting passed
        to Matplotlib text object
    cmaps_object: list
        list of cmap objects for each channel 
        if provided, cmap names are ignored
    show_colorbar: bool
        show colorbar
    show_axis: bool
        show plot axis

    """

    def __init__(
        self, images, cmaps=None, flip_map=False, rescale_type=None, limits=None,
        num_colors=256, proj_type='max', alpha=0.5, volume_proj=None, channel_names=None,
        channel_label_show=False, channel_label_type='title', channel_label_size=0.1,
        channel_label_line_space=0.1, scalebar_thickness=0.02, scalebar_unit_per_pix=None,
        scalebar_size_in_units=None, unit=None, scalebar_location='lower right', scalebar_color='white',
        scalebar_font_size=12, scalebar_kwargs=None, scalebar_font_properties=None,
        ax=None, fig_scaling=3, dpi=72, label_text=None, label_location='upper left',
        label_color='white', label_font_size=15, label_kwargs={}, cmap_objects=None,
        show_colorbar=False, show_axis=False
        ):
        
        self.__dict__.update(locals())
        del self.self

        # if labels are provided convert them to dict from if necessary
        if isinstance(self.label_text, dict):
            self.label_text = label_text
            self.label_location = label_location
            self.label_color = label_color
            self.label_font_size = label_font_size
            self.label_kwargs = label_kwargs

        elif self.label_text is not None:
            self.label_text = {'label': label_text}
            self.label_location = {'label': label_location}
            self.label_color = {'label': label_color}
            self.label_font_size = {'label': label_font_size}
            self.label_kwargs = {'label': label_kwargs}
        else:
            self.label_text = None
            self.label_location = None
            self.label_color = None
            self.label_font_size = None
            self.label_kwargs = {}

        # check input
        is_volume = self.volume_proj is not None
        self.images = colorify.check_input(self.images, is_volume=is_volume)

        # do 3D projection if necessary
        if (self.volume_proj is not None) and (self.images is not None):
            self.images = colorify.project_volume(self.images, self.volume_proj)

        if (not isinstance(self.cmaps, list)) and self.cmaps is not None:
            self.cmaps = [self.cmaps]
        if isinstance(self.channel_names, str):
            self.channel_names = [self.channel_names]
        if not isinstance(self.flip_map, list):
            if self.images is not None:
                self.flip_map = [self.flip_map for i in range(len(self.images))]
            

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

        kwargs['bbox_inches'] = 'tight'
        kwargs['pad_inches'] = 0

        self.fig.savefig(*args, **kwargs)

    savefig.__doc__ = plt.savefig.__doc__

    
    def add_scalebar(
        self, unit, scalebar_size_in_units, scalebar_unit_per_pix, scalebar_thickness=0.02,
        scalebar_location='lower right', scalebar_color='white', scalebar_font_size=12,
        scalebar_kwargs=None, scalebar_font_properties=None):
        """
        Add scalebar to an image.

        Parameters
        ----------
        unit: str
            name of the scale unit
        scalebar_size_in_units: float
            size of scale bar in true units
        scalebar_unit_per_pix: float
            pixel scaling (e.g. 25um per pixel)
        scalebar_thickness: float
            fraction of height of scale bar
        scale_location: str
            upper right, lower left etc.
        scalebar_color: str
            color of scale bar
        scalebar_font_size: float
            relative size of text, set to None for no text
        scalebar_kwargs: dict
            additional options for scalebar formatting passed
        scalebar_font_properties: dict
            font properties for scalebar text
            
        """

        self.unit = unit
        self.scalebar_size_in_units = scalebar_size_in_units
        self.scalebar_unit_per_pix = scalebar_unit_per_pix
        self.scalebar_thickness = scalebar_thickness
        self.scalebar_location = scalebar_location
        self.scalebar_color = scalebar_color
        self.scalebar_font_size = scalebar_font_size
        self.scalebar_kwargs = scalebar_kwargs
        self.scalebar_font_properties = scalebar_font_properties
        
        if len(self.ax.get_images())==0:
            raise Exception(f"You need to have an image in your plot to add a scale bar.\
                Create your Microimage object using the microshow() function.")

        if (unit is None) or (scalebar_size_in_units is None) or (scalebar_unit_per_pix is None):
            raise Exception(f"You need to provide a unit (unit), scale (scalebar_unit_per_pix) and size of your scale bar (scalebar_size_in_units)")

        font_dict = {'size': scalebar_font_size}
        if self.scalebar_font_properties is not None:
            font_dict = {**font_dict, **scalebar_font_properties}
        
        default_scalebar_kwargs = {'frameon': False, 'scale_loc': 'top'}
        if self.scalebar_kwargs is not None:
            default_scalebar_kwargs = {**default_scalebar_kwargs, **self.scalebar_kwargs}
        if scalebar_font_size is None:
            default_scalebar_kwargs['scale_loc'] = 'none'
        
        scalebar = ScaleBar(
            dx=scalebar_unit_per_pix,
            units=unit,
            fixed_value=scalebar_size_in_units,
            width_fraction=scalebar_thickness,
            location=scalebar_location,
            color=scalebar_color,
            font_properties=font_dict,
            **default_scalebar_kwargs)
        self.ax.add_artist(scalebar)

                
    def add_label(
        self, label_text, label_name='default', label_location='upper left',
        label_color='white', label_font_size=15, label_kwargs={}):
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
        label_kwargs: dict
            additional options for label formatting passed
            to Matplotlib text object

        """
        
        if self.label_text is None:
            self.label_text = {}
            self.label_location = {}
            self.label_color = {}
            self.label_font_size = {}
            self.label_kwargs = {}

        self.label_text[label_name] = label_text
        self.label_location[label_name] = label_location
        self.label_color[label_name] = label_color
        self.label_font_size[label_name] = label_font_size
        self.label_kwargs[label_name] = label_kwargs

        r = self.ax.figure.canvas.get_renderer()
        # combine explicit options with Matplotlib kwargs
        fontdict = {'color':label_color, 'size':label_font_size}
        fontdict = {**fontdict, **label_kwargs}
        label_text = self.ax.text(x=0.05, y=0.05, s=label_text,
                         transform=self.ax.transAxes, fontdict=fontdict)

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

    def add_channel_labels(
        self, channel_names=None, channel_label_size=None,
        channel_label_line_space=None, channel_colors=None):
        """
        Add the channel names color with the corresponding colormap as figure title

        Parameters
        ----------
        channel_names: list
            list of channel names, defaults to channel-1, channel-2 etc.
        channel_label_size: float
            relative font size of label
        channel_label_line_space: float
            size of space between labels as fraction of channel_label_size
        channel_colors: list of array
            list of colors to use for the label each channel,
            defaults to the colormap

        """

        if channel_label_size is not None:
            self.channel_label_size = channel_label_size
        if channel_label_line_space is not None:
            self.channel_label_line_space = channel_label_line_space

        if channel_names is not None:
            self.channel_names = channel_names
        elif self.channel_names is None:
            self.channel_names = ['Channel-' + str(i) for i in range(len(self.images))]

        px = 1/plt.rcParams['figure.dpi']
        figheight_px = self.fig.get_size_inches()[1] / px
        
        line_space = self.channel_label_line_space * self.channel_label_size
        nlines = len(self.channel_names)
        tot_space =  ((nlines+0.5) * self.channel_label_size + (nlines-1)*line_space)

        fontsize = int(figheight_px * (1-tot_space) * self.channel_label_size)
        
        self.ax.set_position([self.ax.get_position().bounds[0], self.ax.get_position().bounds[1],
                 self.ax.get_position().bounds[2], self.ax.get_position().bounds[3]-tot_space])

        for i in range(nlines):
            # The factor (1-tot_space) is a rescaling of the y position to take into
            # account that the axis only occupies that portion of the figure
            if channel_colors is not None:
                text_color = channel_colors[nlines-1-i]
            elif self.flip_map[nlines-1-i] is False:
                text_color = self.cmap_objects[nlines-1-i](self.cmap_objects[nlines-1-i].N)
            else:
                text_color = self.cmap_objects[nlines-1-i](0)
            self.ax.text(
                x=0.5,
                y=1 + 0.5 * self.channel_label_size + i * (self.channel_label_size+line_space),#/(1-tot_space),
                s=self.channel_names[nlines-1-i], ha="center", transform=self.ax.transAxes,
                fontdict={'color': text_color,'size':fontsize}
            )

    def add_colorbar(self):
        """Add colorbar"""

        if len(self.cmaps) > 1:
            raise Exception("You can only add a colorbar for single channel images.")

        divider = make_axes_locatable(self.ax)
        ax_cb = divider.new_horizontal(size="5%", pad=0.05)
        self.fig.add_axes(ax_cb)
        for_mapping = mpl.cm.ScalarMappable(cmap=self.cmap_objects[0])
        for_mapping.set_clim(self.image_min_max[0][0],self.image_min_max[0][1])
        self.fig.colorbar(for_mapping, cax=ax_cb)
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
    margin: float
        fraction of figure size reserved for margins between plots
    figscaling: float
        adjust the size of the figure without providing
        explicit size
    figsize: list
        figure size [x, y]
    channel_label_size: float
        font size for channel labels (fraction of figure)
    channel_label_line_space: float
        space between channel labels (fraction of channel_label_size)
    fig_kwargs: parameters normally passed to plt.subplots()

    Attributes
    ----------
    fig: Matplotlib figure object
    ax: list
        list of Matplotlib axis objects
    microplots: 2d array
        array of Microimage objects

    """
    
    def __init__(
        self, rows, cols, margin=0.01, figscaling=5, figsize=None,
        channel_label_size=0.05, channel_label_line_space=0.1, **fig_kwargs):

        self.rows = rows
        self.cols = cols
        self.margin = margin
        self.figsize = figsize
        self.figscaling = figscaling
        self.channel_label_size = channel_label_size
        self.channel_label_line_space = channel_label_line_space
        self.fig_kwargs = fig_kwargs

        self.microplots = np.empty((rows, cols), dtype=object)
        self.construct_figure()
        
    def construct_figure(self):
        """Construct the figure"""
        
        if 'frameon' not in self.fig_kwargs.keys():
            self.fig_kwargs['frameon'] = False

        self.fig, self.ax = plt.subplots(
            nrows=self.rows, ncols=self.cols, figsize=self.figsize,
            squeeze=False,
            gridspec_kw = {'left':0, 'right':1, 'bottom':0, 'top':1, 'wspace':self.margin, 'hspace':self.margin},
            **self.fig_kwargs)

    def add_channel_label(self, channel_label_size=None, channel_label_line_space=None,
                          channel_names=None, channel_colors=None):
        """Add channel labels to all plots and set their size"""

        if channel_label_size is not None:
            self.channel_label_size = channel_label_size
        if channel_label_line_space is not None:
            self.channel_label_line_space = channel_label_line_space

        for i in range(self.rows):
            for j in range(self.cols):
                if self.microplots[i,j] is not None:
                    if channel_names is not None:
                        self.microplots[i,j].channel_names = channel_names[i][j]
                    elif self.microplots[i,j].channel_names is None:
                        self.microplots[i,j].channel_names = ['Channel-' + str(i) for i in range(len(self.microplots[i,j].images))]
        
        ## title params
        px = 1/plt.rcParams['figure.dpi']
        figheight_px = self.fig.get_size_inches()[1] / px

        line_space = self.channel_label_line_space * self.channel_label_size
        nlines = np.max([len(k) for k in [x.channel_names for x in self.microplots.ravel() if x is not None] if k is not None])

        tot_space =  ((nlines+0.5) * self.channel_label_size + (nlines-1)*line_space)
        # make the font size the fraction of the figure height *remaining* after adding the text
        fontsize = int(figheight_px * (1-(self.rows * tot_space)) * self.channel_label_size)

        # adjust figure size with label
        self.fig.clf()
        self.fig.set_size_inches([self.fig.get_size_inches()[0]*(1-tot_space),
                    self.fig.get_size_inches()[1]])
        self.fig, self.ax = plt.subplots(
                nrows=self.rows, ncols=self.cols,
                figsize=[self.fig.get_size_inches()[0]*(1-tot_space),
                    self.fig.get_size_inches()[1]],
                squeeze=False, num=self.fig.number,
                gridspec_kw = {'left':0, 'right':1, 'bottom':0, 'top':1, 'wspace':self.margin, 'hspace':self.margin},
                **self.fig_kwargs)

        for j in range(self.rows):
            for i in range(self.cols):

                self.ax[j,i].set_position([self.ax[j,i].get_position().bounds[0], self.ax[j,i].get_position().bounds[1],
                 self.ax[j,i].get_position().bounds[2], self.ax[j,i].get_position().bounds[3]-tot_space])

        for j in range(self.rows):
            for i in range(self.cols):
                if self.microplots[j, i] is not None:

                    xpos = self.ax[j,i].get_position().bounds[0]+0.5*self.ax[j,i].get_position().bounds[2]
                    ypos = self.ax[j,i].get_position().bounds[1]+self.ax[j,i].get_position().bounds[3]
                    num_lines = len(self.microplots[j,i].channel_names)

                    for k in range(num_lines):
                        
                        # find text color
                        if channel_colors is not None:
                            text_color = channel_colors[j][i][num_lines-1-k]
                        elif self.microplots[j,i].flip_map[num_lines-1-k] is False:
                            text_color = self.microplots[j,i].cmap_objects[num_lines-1-k](self.microplots[j,i].cmap_objects[num_lines-1-k].N)
                        else:
                            text_color = self.microplots[j,i].cmap_objects[num_lines-1-k](0)

                        text_to_plot = " "
                        if self.microplots[j, i].channel_names is not None:
                            text_to_plot = self.microplots[j, i].channel_names[num_lines-1-k]
                        self.fig.text(
                            x=xpos,
                            y = ypos + 0.5 * self.channel_label_size + k * (self.channel_label_size+line_space),
                            s=text_to_plot, ha="center",
                            transform=self.fig.transFigure,
                            fontdict={'color': text_color, 'size':fontsize}
                        )
                    self.ax[j,i].cla()
                    has_label = self.microplots[j,i].channel_label_show
                    self.microplots[j,i].channel_label_show = False
                    self.microplots[j,i].update(self.ax[j,i])
                    self.microplots[j,i].channel_label_show = has_label

    def add_element(self, pos, microim):
        """Add a microimage object to a panel
        
        Parameters
        ----------
        pos: list
            i,j position of the plot in the panel
        microim: Microim object
            object to add to panel

        """

        if isinstance(microim.images, list):
            im_dim = microim.images[0].shape
        else:
            im_dim = microim.images.shape[1:3]
        if self.figsize is None:
            self.fig.set_size_inches(
                w=self.cols*im_dim[1]/np.max(im_dim)*self.figscaling,
                h=self.rows*im_dim[0]/np.max(im_dim)*self.figscaling
            )

        has_label = microim.channel_label_show

        microim.channel_label_show = False
        microim.update(self.ax[pos[0], pos[1]])
    
        self.microplots[pos[0], pos[1]] = microim
        
        microim.channel_label_show = has_label
        

    def savefig(self, *args, **kwargs):

        self.fig.savefig(*args, **kwargs)
    savefig.__doc__ = plt.savefig.__doc__