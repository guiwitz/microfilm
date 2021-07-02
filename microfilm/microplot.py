import inspect

from matplotlib.pyplot import figure, savefig, text
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from numpy import dtype
import numpy as np

from . import colorify

   
def microshow(images=None, cmaps=None, flip_map=False, rescale_type=None, limits=None, num_colors=256,
              proj_type='max', channel_names=None, channel_label_show=False, channel_label_type='title',
              channel_label_size=0.05, scalebar_thickness=5, scalebar_unit_per_pix=None, scalebar_size_in_units=None,
              unit=None, scalebar_ypos=0.05, scalebar_color='white', scalebar_font_size=0.08, scalebar_text_centered=True,
              ax=None, fig_scaling=3, dpi=72, label_text=None, label_location='upper left',
              label_color='white', label_font_size=15, microim=None
             ):
    """
    Plot image
    
    Parameters
    ----------
    images: list or array
        list of 2d arrays or DxMxN array D<4
    cmaps: str of list of str
        colormap names. For single image you can pass a str
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
    channel_names: list
        list of channel names
    channel_label_show: bool
    channel_label_type: str
        'title', 'in_fig'
    channel_label_size: float
        relative font size for channel labels
    scalebar_thickness: int
        height of scale bar
    scalebar_unit_per_pix: float
        pixel scaling (e.g. 25um per pixel)
    scalebar_size_in_units: float
        size of scale bar in true units
    unit: str
        name of the scale unit
    scale_y_pos: float
        y position of scale bar (0-1)
    scalebar_color: str
        color of scale bar
    scalebar_font_size: int
        size of text, set to None for no text
    scalebar_text_centered: bool
        center text above scale bar
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
            
    Returns
    -------
    Microimage object
    
    """

    if microim is None:
        if images is None:
            raise Exception(f"You need to provide at least images")
 
        microim = Microimage(images=images, cmaps=cmaps, flip_map=flip_map, rescale_type=rescale_type,
        limits=limits, num_colors=num_colors, proj_type=proj_type, channel_names=channel_names,
        channel_label_show=channel_label_show, channel_label_type=channel_label_type,
        channel_label_size=channel_label_size, scalebar_thickness=scalebar_thickness, 
        scalebar_unit_per_pix=scalebar_unit_per_pix, scalebar_size_in_units=scalebar_size_in_units, unit=unit,
        scalebar_ypos=scalebar_ypos, scalebar_color=scalebar_color, scalebar_font_size=scalebar_font_size,
        scalebar_text_centered=scalebar_text_centered, ax=ax, fig_scaling=fig_scaling, dpi=dpi, label_text=label_text,
        label_location=label_location, label_color=label_color, label_font_size=label_font_size
        )
    
    #microim.images = colorify.check_input(microim.images)

    microim.rescale_type = colorify.check_rescale_type(microim.rescale_type, microim.limits)

    converted = colorify.multichannel_to_rgb(microim.images, cmaps=microim.cmaps, flip_map=microim.flip_map,
                                    rescale_type=microim.rescale_type, limits=microim.limits,
                                    num_colors=microim.num_colors, proj_type=microim.proj_type)
    
    if microim.cmaps is None:
            rgb = ['pure_red', 'pure_green', 'pure_blue']
            microim.cmaps = [rgb[k] for k in range(len(microim.images))]
    
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
        microim.ax = plt.Axes(microim.fig, [0.0, 0.0, 1.0-0.0, 1.0])
        microim.ax.set_axis_off()
        microim.fig.add_axes(microim.ax)
    else:
        microim.fig = microim.ax.figure
    microim.ax.imshow(converted, interpolation='nearest')
    microim.ax.set_axis_off()
 
    if microim.channel_names is None:
        microim.channel_names = channel_names
    if microim.channel_label_show:
        microim.add_channel_labels(microim.channel_names, microim.channel_label_size)
    
    if microim.unit is not None:
    
        image_width = microim.images[0].shape[1]
        pixelsize = microim.scalebar_size_in_units / microim.scalebar_unit_per_pix
        scale_width = pixelsize / image_width
        microim.add_scalebar(microim.unit, microim.scalebar_size_in_units, microim.scalebar_unit_per_pix,
                             scalebar_thickness=microim.scalebar_thickness, scalebar_ypos=microim.scalebar_ypos,
                             scalebar_color=microim.scalebar_color, scalebar_font_size=microim.scalebar_font_size,
                             scalebar_text_centered=microim.scalebar_text_centered)
    
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
    cmaps: str or list of str
        colormap names. For a single image you can pass as str
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
    channel_names: list
        list of channel names
    channel_label_show: bool
    channel_label_type: str
        'title', 'in_fig'
    channel_label_size: float
        relative font size of channel label
    scalebar_thickness: int
        height of scale bar
    scalebar_unit_per_pix: float
        pixel scaling (e.g. 25um per pixel)
    scalebar_size_in_units: float
        size of scale bar in true units
    unit: str
        name of the scale unit
    scale_y_pos: float
        y position of scale bar (0-1)
    scalebar_color: str
        color of scale bar
    scalebar_font_size: int
        size of text, set to None for no text
    scalebar_text_centered: bool
        center text above scale bar
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

    """

    def __init__(self, images, cmaps=None, flip_map=False, rescale_type=None, limits=None, num_colors=256,
              proj_type='max', channel_names=None, channel_label_show=False,
              channel_label_type='title', channel_label_size=0.05, scalebar_thickness=5, scalebar_unit_per_pix=None,
              scalebar_size_in_units=None, unit=None, scalebar_ypos=0.05, scalebar_color='white',
              scalebar_font_size=0.08, scalebar_text_centered=True, ax=None, fig_scaling=3, dpi=72, label_text=None,
              label_location='upper left', label_color='white', label_font_size=15
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

        # check input
        self.images = colorify.check_input(self.images)
        if isinstance(self.cmaps, str):
            self.cmaps = [self.cmaps]
        if isinstance(self.channel_names, str):
            self.channel_names = [self.channel_names]
            

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

    
    def add_scalebar(self, unit, scalebar_size_in_units, scalebar_unit_per_pix, scalebar_thickness=5, scalebar_ypos=0.05,
        scalebar_color='white', scalebar_font_size=0.08, scalebar_text_centered=True):
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
        scalebar_thickness: int
            height of scale bar
        scale_y_pos: float
            y position of scale bar (0-1)
        scalebar_color: str
            color of scale bar
        scalebar_font_size: float
            relative size of text, set to None for no text
        scalebar_text_centered: bool
            center text above scale bar
            
        """

        self.unit = unit
        self.scalebar_size_in_units = scalebar_size_in_units
        self.scalebar_unit_per_pix = scalebar_unit_per_pix
        self.scalebar_thickness = scalebar_thickness
        self.scalebar_ypos = scalebar_ypos
        self.scalebar_color = scalebar_color
        self.scalebar_font_size = scalebar_font_size
        self.scalebar_text_centered = scalebar_text_centered
        
        if len(self.ax.get_images())==0:
            raise Exception(f"You need to have an image in your plot to add a scale bar.\
                Create your Microimage object using the microshow() function.")

        if (unit is None) or (scalebar_size_in_units is None) or (scalebar_unit_per_pix is None):
            raise Exception(f"You need to provide a unit (unit), scale (scalebar_unit_per_pix) and size of your scale bar (scalebar_size_in_units)")

        scalebar_thickness /= self.ax.get_images()[0].get_array().shape[0]

        pixelsize = scalebar_size_in_units / scalebar_unit_per_pix
        image_width = self.ax.get_images()[0].get_array().shape[1]
        scale_width = pixelsize / image_width

        if unit =='um':
            scale_text = f'{scalebar_size_in_units} $\mu$m'
        else:
            scale_text = f'{scalebar_size_in_units} {unit}'
        
        bar_pad = 0.05
        scale_bar = Rectangle((1-scale_width-bar_pad, scalebar_ypos), width=scale_width, height=scalebar_thickness,
                              transform=self.ax.transAxes, facecolor=scalebar_color)
        
        if self.scalebar_font_size is not None:
                   
            # turn font size into fraction of figure or axis
            fontsize = self.scalebar_font_size*self.ax.get_position().bounds[-1] * self.fig.get_size_inches()[1]*100

            scale_text = self.ax.text(x=0, y=scalebar_ypos+scalebar_thickness+0.03, s=scale_text,
                         transform=self.ax.transAxes, fontdict={'color':scalebar_color, 'size':fontsize})
            text_start = 1-scale_width-bar_pad
            
            # trick https://stackoverflow.com/questions/5320205/matplotlib-text-dimensions
            r = self.ax.figure.canvas.get_renderer()

            right_boundary = (self.ax.get_images()[0].get_window_extent().bounds[0]+self.ax.get_images()[0].get_window_extent().bounds[2])
            axis_width = self.ax.get_images()[0].get_window_extent().bounds[2]
            if scalebar_text_centered:
                bar_middle = 1-0.5*scale_width-bar_pad
                text_start = bar_middle
                scale_text.set_ha('center')

            scale_text.set_x(text_start)

            shift = scale_text.get_tightbbox(r).bounds[0]+scale_text.get_tightbbox(r).bounds[2]-right_boundary
            shift = shift /axis_width
            
            # Check whether scalebar label is outside of figure and shift it if necessary
            if shift > 0:

                shift = shift
                scale_bar = Rectangle((1-scale_width-bar_pad-shift, scalebar_ypos), width=scale_width, 
                                      height=scalebar_thickness, transform=self.ax.transAxes, facecolor=scalebar_color)
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

    def add_channel_labels(self, channel_names=None, channel_label_size=0.05):
        """
        Add the channel names color with the corresponding colormap as figure title

        Parameters
        ----------
        channel_names: list
            list of channel names, defaults to channel-1, channel-2 etc.
        channel_label_size: float
            relative font size of label

        """

        if channel_names is not None:
            self.channel_names = channel_names
        self.channel_label_size = channel_label_size

        if self.channel_names is None:
            self.channel_names = ['Channel-' + str(i) for i in range(len(self.images))]

        figsize = self.fig.get_size_inches()[1]
        fontsize = channel_label_size*figsize*100

        line_space = 0.005 * figsize
        nlines = len(self.channel_names)

        tot_space = nlines * (channel_label_size+line_space)
        self.ax.set_position([self.ax.get_position().bounds[0], self.ax.get_position().bounds[1],
                 self.ax.get_position().bounds[2], self.ax.get_position().bounds[3]-tot_space])

        for i in range(nlines):
            # The factor (1-tot_space) is a rescaling of the y position to take into
            # account that the axis only occupies that portion of the figure
            self.ax.text(
                x=0.5, y=1+line_space+i*(channel_label_size+line_space)/(1-tot_space),
                s=self.channel_names[nlines-1-i], ha="center", transform=self.ax.transAxes,
                fontdict={'color':colorify.color_translate(self.cmaps[nlines-1-i]),'size':fontsize}
            )

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
        channel_label_size=0.05, label_line_space=0.2, **fig_kwargs):

        self.rows = rows
        self.cols = cols
        self.margin = margin
        self.figsize = figsize
        self.figscaling = figscaling
        self.channel_label_size = channel_label_size
        self.label_line_space = label_line_space
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

    def add_channel_label(self, channel_label_size=None, label_line_space=None):
        """Add channel labels to all plots and set their size"""

        if channel_label_size is not None:
            self.channel_label_size = channel_label_size
        if label_line_space is not None:
            self.label_line_space = label_line_space

        for i in range(self.rows):
            for j in range(self.cols):
                if self.microplots[i,j].channel_names is None:
                    self.microplots[i,j].channel_names = ['Channel-' + str(i) for i in range(len(self.microplots[i,j].images))]
        
        ## title params
        line_space = self.label_line_space * self.channel_label_size
        nlines = np.max([len(k) for k in [x.channel_names for x in self.microplots.ravel() if x is not None] if k is not None])

        tot_space = nlines * (self.channel_label_size+line_space)
        fontsize = self.channel_label_size*self.fig.get_size_inches()[1]*self.rows*100

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

                    for k in range(nlines):
                        
                        text_to_plot = " "
                        if self.microplots[j, i].channel_names is not None:
                            text_to_plot = self.microplots[j, i].channel_names[nlines-1-k]
                        self.fig.text(
                            x=xpos,
                            y = ypos + line_space + +k*(self.channel_label_size+line_space),
                            s=text_to_plot, ha="center",
                            transform=self.fig.transFigure,
                            fontdict={'color':colorify.color_translate(self.microplots[j,i].cmaps[nlines-1-k]), 'size':fontsize}
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