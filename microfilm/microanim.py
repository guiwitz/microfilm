from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ipywidgets as ipw
import imageio

from . import colorify
from .colorify import multichannel_to_rgb, check_rescale_type
from .microplot import Microimage, microshow
from .dataset import Nparray

class Microanim(Microimage):
    """
    Class implementing an animation object. This object is a subclass of of the
    Microimage object and takes the same options. The main difference is that it
    takes a time-lapse dataset as obligatory parameter, not a simple image.
    """

    def __init__(
        self, data, channels=None, cmaps=None, flip_map=False, rescale_type=None, limits=None, 
        num_colors=256, proj_type='max', alpha=0.5, volume_proj=None, channel_names=None,
        channel_label_show=False, channel_label_type='title', channel_label_size=0.05,
        channel_label_line_space=0.1, scalebar_thickness=0.1, scalebar_unit_per_pix=None,
        scalebar_size_in_units=None, unit=None, scalebar_location='lower right', scalebar_color='white',
        scalebar_font_size=12, scalebar_kwargs=None, scalebar_font_properties=None,
        ax=None, fig_scaling=3, dpi=72, label_text=None, label_location='upper left',
        label_color='white', label_font_size=15, label_kwargs={}, cmap_objects=None,
        show_colorbar=False, show_axis=False, show_plot=True
    ):
        super().__init__(
            images=None, cmaps=cmaps, flip_map=flip_map, rescale_type=rescale_type, limits=limits,
            num_colors=num_colors, proj_type=proj_type, alpha=alpha, volume_proj=volume_proj, channel_names=channel_names,
            channel_label_show=channel_label_show, channel_label_type=channel_label_type, channel_label_size=channel_label_size,
            channel_label_line_space=channel_label_line_space, scalebar_thickness=scalebar_thickness, scalebar_unit_per_pix=scalebar_unit_per_pix,
            scalebar_size_in_units=scalebar_size_in_units,
            unit=unit, scalebar_location=scalebar_location, scalebar_color=scalebar_color,
            scalebar_font_size=scalebar_font_size, scalebar_kwargs=scalebar_kwargs, scalebar_font_properties=scalebar_font_properties,
            ax=ax, fig_scaling=fig_scaling, dpi=dpi, label_text=label_text, label_location=label_location,
            label_color=label_color, label_font_size=label_font_size, label_kwargs=label_kwargs, cmap_objects=cmap_objects,
            show_colorbar=show_colorbar, show_axis=show_axis
        )

        # check for correct dimensions
        if isinstance(data, np.ndarray):
            target_dim=5 if (volume_proj is not None) else 4
            if data.ndim != target_dim:
                raise ValueError(f"The array needs {target_dim} dimensions, yours has {data.ndim}")
            data = Nparray(nparray=data)

        self.data = data
        if volume_proj is not None:
            if type(data) != Nparray:
                raise ValueError(f"Volume projection requires a dataset.Nparray object, you have {type(data)}")
        self.max_time = self.data.K-1

        if not isinstance(self.flip_map, list):
            self.flip_map = [self.flip_map for i in range(len(self.data.channel_name))]
        
        if channels is None:
            self.channels = self.data.channel_name
        else:
            self.channels = channels
        
        self.rescale_type = check_rescale_type(rescale_type, limits)

        self.time_slider = ipw.IntSlider(
            description="Time", min=0, max=self.data.K-1, value=0, continuous_update=True
        )
        self.time_slider.observe(self.update_timeslider, names="value")

        self.output = ipw.Output()
        self.timestamps = None

        self.images = [self.data.load_frame(x, 0) for x in self.channels]
        # do 3D projection if necessary
        if (self.volume_proj is not None):
            self.images = colorify.project_volume(self.images, self.volume_proj)

        if show_plot:
            self.show()
        self.ui = ipw.VBox([self.output, self.time_slider])

    def show(self):
        """Display animation object"""

        with self.output:
            self.update()


    def update_timeslider(self, change=None):
        """Update segmentation plot"""

        t = self.time_slider.value
        self.update_animation(t)
        
    def update_animation(self, t):
        """Update animation to time t"""

        self.images = [self.data.load_frame(x, t) for x in self.channels]
        # do 3D projection if necessary
        if (self.volume_proj is not None):
            self.images = colorify.project_volume(self.images, self.volume_proj)

        converted, _, _, _ = multichannel_to_rgb(self.images, cmaps=self.cmaps, flip_map=self.flip_map,
                                    rescale_type=self.rescale_type, limits=self.limits, num_colors=self.num_colors,
                                    cmap_objects=self.cmap_objects)

        self.ax.get_images()[0].set_data(converted)

        if self.label_text is not None:
            if 'time_stamp' in self.label_text.keys():
                #print("timestamps")
                self.timestamps.set_text(self.times[t])

    def add_time_stamp(self, unit, unit_per_frame, location='upper left',
        timestamp_size=15, timestamp_color='white', time_format='HH:MM:SS', time_stamp_kwargs={}):
        """
        Add time-stamp to movie
        
        Parameters
        ----------
        unit: str
            unit of time 'mmm' millisecond, 'SS' seconds, 'MM' minute, 'HH' hours
        unit_per_frame: int
            number of units per frame e.g. 5 for 5s steps
        location: str or list
            position of the time-stamp on the image, can be
            'upper left', 'upper right', 'lower left', 'lower right' or
            a list with xy coordinates [xpos, ypos] where 0 < xpos, ypos < 1
        timestamp_size: int
            size of timestamp font
        timestamp_color: str
            color of label
        time_format: str
            How the time stamp should be represented. The string should contain
            the 'HH' for hours, 'MM' for minutes, 'SS' for seconds and 'mmm' for milliseconds
        time_stamp_kwargs: dict
            additional options for label formatting passed
            to Matplotlib text object

        """

        periods = self.data.K
        #times = pd.date_range(start=0, periods=periods, freq=str(unit_per_frame)+unit)
        #self.times = times.strftime('%H:%M:%S')
        _, self.times = self.time_range(unit=unit, unit_per_frame=unit_per_frame, num_step=periods, time_format=time_format)

        self.timestamps = self.add_label(self.times[0], 'time_stamp', label_location=location,
        label_font_size=timestamp_size, label_color=timestamp_color, label_kwargs=time_stamp_kwargs)


    def time_range(self, unit, unit_per_frame, num_step, time_format = 'HH:MM:SS:mmm'):
        """
        Create string time-stamps with arbitrary formatting.
        
        Parameters
        ----------
        unit: str
            unit of time 'mmm' millisecond, 'SS' seconds, 'MM' minute, 'HH' hours
        unit_per_frame: int
            number of units per frame e.g. 5 for 5s steps
        num_steps: int
            number of steps
        time_format: str
            How the time stamp should be represented. The string should contain
            the 'HH' for hours, 'MM' for minutes, 'SS' for seconds and 'mmm' for milliseconds

        """
        times = np.zeros((num_step, 4), dtype=np.uint16)

        pos_increment = 0
        if unit == 'MM':
            pos_increment = 1
        elif unit == 'SS':
            pos_increment = 2
        elif unit == 'mmm':
            pos_increment = 3
        for i in range(1, num_step):
            times[i, :] = times[i-1, :].copy()
            times[i, pos_increment] = times[i, pos_increment] + unit_per_frame
            if times[i, 3] >= 1000:
                times[i, 3] = 0
                times[i,2]+=1
            if times[i, 2] >= 60:
                times[i, 2] = 0
                times[i, 1] = times[i,1].copy()+1
            if times[i, 1] >= 60:
                times[i, 1] = 0
                times[i, 0] = times[i,0].copy()+1
                
        maxH = len(str(times[-1,0]))
        
        times_text = []
        for i in range(num_step):
            current_text = time_format
            current_text = current_text.replace('mmm', str(times[i,3]).zfill(4))
            current_text = current_text.replace('HH', str(times[i,0]).zfill(maxH))
            current_text = current_text.replace('MM', str(times[i,1]).zfill(2))
            current_text = current_text.replace('SS', str(times[i,2]).zfill(2))
            times_text.append(current_text)
        
        return times, times_text
    
    def save_movie(self, movie_name, fps=20, quality=5, format=None):
        save_movie(self, movie_name, fps=fps, quality=quality, format=format)

class Microanimpanel:
    """
    Class implementing a multi-panel animation. All animations should 
    have the same number of time points.

    Parameters
    ----------
    rows: int
        number of rows
    cols: int
        number of columns
    margin: float
        fraction of figure size reserved for margins between plots
    figscaling: float
        adjust the size of the figure without providing
        explicit size
    figsize: float or list
        figure size, either square or rectangular
    dpi: int
        dots per inches passed to plt.figure
    channel_label_size: float
        font size for channel labels (fraction of figure)
    channel_label_line_space: float
        space between label lines in fraction of channel_label_size
    fig_kwargs: parameters normally passed to plt.subplots()

    Attributes
    ----------
    microanims: list
        list of Microanim objects
    time_slider: ipywidget slider
        time slider
    output: ipywidget Output
        widget to display plot
    fig: Matplotlib figure object
        figure containing the panel
    ax: list
        list of Matplotlib axis objects
    ui: ipywidgets box
        animation interface
    max_time: int
        number of time points

    """
    
    def __init__(
        self, rows, cols, margin=0.01, figscaling=5, 
        figsize=None, channel_label_size=0.05, channel_label_line_space=0.2, **fig_kwargs):

        self.rows = rows
        self.cols = cols
        self.margin = margin
        self.figsize = figsize
        self.figscaling = figscaling
        self.channel_label_size = channel_label_size
        self.channel_label_line_space = channel_label_line_space
        self.fig_kwargs = fig_kwargs
        if 'frameon' not in self.fig_kwargs.keys():
            self.fig_kwargs['frameon'] = False

        self.time_slider = ipw.IntSlider(
            description="Time", min=0, max=0, value=0, continuous_update=True
        )
        self.time_slider.observe(self.update_timeslider, names="value")

        self.output = ipw.Output()

        self.microanims = np.empty((rows, cols), dtype=object)

        with self.output:
            
            self.fig, self.ax = plt.subplots(
                nrows=self.rows, ncols=self.cols, figsize=self.figsize,
                squeeze=False,
                gridspec_kw = {'left':0, 'right':1, 'bottom':0, 'top':1, 'wspace':self.margin, 'hspace':self.margin},
                **self.fig_kwargs)
        
        self.ui = ipw.VBox([self.output, self.time_slider])

        self.debug = ipw.Output()

    def add_channel_label(self, channel_label_size=None, channel_label_line_space=None):
        """Add channel labels to all plots and set their size"""
        
        if channel_label_size is not None:
            self.channel_label_size = channel_label_size
        if channel_label_line_space is not None:
            self.channel_label_line_space = channel_label_line_space

        for i in range(self.rows):
            for j in range(self.cols):
                if self.microanims[i,j].channel_names is None:
                    self.microanims[i,j].channel_names = ['Channel-' + str(i) for i in range(len(self.microanims[i,j].images))]
        
        ## title params
        px = 1/plt.rcParams['figure.dpi']
        figheight_px = self.fig.get_size_inches()[1] / px
        
        line_space = self.channel_label_line_space * self.channel_label_size
        nlines = np.max([len(k) for k in [x.channel_names for x in self.microanims.ravel() if x is not None] if k is not None])
        
        tot_space =  ((nlines+0.5) * self.channel_label_size + (nlines-1)*line_space)
        fontsize = int(figheight_px * (1-(self.rows * tot_space)) * self.channel_label_size)

        self.output.clear_output()
        with self.output:
            self.fig, self.ax = plt.subplots(
                nrows=self.rows, ncols=self.cols,
                figsize=[self.fig.get_size_inches()[0]*(1-tot_space),
                    self.fig.get_size_inches()[1]],
                squeeze=False,
                gridspec_kw = {'left':0, 'right':1, 'bottom':0, 'top':1, 'wspace':self.margin, 'hspace':self.margin},
                **self.fig_kwargs)
        

        for j in range(self.rows):
            for i in range(self.cols):

                self.ax[j,i].set_position([self.ax[j,i].get_position().bounds[0], self.ax[j,i].get_position().bounds[1],
                 self.ax[j,i].get_position().bounds[2], self.ax[j,i].get_position().bounds[3]-tot_space])

        for j in range(self.rows):
            for i in range(self.cols):
                if self.microanims[j, i] is not None:
                    if self.microanims[j, i].channel_names is not None:

                        xpos = self.ax[j,i].get_position().bounds[0]+0.5*self.ax[j,i].get_position().bounds[2]
                        ypos = self.ax[j,i].get_position().bounds[1]+self.ax[j,i].get_position().bounds[3]
                        num_lines = len(self.microanims[j,i].channel_names)

                        for k in range(num_lines):
                            # find text color
                            if self.microanims[j,i].flip_map[nlines-1-k] is False:
                                text_color = self.microanims[j,i].cmap_objects[nlines-1-k](self.microanims[j,i].cmap_objects[nlines-1-k].N)
                            else:
                                text_color = self.microanims[j,i].cmap_objects[nlines-1-k](0)
                            
                            self.fig.text(
                                x=xpos,
                                y=ypos + 0.5 * self.channel_label_size + k * (self.channel_label_size+line_space),
                                s=self.microanims[j, i].channel_names[nlines-1-k], ha="center",
                                transform=self.fig.transFigure,
                                fontdict={'color': text_color, 'size':fontsize}
                            )
                        self.ax[j,i].cla()
                        self.add_element(pos=[j,i], microanim=self.microanims[j, i], fig_update=False)

    def add_element(self, pos, microanim, fig_update=True):
        """Add an animation object to a panel
        
        Parameters
        ----------
        pos: list
            i,j position of the plot in the panel
        microanim: Microanim object
            object to add to panel

        """

        im_dim = microanim.data.dims
        if (self.figsize is None) and (fig_update):
            self.fig.set_size_inches(
                w=self.cols*im_dim[1]/np.max(im_dim)*self.figscaling,
                h=self.rows*im_dim[0]/np.max(im_dim)*self.figscaling
            )

        has_label = microanim.channel_label_show

        microanim.channel_label_show = False
        selaxis = self.ax[pos[0], pos[1]]
        
        newanim = Microanim(microanim.data, show_plot=False, ax=selaxis)
        
        micro_dict = microanim.__dict__
        for k in micro_dict:
            if (k != 'ax') and (k!='fig'):
                newanim.__setattr__(k, micro_dict[k])
        with self.output:
            newanim.update(selaxis)
        
        if newanim.label_text is not None:
            if 'time_stamp' in newanim.label_text.keys():
                newanim.timestamps = newanim.add_label(newanim.times[0], 'time_stamp',
                label_location=newanim.label_location['time_stamp'],
                label_color=newanim.label_color['time_stamp'],
                label_font_size=newanim.label_font_size['time_stamp'],
                label_kwargs=newanim.label_kwargs['time_stamp'])

        self.microanims[pos[0], pos[1]] = newanim
        self.max_time = newanim.data.K-1
        self.time_slider.max = newanim.data.K-1

        microanim.channel_label_show = has_label

    def update_timeslider(self, change=None):
        """Update segmentation plot"""

        t = self.time_slider.value
        self.update_animation(t)
    
    def update_animation(self, t):
        """Update all subplots"""

        for a in self.microanims.ravel():
            if a is not None:
                a.update_animation(t)
                #with self.debug:
                #    print(a.timestamps)

    def save_movie(self, movie_name, fps=20, quality=5, format=None):
        save_movie(self, movie_name, fps=fps, quality=quality, format=format)

def save_movie(anim_object, movie_name, fps=20, quality=5, format=None):
        """Save a movie
        
        Parameters
        ----------
        movie_name: str or path object
            where to save the movie
        fps: int
            frames per second
        quality: int
            quality of images (see imageio)
        format: str
            format for export
        """

        path_obj = Path(movie_name)

        if path_obj.suffix in [".mov", ".avi", ".mpg", ".mpeg", ".mp4", ".mkv", ".wmv"]:
            writer = imageio.get_writer(
                path_obj,
                fps=fps,
                quality=quality,
                format=format,
            )
        else:
            writer = imageio.get_writer(path_obj, fps=fps, format=format)

        for t in range(anim_object.max_time):
            anim_object.update_animation(t)
            #self.ax.figure.canvas.draw()
            anim_object.fig.canvas.draw()
            #buf = np.frombuffer(self.ax.figure.canvas.tostring_rgb(), dtype=np.uint8 )
            buf = np.frombuffer(anim_object.fig.canvas.tostring_rgb(), dtype=np.uint8 )
            #w,h = anim_object.ax.figure.canvas.get_width_height()
            #w,h = anim_object.fig.canvas.get_width_height()
            w,h = map(int, anim_object.fig.canvas.renderer.get_canvas_width_height())
            buf.shape = (h, w, 3)
            writer.append_data(buf)
            
        writer.close()
    