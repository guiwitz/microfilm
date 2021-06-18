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
        self, data, channels=None, cmaps=None, flip_map=False, rescale_type=None, limits=None, num_colors=256,
        proj_type='max', channel_names=None, channel_label_show=False, channel_label_type='title',
        channel_label_size=0.05, scalebar_thickness=3, scalebar_unit_per_pix=None, scalebar_size_in_units=None, unit=None,
        scalebar_ypos=0.05, scalebar_color='white', scalebar_font_size=0.08, scalebar_text_centered=False,
        ax=None, fig_scaling=3, label_text=None, label_location='upper left',
        label_color='white', label_font_size=15, show_plot=True
    ):
        super().__init__(
            None, cmaps, flip_map, rescale_type, limits, num_colors,
            proj_type, channel_names, channel_label_show, channel_label_type,
            channel_label_size, scalebar_thickness, scalebar_unit_per_pix, scalebar_size_in_units, unit,
            scalebar_ypos, scalebar_color, scalebar_font_size, scalebar_text_centered,
            ax, fig_scaling, label_text, label_location,
            label_color, label_font_size
        )

        if isinstance(data, np.ndarray):
            data = Nparray(nparray=data)

        self.data = data
        self.max_time = self.data.K-1
        
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

        converted = multichannel_to_rgb(self.images, cmaps=self.cmaps, flip_map=self.flip_map,
                                    rescale_type=self.rescale_type, limits=self.limits, num_colors=self.num_colors)

        self.ax.get_images()[0].set_data(converted)
        #print("timestamps1")
        if self.label_text is not None:
            if 'time_stamp' in self.label_text.keys():
                #print("timestamps")
                self.timestamps.set_text(self.times[t])

    def add_time_stamp(self, unit, unit_per_frame, location='upper left',
        timestamp_size=15, timestamp_color='white'):
        """
        Add time-stamp to movie
        
        Parameters
        ----------
        unit: str
            unit of time 'S' seconds, 'T' minute, 'H' hours
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

        """

        periods = self.data.K
        times = pd.date_range(start=0, periods=periods, freq=str(unit_per_frame)+unit)
        self.times = times.strftime('%H:%M:%S')

        self.timestamps = self.add_label(self.times[0], 'time_stamp', label_location=location,
        label_font_size=timestamp_size, label_color=timestamp_color)
    
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
    figsize: float or list
        figure size, either square or rectangular
    channel_label_size: float
        font size for channel labels (fraction of figure)
    fig_kwargs: parameters normally passed to plt.figure()

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
    
    def __init__(self, rows, cols, margin=0.01, figsize=2, channel_label_size=0.05, **fig_kwargs):

        if not isinstance(figsize, list):
            figsize = [figsize, figsize]

        self.rows = rows
        self.cols = cols
        self.margin = margin
        self.figsize = figsize
        self.channel_label_size = channel_label_size

        self.time_slider = ipw.IntSlider(
            description="Time", min=0, max=0, value=0, continuous_update=True
        )
        self.time_slider.observe(self.update_timeslider, names="value")

        self.output = ipw.Output()

        self.microanims = np.empty((rows, cols), dtype=object)
        
        ## grid params
        margin = margin * np.max(figsize)

        ## part size
        part_size_w = (1 - margin * (cols - 1))/cols
        part_size_h = (1 - margin * (rows - 1))/rows

        with self.output:
            
            self.fig, self.ax = plt.subplots(
                nrows=self.rows, ncols=self.cols, figsize=(self.figsize[1]*self.cols, self.figsize[0]*self.rows),
                squeeze=False, frameon=False,
                gridspec_kw = {'left':0, 'right':1, 'bottom':0, 'top':1, 'wspace':self.margin, 'hspace':self.margin},
                **fig_kwargs)

            '''fig = plt.figure(frameon=False)
            fig.set_size_inches(figsize[1]*cols, figsize[0]*rows, forward=False)

            # add axes
            ax = np.empty((rows, cols), dtype=object)
            for j in range(rows):
                for i in range(cols):
                    ax[rows-1-j,i] = plt.Axes(fig,
                                    [i*(part_size_w+margin),
                                        j*(part_size_h+margin),
                                        part_size_w,
                                        part_size_h])
                    fig.add_axes(ax[rows-1-j,i])

            self.ax = ax
            self.fig = fig'''
        
        self.ui = ipw.VBox([self.output, self.time_slider])

        self.debug = ipw.Output()

    def add_channel_label(self, channel_label_size=None):
        """Add channel labels to all plots and set their size"""
        
        if channel_label_size is not None:
            self.channel_label_size = channel_label_size
        ## title params
        line_space = 0.01
        nlines = np.max([len(k) for k in [x.channel_names for x in self.microanims.ravel() if x is not None] if k is not None])
        tot_space = nlines * (self.channel_label_size+line_space)
        fontsize = self.channel_label_size*self.figsize[0]*self.rows*100

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

                        for k in range(nlines):
                        
                            self.fig.text(
                                x=xpos,
                                y=ypos+line_space+k*(self.channel_label_size+line_space),
                                s=self.microanims[j, i].channel_names[k], ha="center",
                                transform=self.fig.transFigure,
                                fontdict={'color':colorify.color_translate(self.microanims[j,i].cmaps[k]), 'size':fontsize}
                            )
                        self.ax[j,i].cla()
                        self.add_element(pos=[j,i], microanim=self.microanims[j, i])
                        #self.microanims[j,i].update(self.ax[j,i])

    def add_element(self, pos, microanim):
        """Add an animation object to a panel
        
        Parameters
        ----------
        pos: list
            i,j position of the plot in the panel
        microanim: Microanim object
            object to add to panel

        """

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
                label_color=newanim.label_color['time_stamp'], label_font_size=newanim.label_font_size['time_stamp'])

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
            a.update_animation(t)
            with self.debug:
                print(a.timestamps)

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
            w,h = anim_object.fig.canvas.get_width_height()
            buf.shape = (h, w, 3)
            writer.append_data(buf)
            
        writer.close()
    