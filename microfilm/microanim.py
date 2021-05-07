from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import pandas as pd
import ipywidgets as ipw
import imageio
from .microplot import microshow, multichannel_to_rgb, check_rescale_type
from .dataset import Nparray
class Microanim:
    
    def __init__(self, data, channels=None, cmaps=None, flip_map=False, rescale_type=None,
        limits=None, num_colors=256, height_pixels=3, unit_per_pix=None, 
        scalebar_units=None, unit=None, scale_ypos=0.05, scale_color='white', scale_font_size=12,
        scale_text_centered=False, ax=None, fig_scaling=3):

        """
        Class implementing methods to create interactive animations via ipywidgets in notebooks
        and to save animations as movies. Most options parameters are the same as for microplots.
        
        Parameters
        ----------
        data: microfilm.dataset.Data object or ndarray
            object allowing for easy loading of images. If an
            ndarray is used it needs dimension ordering CTXY
        channels: list of str
            list of channels from data object to plot
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
        
        Attributes
        ----------
        
        """
        
        if isinstance(data, np.ndarray):
            data = Nparray(nparray=data)

        self.data = data
        
        if channels is None:
            self.channels = self.data.channel_name
        else:
            self.channels = channels

        self.cmaps=cmaps
        self.flip_map=flip_map
        self.num_colors=num_colors

        self.rescale_type = check_rescale_type(rescale_type, limits)
        self.limits=limits
        
        self.time_slider = ipw.IntSlider(
            description="Time", min=0, max=self.data.K-1, value=0, continuous_update=True
        )
        self.time_slider.observe(self.update_timeslider, names="value")

        self.output = ipw.Output()
        self.timestamps = None

        # initialize
        images = [self.data.load_frame(x, 0) for x in self.channels]
        with self.output:
            self.microim = microshow(
                images, cmaps=cmaps, flip_map=flip_map, rescale_type=rescale_type, limits=limits, num_colors=num_colors,
                height_pixels=height_pixels, unit_per_pix=unit_per_pix,
                scalebar_units=scalebar_units, unit=unit, scale_ypos=scale_ypos,
                scale_color=scale_color, scale_font_size=scale_font_size,
                scale_text_centered=scale_text_centered, ax=ax, fig_scaling=fig_scaling)

        self.ui = ipw.VBox([self.output, self.time_slider])
            

    def update_timeslider(self, change=None):
        """Update segmentation plot"""

        t = self.time_slider.value
        self.update_animation(t)
        
    def update_animation(self, t):
        """Update animation to time t"""

        images = [self.data.load_frame(x, t) for x in self.channels]

        converted = multichannel_to_rgb(images, cmaps=self.cmaps, flip_map=self.flip_map,
                                    rescale_type=self.rescale_type, limits=self.limits, num_colors=self.num_colors)

        self.microim.ax.get_images()[0].set_data(converted)
        if self.timestamps:
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

        self.timestamps = self.microim.add_label(self.times[0], label_location=location,
        label_font_size=timestamp_size, label_color=timestamp_color)

    def save_movie(self, movie_name, fps=20, quality=5, format=None):
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

        for t in range(self.data.K-1):
            self.update_animation(t)
            self.microim.ax.figure.canvas.draw()
            buf = np.frombuffer(self.microim.ax.figure.canvas.tostring_rgb(), dtype=np.uint8 )
            w,h = self.microim.ax.figure.canvas.get_width_height()
            buf.shape = (h, w, 3)
            writer.append_data(buf)
            
        writer.close()



    
