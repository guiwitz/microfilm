import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import pandas as pd
import ipywidgets as ipw
from .microplot import microshow, multichannel_to_rgb

class Microanim:
    
    def __init__(self, data, channels, cmaps=None, flip_map=False, rescale_type='min_max',
        limits=None, num_colors=256, scalebar=False, height_pixels=3, unit_per_pix=None, 
        scalebar_units=None, unit=None, scale_ypos=0.05, scale_color='white', scale_font_size=12,
        scale_text_centered=False, ax=None):

        """Standard __init__ method.
        
        Parameters
        ----------
        ax : matplotlib axis
        
        Attributes
        ----------
        
        """
        
        #self.ax = ax
        self.data = data
        self.channels = channels

        self.cmaps=cmaps
        self.flip_map=flip_map
        self.rescale_type=rescale_type
        self.limits=limits
        self.num_colors=num_colors

        self.time_slider = ipw.IntSlider(
            description="Time", min=0, max=self.data.K-1, value=0, continuous_update=True
        )
        self.time_slider.observe(self.show_segmentation, names="value")

        self.output = ipw.Output()
        self.timestamps = None

        # initialize
        images = [self.data.load_frame(x, 0) for x in self.channels]
        with self.output:
            self.microim = microshow(
                images, cmaps=cmaps, flip_map=flip_map, rescale_type=rescale_type, limits=limits, num_colors=num_colors,
                scalebar=scalebar, height_pixels=height_pixels, unit_per_pix=unit_per_pix,
                scalebar_units=scalebar_units, unit=unit, scale_ypos=scale_ypos,
                scale_color=scale_color, scale_font_size=scale_font_size,
                scale_text_centered=scale_text_centered, ax=ax)

        self.ui = ipw.VBox([self.output, self.time_slider])
            

    def show_segmentation(self, change=None):
        """Update segmentation plot"""

        t = self.time_slider.value
        
        images = [self.data.load_frame(x, t) for x in self.channels]

        converted = multichannel_to_rgb(images, cmaps=self.cmaps, flip_map=self.flip_map,
                                    rescale_type=self.rescale_type, limits=self.limits, num_colors=self.num_colors)

        self.microim.ax.get_images()[0].set_data(converted)
        if self.timestamps:
            self.timestamps.set_text(self.times[t])

    def add_time_stamp(self, unit, unit_per_frame, location='upper left'):
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

        """

        periods = self.data.K
        times = pd.date_range(start=0, periods=periods, freq=str(unit_per_frame)+unit)
        self.times = times.strftime('%H:%M:%S')

        self.timestamps = self.microim.add_label(self.times[0], location=location)


    
