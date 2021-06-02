import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ipywidgets as ipw

from .microplot import multichannel_to_rgb, check_rescale_type
from .microplot import Microimage
from .dataset import Nparray

class Microanim(Microimage):
    """Data class to ingest multipage tiff files. A folder should contain multiple 
    single-plane multi-page tif files, one for each channel."""

    def __init__(
        self, data, channels=None, cmaps=None, flip_map=False, rescale_type=None, limits=None, num_colors=256,
        proj_type='max', height_pixels=3, unit_per_pix=None, scalebar_units=None, unit=None,
        scale_ypos=0.05, scale_color='white', scale_font_size=12, scale_text_centered=False,
        ax=None, fig_scaling=3, label_text=None, label_location='upper left',
        label_color='white', label_font_size=15
    ):
        super().__init__(
            None, cmaps, flip_map, rescale_type, limits, num_colors,
            proj_type, height_pixels, unit_per_pix, scalebar_units, unit,
            scale_ypos, scale_color, scale_font_size, scale_text_centered,
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
        
        with self.output:
            self.update()

        self.ui = ipw.VBox([self.output, self.time_slider])

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

class Microanimpanel:
    
    def __init__(self, rows, cols):

        self.microanims = []

        self.time_slider = ipw.IntSlider(
            description="Time", min=0, max=0, value=0, continuous_update=True
        )
        self.time_slider.observe(self.update_timeslider, names="value")

        self.output = ipw.Output()
        with self.output:
            self.fig, self.ax = plt.subplots(rows, cols)
        self.ui = ipw.VBox([self.output, self.time_slider])

        self.debug = ipw.Output()

    def add_element(self, pos, microanim):
        if isinstance(pos, list):
            microanim.ax = self.ax[pos[0], pos[1]]
        else:
            microanim.ax = self.ax[pos]
        microanim.show()
        if 'time_stamp' in microanim.label_text.keys():
            microanim.timestamps = microanim.add_label(microanim.times[0], 'time_stamp',
            label_location=microanim.label_location['time_stamp'],
            label_color=microanim.label_color['time_stamp'], label_font_size=microanim.label_font_size['time_stamp'])

        self.microanims.append(microanim)
        self.time_slider.max = microanim.data.K-1

    def update_timeslider(self, change=None):
        """Update segmentation plot"""

        t = self.time_slider.value
        self.update_animation(t)
    
    def update_animation(self, t):
        """Update all subplots"""

        for a in self.microanims:
            a.update_animation(t)
            with self.debug:
                print(a.timestamps)