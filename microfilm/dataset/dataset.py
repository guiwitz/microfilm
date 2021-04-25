import os
import re
from pathlib import Path
import warnings

from aicsimageio import AICSImage
from nd2reader import ND2Reader
import h5py
import skimage.io
import numpy as np
import natsort


class Data:
    """
    Class defining and handling datasets of 2D multi-channel time-lapse images.
    Given an experiment directory (tiff files or stacks) or a a file (ND2, H5)
    the available data are automatically parsed. Parameters specific to an analysis
    run such as bad frame indices or the time steps are also stored in the Data object.

    Parameters
    ----------
    expdir: str
        path to data folder (tif) or file (ND2)
    channel_name: list of str
        names of data to use as channels
    bad_frames: list, optional
        list of time-points to discard
    step: int
        time step to use when iterating across frames
    max_time: int. optional
        last frame to consider
    data_type: str,
        type of data considers ("series", "multi", "nd2", "h5)

    Attributes
    ----------
    K: int
        Number of frames considered (including steps, bad frames)
    dims: tuple
        XY image dimensions
    valid_frames: 1D array
        indices of considered frames
    channelfile: str or list
        'series':
            list of list of str, each element is a filename,
            files are grouped in a list for each channel, and
            all list grouped in a larger list
        'multi', 'h5':
            list of str, each element is a filename corresponding
            to a channel
        'nd2':
            list of str, each element is a channel name matching
            metadata information

    """

    def __init__(
        self,
        expdir,
        channel_name=None,
        bad_frames=[],
        step=1,
        max_time=None,
        data_type=None,
    ):

        self.data_type = data_type
        self.expdir = Path(expdir)
        self.channel_name = channel_name
        self.bad_frames = np.array(bad_frames)
        self.step = step
        self.max_time = max_time

        self.dims = None
        self.channelfile = None
        self.data_type = data_type

    def set_valid_frames(self):
        """Create a list of indices of valid frames given a set of bad frames"""

        self.valid_frames = np.arange(self.max_time)
        self.valid_frames = self.valid_frames[
            ~np.in1d(self.valid_frames, self.bad_frames)
        ]
        self.valid_frames = self.valid_frames[:: self.step]
        self.K = len(self.valid_frames)

    def find_files(self, folderpath):
        """Given a folder, find all tif files contained in it and try to sort as time-lapse"""

        image_names = os.listdir(folderpath)
        image_names = np.array([x for x in image_names if x[0] != "."])
        if len(image_names) > 0:
            # check if xxx_t0.tif structure is found
            times = [re.findall(".*\_t*(\d+)\.(?:tif|TIF|tiff)", x) for x in image_names]
            
            # if any element doesn't have xxx_t0.tif structure, find tif files and use natsort
            if any([len(x)==0 for x in times]):
                image_names = [re.findall(".*\.(?:tif|TIF|tiff)", x) for x in image_names]
                image_names = [t[0] for t in image_names if len(t)>0]
                if len(image_names) > 0:
                    image_names = natsort.natsorted(image_names)
                    warnings.warn(f"No times detected, using natural name sorting")
                else:
                    raise Exception(f"Sorry, no tif/tiff/TIF files could be found")
            else:
                times = [int(x[0]) for x in times if len(x) > 0]
                image_names = image_names[np.argsort(times)]
        else:
            raise Exception(f"Sorry, files found in {folderpath}")
        return image_names

    def update_params(self, params):
        """Update frame parameters"""

        self.max_time = params.max_time
        self.step = params.step
        self.bad_frames = params.bad_frames
        self.set_valid_frames()

    def check_channel_time_available(self, channel, frame):
        """Given a channel and frame index, check that they exist"""

        assert (channel in self.channel_name), f"{channel} is not in the list of available channels {self.channel_name}."
        assert (frame < len(self.valid_frames)), f"Only {len(self.valid_frames)} frames available."

    def load_frame(self, channel_name, frame):
        """Load index k of valid frames of channel index m in self.channelfile"""
        raise NotImplementedError

    def frame_generator(self, channel):
        """Create a generator returning successive frames of a given channel"""
        
        for t in self.valid_frames:
            image = self.load_frame(channel, t)
            yield image

    def get_channel_name(self, m):
        """Get name of channel index m"""

        return self.channel_name[m]

class TIFFSeries(Data):
    def __init__(
        self,
        expdir,
        channel_name=None,
        bad_frames=[],
        step=1,
        max_time=None,
        data_type="series",
    ):
        super().__init__(
            expdir,
            channel_name,
            bad_frames,
            step,
            max_time,
            data_type,
        )

        self.initialize()

    def initialize(self):

        # if no channel names are provided, consider all folders as channel
        if self.channel_name is None:
            self.channel_name = []
            for f in self.expdir.glob('*'):
                if (f.is_dir()) and (f.name[0] != '.'):
                    self.channel_name.append(f.name)
        if len(self.channel_name) == 0:
            raise Exception(f"Sorry, no folders found in {self.expdir}")

        self.channelfile = [
                self.find_files(os.path.join(self.expdir, x)) for x in self.channel_name
            ]

        if self.max_time is None:
            self.max_time = len(self.channelfile[0])
            # print(self.max_time)

        self.set_valid_frames()

        image = self.load_frame(self.channel_name[0], 0)
        self.dims = image.shape
        self.shape = image.shape

    def load_frame(self, channel_name, frame):
        """Load index k of valid frames of channel index m in self.channelfile"""

        self.check_channel_time_available(channel_name, frame)
        ch_index = self.channel_name.index(channel_name)

        time = self.valid_frames[frame]
        full_path = os.path.join(
            self.expdir, channel_name, self.channelfile[ch_index][time]
        )
        return skimage.io.imread(full_path).astype(dtype=np.uint16)


class MultipageTIFF(Data):
    def __init__(
        self,
        expdir,
        channel_name=None,
        bad_frames=[],
        step=1,
        max_time=None,
        data_type="multi",
    ):
        super().__init__(
            expdir,
            channel_name,
            bad_frames,
            step,
            max_time,
            data_type,
        )

        self.initialize()

    def initialize(self):
        
        # if no channel names are provided, consider all folders as channel
        if self.channel_name is None:
            self.channel_name = []
            for ext in ('*.tif', '*.TIFF', '*.tiff'):
                files = self.expdir.glob(ext)
                for f in files:
                    self.channel_name.append(f.name)
        if len(self.channel_name) == 0:
            raise Exception(f"Sorry, no tif/tiff/TIFF files found in {self.expdir}")

        self.channelfile = self.channel_name

        self.channel_imobj = [
            AICSImage(os.path.join(self.expdir, x), known_dims="TYX")
            for x in self.channel_name
        ]

        if self.max_time is None:
            self.max_time = self.channel_imobj[0].size_t

        self.set_valid_frames()

        image = self.load_frame(self.channel_name[0],0)
        self.dims = image.shape
        self.shape = image.shape

    def load_frame(self, channel_name, frame):
        """Load index k of valid frames of channel index m in self.channelfile"""

        self.check_channel_time_available(channel_name, frame)

        time = self.valid_frames[frame]
        ch_index = self.channel_name.index(channel_name)

        image = self.channel_imobj[ch_index].get_image_data("YX", S=0, T=time, C=0, Z=0)
        return image.astype(dtype=np.uint16)



class ND2(Data):
    def __init__(
        self,
        expdir,
        channel_name=None,
        bad_frames=[],
        step=1,
        max_time=None,
        data_type="nd2",
    ):
        super().__init__(
            expdir,
            channel_name,
            bad_frames,
            step,
            max_time,
            data_type,
        )

        self.initialize()

    def initialize(self):
        
        self.nd2file = ND2Reader(self.expdir.as_posix())
        self.nd2file.metadata["z_levels"] = range(0)

        if self.channel_name is None:
            self.channel_name = list(self.nd2file.metadata["channels"])
        self.channelfile = self.channel_name


        if self.max_time is None:
            self.max_time = self.nd2file.sizes["t"]

        self.set_valid_frames()

        image = self.load_frame(self.channel_name[0], 0)
        self.dims = image.shape
        self.shape = image.shape

    def load_frame(self, channel_name, frame):
        """Load index k of valid frames of channel index m in self.channelfile"""

        self.check_channel_time_available(channel_name, frame)

        time = self.valid_frames[frame]

        ch_index = self.nd2file.metadata["channels"].index(channel_name)
        image = self.nd2file.get_frame_2D(x=0, y=0, z=0, c=ch_index, t=time, v=0)
        return image


class H5(Data):
    def __init__(
        self,
        expdir,
        channel_name=None,
        bad_frames=[],
        step=1,
        max_time=None,
        data_type="h5",
    ):
        super().__init__(
            expdir,
            channel_name,
            bad_frames,
            step,
            max_time,
            data_type,
        )

        self.initialize()

    def initialize(self):

        if self.channel_name is None:
            self.channel_name = []
            for ext in ('*.h5','*.H5'):
                files = self.expdir.glob(ext)
                for f in files:
                    self.channel_name.append(f.name)
        if len(self.channel_name) == 0:
            raise Exception(f"Sorry, no tif/tiff/TIFF files found in {self.expdir}")

        self.channelfile = self.channel_name
        
        self.channel_imobj = [
            h5py.File(os.path.join(self.expdir, x), "r").get("volume")
            for x in self.channel_name
        ]

        if self.max_time is None:
            self.max_time = self.channel_imobj[0].shape[0]

        self.set_valid_frames()

        image = self.load_frame(self.channel_name[0], 0)
        self.dims = image.shape
        self.shape = image.shape

    def load_frame(self, channel_name, frame):
        """Load index k of valid frames of channel index m in self.channelfile"""

        self.check_channel_time_available(channel_name, frame)

        time = self.valid_frames[frame]
        ch_index = self.channel_name.index(channel_name)
        return self.channel_imobj[ch_index][time, :, :]