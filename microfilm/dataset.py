import os
import re
from aicsimageio import AICSImage
from nd2reader import ND2Reader
import h5py
import skimage.io
import numpy as np
from pathlib import Path


class Data:
    """
    Class defining and handling datasets. Given an experiment directory
    (tiff files or stacks) or a a file (ND2) the available data are
    automatically parsed. Parameters specific to an analysis run such as
    bad frame indices or the time steps are also stored in the Data object.

    Parameters
    ----------
    expdir: str
        path to data folder (tif) or file (ND2)
    signal_name: list of str
        names of data to use as signals
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
    signalfile: str or list
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
        signal_name=None,
        bad_frames=[],
        step=1,
        max_time=None,
        data_type=None,
    ):

        self.data_type = data_type
        self.expdir = Path(expdir)
        self.signal_name = signal_name
        self.bad_frames = np.array(bad_frames)
        self.step = step
        self.max_time = max_time

        self.dims = None
        self.signalfile = None
        self.data_type = data_type

    def set_valid_frames(self):
        """Create a list of indices of valid frames"""

        self.valid_frames = np.arange(self.max_time)
        self.valid_frames = self.valid_frames[
            ~np.in1d(self.valid_frames, self.bad_frames)
        ]
        self.valid_frames = self.valid_frames[:: self.step]
        self.K = len(self.valid_frames)

    def find_files(self, folderpath):
        """Given a folder, parse contents to find all time points"""

        image_names = os.listdir(folderpath)
        image_names = np.array([x for x in image_names if x[0] != "."])
        if len(image_names) > 0:
            times = [re.findall(".*\_t*(\d+)\.(?:tif|TIF|tiff)", x) for x in image_names]
            times = [int(x[0]) for x in times if len(x) > 0]
            if len(times) < len(image_names):
                raise Exception(f"Sorry, file names could not be parsed.\
                    Make sure they conform to XXX_txxx.tif/TIF/tiff")
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

        assert (channel < len(self.signalfile)), f"Only {len(self.signalfile)} channels available."
        assert (frame < len(self.valid_frames)), f"Only {len(self.valid_frames)} frames available."

    def load_frame_signal(self, m, k):
        """Load index k of valid frames of channel index m in self.signalfile"""
        raise NotImplementedError

    def frame_generator(self, channel):
        
        for t in self.valid_frames:
            image = self.load_frame_signal(channel, t)
            yield image

    def get_channel_name(self, m):
        """Get name of channel index m"""

        return self.signal_name[m]

class TIFFSeries(Data):
    def __init__(
        self,
        expdir,
        signal_name=None,
        bad_frames=[],
        step=1,
        max_time=None,
        data_type="series",
    ):
        super().__init__(
            expdir,
            signal_name,
            bad_frames,
            step,
            max_time,
            data_type,
        )

        self.initialize()

    def initialize(self):

        # if no signal names are provided, consider all folders as signal
        if self.signal_name is None:
            self.signal_name = []
            for f in self.expdir.glob('*'):
                if f.is_dir():
                    self.signal_name.append(f.name)
        if len(self.signal_name) == 0:
            raise Exception(f"Sorry, no folders found in {self.expdir}")

        self.signalfile = [
                self.find_files(os.path.join(self.expdir, x)) for x in self.signal_name
            ]

        if self.max_time is None:
            self.max_time = len(self.signalfile[0])
            # print(self.max_time)

        self.set_valid_frames()

        image = self.load_frame_signal(0, 0)
        self.dims = image.shape
        self.shape = image.shape

    def load_frame_signal(self, m, k):
        """Load index k of valid frames of channel index m in self.signalfile"""

        self.check_channel_time_available(m, k)

        time = self.valid_frames[k]
        full_path = os.path.join(
            self.expdir, self.signal_name[m], self.signalfile[m][time]
        )
        return skimage.io.imread(full_path).astype(dtype=np.uint16)


class MultipageTIFF(Data):
    def __init__(
        self,
        expdir,
        signal_name=None,
        bad_frames=[],
        step=1,
        max_time=None,
        data_type="multi",
    ):
        super().__init__(
            expdir,
            signal_name,
            bad_frames,
            step,
            max_time,
            data_type,
        )

        self.initialize()

    def initialize(self):
        
        # if no signal names are provided, consider all folders as signal
        if self.signal_name is None:
            self.signal_name = []
            for ext in ('*.tif', '*.TIFF', '*.tiff'):
                files = self.expdir.glob(ext)
                for f in files:
                    self.signal_name.append(f.name)
        if len(self.signal_name) == 0:
            raise Exception(f"Sorry, no tif/tiff/TIFF files found in {self.expdir}")

        self.signalfile = self.signal_name

        self.signal_imobj = [
            AICSImage(os.path.join(self.expdir, x), known_dims="TYX")
            for x in self.signal_name
        ]

        if self.max_time is None:
            self.max_time = self.signal_imobj[0].size_t

        self.set_valid_frames()

        image = self.load_frame_signal(0,0)
        self.dims = image.shape
        self.shape = image.shape

    def load_frame_signal(self, m, k):
        """Load index k of valid frames of channel index m in self.signalfile"""

        self.check_channel_time_available(m, k)

        time = self.valid_frames[k]

        image = self.signal_imobj[m].get_image_data("YX", S=0, T=time, C=0, Z=0)
        return image.astype(dtype=np.uint16)



class ND2(Data):
    def __init__(
        self,
        expdir,
        signal_name=None,
        bad_frames=[],
        step=1,
        max_time=None,
        data_type="nd2",
    ):
        super().__init__(
            expdir,
            signal_name,
            bad_frames,
            step,
            max_time,
            data_type,
        )

        self.initialize()

    def initialize(self):
        
        self.nd2file = ND2Reader(self.expdir.as_posix())
        self.nd2file.metadata["z_levels"] = range(0)

        if self.signal_name is None:
            self.signal_name = list(self.nd2file.metadata["channels"])
        self.signalfile = self.signal_name


        if self.max_time is None:
            self.max_time = self.nd2file.sizes["t"]

        self.set_valid_frames()

        image = self.load_frame_signal(0, 0)
        self.dims = image.shape
        self.shape = image.shape

    def load_frame_signal(self, m, k):
        """Load index k of valid frames of channel index m in self.signalfile"""

        self.check_channel_time_available(m, k)

        time = self.valid_frames[k]

        ch_index = self.nd2file.metadata["channels"].index(self.signalfile[m])
        image = self.nd2file.get_frame_2D(x=0, y=0, z=0, c=ch_index, t=time, v=0)
        return image


class H5(Data):
    def __init__(
        self,
        expdir,
        signal_name=None,
        bad_frames=[],
        step=1,
        max_time=None,
        data_type="h5",
    ):
        super().__init__(
            expdir,
            signal_name,
            bad_frames,
            step,
            max_time,
            data_type,
        )

        self.initialize()

    def initialize(self):

        if self.signal_name is None:
            self.signal_name = []
            for ext in ('*.h5','*.H5'):
                files = self.expdir.glob(ext)
                for f in files:
                    self.signal_name.append(f.name)
        if len(self.signal_name) == 0:
            raise Exception(f"Sorry, no tif/tiff/TIFF files found in {self.expdir}")

        self.signalfile = self.signal_name
        
        self.signal_imobj = [
            h5py.File(os.path.join(self.expdir, x), "r").get("volume")
            for x in self.signal_name
        ]

        if self.max_time is None:
            self.max_time = self.signal_imobj[0].shape[0]

        self.set_valid_frames()

        image = self.load_frame_signal(0, 0)
        self.dims = image.shape
        self.shape = image.shape

    def load_frame_signal(self, m, k):
        """Load index k of valid frames of channel index m in self.signalfile"""

        self.check_channel_time_available(m, k)

        time = self.valid_frames[k]

        return self.signal_imobj[m][time, :, :]