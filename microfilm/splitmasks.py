import itertools

import skimage.measure
import skimage.io
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd


def get_roi_cm(roi_path=None, roi_im=None):
    """
    Find the center of mass of a roi defined as a single label in an image.

    Parameters
    ----------
    roi_path: str
        path to roi image
    roi_im = 2d array
        image containing roi

    Returns
    -------
    cm: 1d array
        center of mass of roi

    """
    
    if roi_path is None:
        if roi_im is None:
            raise Exception("You have to provide an image if you don't provide a path")              
    else:
        roi_im = skimage.io.imread(roi_path)
    
    roi_props = skimage.measure.regionprops_table(roi_im, properties=('label','centroid'))
    cm = [int(roi_props['centroid-0']),int(roi_props['centroid-1'])]

    return cm

def create_concentric_mask(center, im_dims, sector_width=10, num_sectors=10):
    """
    Create a labelled mask of disk split in concentric rings.

    Parameters
    ----------
    center: list
        2d position of center of disk
    im_dims: list
        image size
    sector_width: int
        ring thickness
    num_sectors: int
        number of rings to define

    Returns
    -------
    concentric_labels: 2d array
        labelled image with concentric rings

    """
    
    yy, xx = np.meshgrid(np.arange(im_dims[1]),np.arange(im_dims[0]))
    roi_mask = np.zeros(im_dims, dtype=np.bool_)
    concentric_masks = [roi_mask]

    for ind, i in enumerate(np.arange(sector_width, sector_width*num_sectors+1, sector_width)):

        temp_roi = np.sqrt((xx - center[0])**2 + (yy - center[1])**2) < i
        concentric_masks.append((ind+1)*(temp_roi*~roi_mask))
        roi_mask = temp_roi

    concentric_labels = np.sum(np.array(concentric_masks),axis=0)
    
    return concentric_labels

def create_sector_mask(center, im_dims, angular_width=20, max_rad=50, ring_width=None):
    """
    Create a labelled mask of a disk or ring split in angular sectors. If radius is
    provided, the mask is a ring otherwise a disk.

    Parameters
    ----------
    center: list
        2d position of center of disk
    im_dims: list
        image size
    angular_width: int
        size of angular sectors in degrees
    max_rad: float
        disk radius
    ring_width: int
        ring width


    Returns
    -------
    sector_labels: 2d array
        labelled image with disk split in sectors

    """

    yy, xx = np.meshgrid(np.arange(im_dims[1]),np.arange(im_dims[0]))
    angles = np.arctan2(xx-center[0],yy-center[1])
    angles %= (2*np.pi)
    rad_mask = np.sqrt((xx - center[0])**2 + (yy - center[1])**2)
    
    if ring_width is None:
        rad_mask = rad_mask < max_rad
    else:
        rad_mask = (rad_mask < max_rad) & (rad_mask > max_rad-ring_width)

    sector_masks = [rad_mask*(ind+1)*((angles >= np.deg2rad(i)) *(angles < np.deg2rad(i+angular_width)))
                    for ind, i in enumerate(np.arange(0,360-angular_width+1,angular_width))]
    sector_labels = np.sum(np.array(sector_masks),axis=0)

    return sector_labels


def get_cmap_labels(im_label, cmap_name='cool'):
    """Create list of L colors where L is the number of labels in the image"""

    cmap_original = plt.get_cmap(cmap_name)
    colors = cmap_original(np.linspace(0,1,im_label.max()))
    cmap = matplotlib.colors.ListedColormap(colors)
    
    return colors, cmap
    
def nan_labels(im_label):
    """Return a label image where background is set to nan for transparency"""
    
    im_label_nan = im_label.astype(float)
    im_label_nan[im_label_nan==0] = np.nan
    
    return im_label_nan

def measure_intensities(time_image, im_labels, min_time=0, max_time=None, step=1):
    """
    Measure average intensity in a time-lapse image using a labelled image

    Parameters
    ----------
    time_image: iterable 2d array
        can be a TxHxW numpy array or iterable generating HxW numpy arrays
    im_labels: 2d array
        labelled image
    max_time: int
        last time point to consider

    Returns
    -------
    signal: 2d array
        TxL array with mean intensity for each time point T in each labelled
        region L

    """

    measures = []
    time_image_part = itertools.islice(time_image, min_time, max_time, step)

    for im_np in time_image_part:
        measures.append(skimage.measure.regionprops_table(im_labels, 
                                                          intensity_image=im_np, properties=('label','mean_intensity')))

    if im_np.ndim == 3:
        signals = [np.stack([x['mean_intensity-'+str(k)] for x in measures],axis=0) for k in range(im_np.shape[2])]
        signal = np.stack(signals, axis=2)
    else:
        signal = np.stack([x['mean_intensity'] for x in measures],axis=0)
    
    return signal

def plot_signals(signal, color_array=None, ax=None):
    """
    Plot extracted signal with specific colormap
    
    Parameters
    ----------
    signal: 2d array
        signal array with rows as time points and columns as sectors
    color_array: 2d array
        N x 4 array of RGBA colors where N >= sectors
    ax: Matplotlib axis

    Returns
    -------
    ax if ax passed as input otherwise fig
    
    """

    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(6,6))
    for i in range(signal.shape[1]):
        if color_array is not None:
            ax.plot(signal[:,i], color=color_array[i])
        else:
            ax.plot(signal[:,i])
    ax.set_xlabel('Time', fontsize=20)
    ax.set_ylabel('Intensity', fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=15)
    
    if fig is not None:
        fig.tight_layout()
        return fig
    
    return ax

def plot_sectors(image, sectors, channel=None, time=0, cmap=None, im_cmap=None, ax=None):
    """
    Plot image and overlayed sectors with a given colormap
    
    Parameters
    ----------
    image: dataset object
        image to be plotted
    sectors: 2d array
        labelled image of sectors
    channel: str
        name of channel to plot
    time: int
        frame to plot
    cmap: Matplotlib colormap
        colormap for split mask
    im_cmap: Matplotlib colormap
        colormap for image
    ax: Matplotlib axis

    Returns
    -------
    ax if ax passed as input otherwise fig
    
    """
    
    if im_cmap is None:
        im_cmap = plt.get_cmap('gray')

    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(6,6))
    if channel is None:
        channel = image.channel_name[0]

    sector_labels_nan = nan_labels(sectors)

    ax.imshow(image.load_frame(channel,0), cmap=im_cmap)
    ax.imshow(sector_labels_nan, cmap=cmap, interpolation='none', alpha=0.5)
    if fig is not None:
        fig.tight_layout()
        return fig
    
    return ax

def save_signal(signal, name='mycsv.csv',format='long', channels=None):
    """
    Save the sector signal in a CSV file with a given name

    Parameters
    ----------
    signal: array
        signal array with rows as time points and columns as sectors
        a third dimension indicates multiple channels
    name: str
        file name for export (should end in .csv)
    format: str
        'long' or 'wide'
        in 'long' format, the table has three columns, time/sector/intensity
        in 'wide' format, rows correspond to time and columns to sectors

    Returns
    -------
    ax if ax passed as input otherwise fig
    
    """
    
    if signal.ndim == 3:
        format = 'long'
        if channels is None:
            channels = ['channel-'+str(i) for i in range(signal.shape[2])]

    def reshape_df(signal_array, int_name='intensity'):
        df = pd.DataFrame(signal_array)
        df = df.reset_index().rename({'index': 'time'}, axis='columns')
        df = pd.melt(df, id_vars='time', var_name='sector', value_name=int_name)
        return df

    if format == 'wide':
        signal_df = pd.DataFrame(signal)
        signal_df.to_csv(name, index=False)
    elif format == 'long':
        if signal.ndim == 2:
            signal_df = reshape_df(signal)
        else:
            dfs = []
            for k in range(signal.shape[2]):
                temp_df = reshape_df(signal[:, :, k], int_name='intensity')
                temp_df['channel'] = channels[k]
                dfs.append(temp_df)
            signal_df = pd.concat(dfs)
            
        signal_df.to_csv(name, index=False)