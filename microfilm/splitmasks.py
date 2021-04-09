import skimage.measure
import skimage.io
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


def get_roi_cm(roi_path=None, roi_im=None):
    """
    This is my function.

    Parameters
    ----------
    roi_path: str
        path to roi image

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
    
    yy, xx = np.meshgrid(np.arange(im_dims[0]),np.arange(im_dims[1]))
    roi_mask = np.zeros(im_dims, dtype=np.bool_)
    concentric_masks = [roi_mask]

    for ind, i in enumerate(np.arange(sector_width, sector_width*num_sectors+1, sector_width)):

        temp_roi = np.sqrt((xx - center[0])**2 + (yy - center[1])**2) < i
        concentric_masks.append((ind+1)*(temp_roi*~roi_mask))
        roi_mask = temp_roi

    concentric_labels = np.sum(np.array(concentric_masks),axis=0)
    
    return concentric_labels

def create_sector_mask(center, im_dims, angular_width=20, max_rad=50):
    """
    Create a labelled mask of disk split in angular sectors.

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

    Returns
    -------
    sector_labels: 2d array
        labelled image with disk split in sectors

    """

    yy, xx = np.meshgrid(np.arange(im_dims[0]),np.arange(im_dims[1]))
    angles = np.arctan2(xx-center[0],yy-center[1])
    angles %= (2*np.pi)
    rad_mask = np.sqrt((xx - center[0])**2 + (yy - center[1])**2) < max_rad
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

def measure_intensities(time_image, im_labels, max_time):
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

    for im_np in time_image:
        #im_np = nd2_image.get_frame_2D(x=0, y=0, c=channel, t=t)
        measures.append(skimage.measure.regionprops_table(im_labels, 
                                                          intensity_image=im_np, properties=('label','mean_intensity')))

    signal = np.stack([x['mean_intensity'] for x in measures],axis=0)
    
    return signal
    
