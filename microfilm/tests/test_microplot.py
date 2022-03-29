import os
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from microfilm import microplot, colorify

image = 100*np.ones((3,3), dtype=np.uint8)
image[0,0] = 200
image2 = 100*np.ones((3,3), dtype=np.uint8)
image2[0,1] = 180

more_than_3d = np.zeros((5,3,3), dtype=np.uint8)
more_than_3d[0,0,0] = 1
more_than_3d[1,1,0] = 1
more_than_3d[2,2,0] = 1
more_than_3d[3,1,1] = 1
more_than_3d[4,2,1] = 1


def verify_image(microim):
    
    # check image
    assert np.any(microim.ax.get_images()[0].get_array()[:,:,0] > 0) == False, "Red should not be present"
    np.testing.assert_array_equal(microim.ax.get_images()[0].get_array()[:,:,1], np.array([[0,1,0], [0,0,0], [0,0,0]]), "Green channel not correct")
    np.testing.assert_array_equal(microim.ax.get_images()[0].get_array()[:,:,2], np.array([[1,0,0], [0,0,0], [0,0,0]]), "Blue channel not correct")

def verify_label(microim):
    
    assert microim.ax.texts[0].get_text() == 'b', "Wrong channel label"
    assert microim.ax.texts[1].get_text() == 'a', "Wrong channel label"    
    
def test_microshow():

    microim = microplot.microshow(
        images=[image, image2], cmaps=['pure_blue', 'pure_green'], channel_names=['a', 'b'], channel_label_show=True, unit='mm',
        scalebar_unit_per_pix=0.5, scalebar_size_in_units=1, scalebar_thickness=0.1, scalebar_color='red',
        label_text='A', label_color='pink')

    assert isinstance(microim, microplot.Microimage)

    # check image
    verify_image(microim)
    
    # check channel labels
    verify_label(microim)

    # check scalebar
    assert microim.ax.artists[0].units == 'mm', "Wrong scalebar unit"
    assert microim.ax.artists[0].color == 'red', "Wrong scalebar color"
    assert microim.ax.artists[0].width_fraction == 0.1, "Wrong scalebar width"

    # check label
    assert microim.ax.texts[2].get_text() == 'A', "Wrong label"
    assert microim.ax.texts[2].get_color() == 'pink', "Wrong label color"

def test_default_random_gradient():
    # test that images with > 3 channels use random gradient by default
    microim = microplot.microshow(more_than_3d)
    assert microim.cmaps[4] == "ran_gradient", "Random gradient not select for d>3"

def test_mixed_cmaps():
    # test that "name" cmaps and "object" cmaps can be mixed
    summer_cmap = colorify.cmaps_def(cmap_name='summer')
    microim = microplot.microshow(
        images=[image, image2], cmaps=[summer_cmap, 'pure_blue'])
    
    assert isinstance(microim.cmap_objects[0], matplotlib.colors.LinearSegmentedColormap), "Wrong colormap for summer cmap"
    assert isinstance(microim.cmap_objects[1], matplotlib.colors.ListedColormap), "Wrong colormap for pure_blue cmap"

def test_add_scalebar():
    
    microim = microplot.microshow(
        images=[image, image2], cmaps=['pure_blue', 'pure_green'])
    
    microim.add_scalebar(unit='mm', scalebar_unit_per_pix=0.5, scalebar_size_in_units=1, scalebar_thickness=0.1, scalebar_color='red')
    
    # check scalebar
    assert microim.ax.artists[0].units == 'mm', "Wrong scalebar unit"
    assert microim.ax.artists[0].color == 'red', "Wrong scalebar color"
    assert microim.ax.artists[0].width_fraction == 0.1, "Wrong scalebar width"
    
def test_add_label():
    
    microim = microplot.microshow(
        images=[image, image2], cmaps=['pure_blue', 'pure_green'])
    microim.add_label(label_text='A', label_color='pink')
    # check label
    assert microim.ax.texts[0].get_text() == 'A', "Wrong label"
    assert microim.ax.texts[0].get_color() == 'pink', "Wrong label color"
    
def test_add_channel_labels():
    
    microim = microplot.microshow(
        images=[image, image2], cmaps=['pure_blue', 'pure_green'])
    
    # check channel labels
    microim.add_channel_labels(channel_names=['a', 'b'])
    verify_label(microim)
    
    assert microim.ax.texts[1].get_color() == (0.0, 0.0, 1.0, 1.0), "Wrong label color"
    assert microim.ax.texts[0].get_color() == (0.0, 1.0, 0.0, 1.0), "Wrong label color"
    
def test_update():
    
    microimage = microplot.Microimage(images=[image, image2], cmaps=['pure_blue', 'pure_green'])
    assert microimage.ax is None

    fig, ax = plt.subplots(1, 2)
    microimage.update(ax[1])
    
    verify_image(microimage)
    
def test_save():
    microimage = microplot.microshow(images=[image, image2], cmaps=['pure_blue', 'pure_green'])
    microimage.savefig('test_saveimage.png')
    os.path.isfile('test_saveimage.png')
    os.remove('test_saveimage.png')
    
def test_micropanel():
    
    microimage1 = microplot.Microimage(images=[image, image2], cmaps=['pure_blue', 'pure_green'])
    microimage2 = microplot.Microimage(images=[image, image2], cmaps=['pure_cyan', 'pure_magenta'])

    micropanel = microplot.Micropanel(1, 2)
    assert isinstance(micropanel, microplot.Micropanel)

    micropanel.add_element([0,0], microimage1)
    micropanel.add_element([0,1], microimage2)
    
    # check grid shape
    micropanel.microplots.shape == (1,2)
    
    # Check that plots are in the correct place
    np.testing.assert_array_equal(micropanel.microplots[0,0].ax.get_images()[0].get_array()[:,:,0], np.array([[0,0,0], [0,0,0], [0,0,0]]),
                              "Red channel in first panel not correct")
    np.testing.assert_array_equal(micropanel.microplots[0,0].ax.get_images()[0].get_array()[:,:,1], np.array([[0,1,0], [0,0,0], [0,0,0]]),
                                  "Green channel in first panel not correct")
    np.testing.assert_array_equal(micropanel.microplots[0,0].ax.get_images()[0].get_array()[:,:,2], np.array([[1,0,0], [0,0,0], [0,0,0]]),
                                  "Blue channel in first panel not correct")
    np.testing.assert_array_equal(micropanel.microplots[0,1].ax.get_images()[0].get_array()[:,:,0], np.array([[0,1,0], [0,0,0], [0,0,0]]),
                                  "Red channel in second panel not correct")
    np.testing.assert_array_equal(micropanel.microplots[0,1].ax.get_images()[0].get_array()[:,:,1], np.array([[1,0,0], [0,0,0], [0,0,0]]),
                                  "Green channel in second panel not correct")
    np.testing.assert_array_equal(micropanel.microplots[0,1].ax.get_images()[0].get_array()[:,:,2], np.array([[1,1,0], [0,0,0], [0,0,0]]),
                                  "Blue channel in second panel not correct")
    
    # check labels and their positions
    micropanel.add_channel_label()
    
    assert micropanel.fig.texts[0].get_text() == 'Channel-1', "Wrong channel label"
    assert micropanel.fig.texts[1].get_text() == 'Channel-0', "Wrong channel label"
    assert micropanel.fig.texts[2].get_text() == 'Channel-1', "Wrong channel label"
    assert micropanel.fig.texts[3].get_text() == 'Channel-0', "Wrong channel label"

    assert micropanel.fig.texts[0].get_position()[1] > 0.8, "Wrong y position for first Channel-1 label"
    assert micropanel.fig.texts[1].get_position()[1] > 0.9, "Wrong y position for first Channel-0 label"
    assert micropanel.fig.texts[2].get_position()[1] > 0.8, "Wrong y position for second Channel-1 label"
    assert micropanel.fig.texts[3].get_position()[1] > 0.9, "Wrong y position for second Channel-0 label"

    assert micropanel.fig.texts[0].get_position()[0] < 0.5, "Wrong x position for first Channel-1 label"
    assert micropanel.fig.texts[1].get_position()[0] < 0.5, "Wrong x position for first Channel-0 label"
    assert micropanel.fig.texts[2].get_position()[0] > 0.5, "Wrong x position for second Channel-1 label"
    assert micropanel.fig.texts[3].get_position()[0] > 0.5, "Wrong x position for second Channel-0 label"

def test_savepanel():
    
    microimage1 = microplot.Microimage(images=[image, image2], cmaps=['pure_blue', 'pure_green'])
    microimage2 = microplot.Microimage(images=[image, image2], cmaps=['pure_cyan', 'pure_magenta'])

    micropanel = microplot.Micropanel(1, 2)
    micropanel.add_element([0,0], microimage1)
    micropanel.add_element([0,1], microimage2)
    
    micropanel.savefig('test_savepanel.png')
    os.path.isfile('test_savepanel.png')
    os.remove('test_savepanel.png')
