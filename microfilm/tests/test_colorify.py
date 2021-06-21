import matplotlib
import numpy as np
from microfilm import colorify


image = 100*np.ones((3,3), dtype=np.uint8)
image[0,0] = 200
image2 = 100*np.ones((3,3), dtype=np.uint8)
image2[0,0] = 180

rgb1 = np.zeros((3,3,3), dtype=np.float16)
rgb1[0,0,0] = 0.1
rgb1[0,1,0] = 0.3
rgb1[0,2,0] = 0.3

rgb2 = np.zeros((3,3,3), dtype=np.float16)
rgb2[0,0,0] = 0.2
rgb2[0,1,0] = 0.8
rgb2[0,2,1] = 0.8

def test_cmaps_def():
    
    red_map = colorify.cmaps_def('pure_red', num_colors=300, flip_map=True)
    assert isinstance(red_map, matplotlib.colors.ListedColormap), "No colormap returned"
    np.testing.assert_array_equal(red_map.colors[0], np.array([1,0,0]))
    np.testing.assert_array_equal(red_map.colors[-1], np.array([0,0,0]))
    assert len(red_map.colors) == 300, "Wrong number of colors"
    
    red_map = colorify.cmaps_def('pure_red', num_colors=300, flip_map=False)
    assert isinstance(red_map, matplotlib.colors.ListedColormap), "No colormap returned"
    np.testing.assert_array_equal(red_map.colors[0], np.array([0,0,0]))
    np.testing.assert_array_equal(red_map.colors[-1], np.array([1,0,0]))
    
def test_color_translate():
    
    assert colorify.color_translate('pure_red') == 'red', "Wrong color name"
    assert colorify.color_translate('non_existing') == 'black', "Black not returned for non-existing cmap"
    
def test_random_cmap():

    ran_cmap = colorify.random_cmap(alpha=0.3, num_colors=300)
    assert ran_cmap.colors[0,-1] == 0
    assert ran_cmap.colors[1, -1] == 0.3
    assert len(ran_cmap.colors) == 300
    
def test_colorify_by_name():

    im1 = colorify.colorify_by_name(image, 'pure_red')
    np.testing.assert_array_equal(im1[0][0], np.array([1,0,0,1]))
    
    im1 = colorify.colorify_by_name(image, 'pure_red', rescale_type='dtype')
    np.testing.assert_array_equal(im1[0][0], np.array([200/255,0,0,1]))
    np.testing.assert_array_equal(im1[0][1], np.array([100/255,0,0,1]))
    
    im1 = colorify.colorify_by_name(image, 'pure_red', rescale_type='limits', limits=[100,200])
    np.testing.assert_array_equal(im1[0][0], np.array([1,0,0,1]))
    np.testing.assert_array_equal(im1[0][1], np.array([0,0,0,1]))

def test_colorify_by_hex():
    
    im1 = colorify.colorify_by_hex(image, cmap_hex='#D53CE7')
    np.testing.assert_array_equal(im1[0][0,0:3], np.array([213, 60, 231])/255, "Not correct color returned for #D53CE7")
    
def test_rescale_image():
    
    im_resc = colorify.rescale_image(image)
    np.testing.assert_array_equal(im_resc[0], np.array([1,0,0]))
    
    im_resc = colorify.rescale_image(image, rescale_type='dtype')
    np.testing.assert_array_equal(im_resc[0], np.array([200/255,100/255,100/255]))
    
    im_resc = colorify.rescale_image(image, limits=[110, 230], rescale_type='limits')
    np.testing.assert_array_equal(im_resc[0], np.array([(200-110)/(230-110),0,0]))

def test_check_rescale_type():
    assert colorify.check_rescale_type(rescale_type='limits', limits=None) == 'min_max', "Wrong rescaling"
    assert colorify.check_rescale_type(rescale_type='limits', limits=[0,1]) == 'limits', "Wrong rescaling"
    assert colorify.check_rescale_type(rescale_type='min_max', limits=[0,1]) == 'min_max', "Wrong rescaling"
    
def test_combine_image():
    combined = colorify.combine_image([rgb1, rgb2], proj_type='max')
    np.testing.assert_almost_equal(combined[0], np.array([[0.2, 0, 0], [0.8, 0, 0], [0.3, 0.8, 0]]),decimal=3)
    combined = colorify.combine_image([rgb1, rgb2], proj_type='sum')
    np.testing.assert_almost_equal(combined[0], np.array([[0.3, 0, 0], [1, 0, 0], [0.3, 0.8, 0]]), decimal=3)

def test_multichannel_to_rgb():
    
    multic = colorify.multichannel_to_rgb(images=[image, image2], cmaps=['pure_blue', 'pure_red'],
                                          rescale_type='limits', limits=[130, 190], num_colors=1000)
    assert multic.ndim == 3, "Wrong dimensions, not RGB image"
    np.testing.assert_almost_equal(multic[:,:,0][0], np.array([(180-130)/(190-130), 0, 0]), decimal=3)
    np.testing.assert_almost_equal(multic[:,:,2][0], np.array([1, 0, 0]), decimal=3)
    
def test_check_input():
    
    im = colorify.check_input(images = np.ones((3, 20,20)))
    shapes = np.testing.assert_array_equal(np.array([x.shape for x in im]), 20*np.ones((3,2)),
                                          "3D numpy array not converted properly")
    
    im = colorify.check_input(images = [np.ones((20,20)), np.ones((20,20))])
    shapes = np.testing.assert_array_equal(np.array([x.shape for x in im]), 20*np.ones((2,2)),
                                          "list of 2D arrays not converted properly")
    
    im = colorify.check_input(images = np.ones((20,20)))
    shapes = np.testing.assert_array_equal(np.array([x.shape for x in im]), 20*np.ones((1,2)),
                                          "single 2d array not converted properly")
    
 