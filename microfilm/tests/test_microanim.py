import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from microfilm import microanim, dataset


test_image = np.zeros((3, 4, 5, 5), dtype=np.uint8)
test_image[0,0,:,:] = 100
test_image[1,1,:,:] = 100
test_image[2,2,:,:] = 100

data_np = dataset.Nparray(test_image)

anim = microanim.Microanim(data=data_np, cmaps=['pure_red', 'pure_green', 'pure_blue'])
anim2 = microanim.Microanim(data=data_np, cmaps=['pure_cyan', 'pure_magenta', 'pure_yellow'])

anim_panel = microanim.Microanimpanel(rows=1, cols=2)
anim_panel.add_element([0,0], anim)
anim_panel.add_element([0,1], anim2)  

def verify_colors(anim, color, t):
    
    if color == 'red':
        r,g,b = [np.ones((5,5)), np.zeros((5,5)), np.zeros((5,5))]
    elif color == 'green':
        r,g,b = [np.zeros((5,5)), np.ones((5,5)), np.zeros((5,5))]
    elif color == 'blue':
        r,g,b = [np.zeros((5,5)), np.zeros((5,5)), np.ones((5,5))]
    elif color == 'cyan':
        r,g,b = [np.zeros((5,5)), np.ones((5,5)), np.ones((5,5))]
    elif color == 'magenta':
        r,g,b = [np.ones((5,5)), np.zeros((5,5)), np.ones((5,5))]
    elif color == 'yellow':
        r,g,b = [np.ones((5,5)), np.ones((5,5)), np.zeros((5,5))]
        
    np.testing.assert_array_equal(anim.ax.get_images()[0].get_array()[:,:,0], r, f"at time point {t} color should be {color}; wrong red channel")
    np.testing.assert_array_equal(anim.ax.get_images()[0].get_array()[:,:,1], g, f"at time point {t} color should be {color}; wrong green channel")
    np.testing.assert_array_equal(anim.ax.get_images()[0].get_array()[:,:,2], b, f"at time point {t} color should be {color}; wrong blue channel")

def test_microanim_obj():
    
    assert isinstance(anim, microanim.Microanim), "Microanim class not instantiated"
    
    # check that time is computed
    assert anim.max_time == 3, "Max time not properly computed"
    

def test_add_time_stamp():
    
    anim.add_time_stamp(unit='T', unit_per_frame='3', location='lower left')
    assert anim.ax.texts[0].get_text() == '00:00:00', "Timer not properly set initially"
    
def test_update_animation():
    
    anim.update_animation(0)
    verify_colors(anim, 'red', 0)

    anim.update_animation(1)
    # check that timer gets updated
    assert anim.ax.texts[0].get_text() == '00:03:00', "Timer not properly updating"
    
    verify_colors(anim, 'green', 1)

    
def test_time_slider():
    
    anim.time_slider.value = 2
    assert anim.ax.texts[0].get_text() == '00:06:00', "Timer not properly updating when time slider updates"

    verify_colors(anim, 'blue', 2)
    
def test_save():
    
    anim_save = microanim.Microanim(data=data_np, cmaps=['pure_red', 'pure_green', 'pure_blue'])
    anim_save.fig.draw(anim_save.fig.canvas.get_renderer())
    anim_save.save_movie('test_movie.mp4')
    os.path.isfile('test_movie.mp4')
    os.remove('test_movie.mp4')

def test_animationpanel():

    assert isinstance(anim_panel, microanim.Microanimpanel), "Microanimpanel class not instantiated"
    
    # check grid shape
    anim_panel.microanims.shape == (1,2)
    
    # check position of images
    verify_colors(anim_panel.microanims[0,0], 'red', 0)
    verify_colors(anim_panel.microanims[0,1], 'cyan', 0)
    
def test_animationpanel_update_animation():
    
    anim_panel.update_animation(1)
    
    # check that update happened
    verify_colors(anim_panel.microanims[0,0], 'green', 0)
    verify_colors(anim_panel.microanims[0,1], 'magenta', 0)
    
def test_animationpanel_add_channel_label():
    
    anim_panel.add_channel_label()
    
    assert anim_panel.fig.texts[0].get_text() == 'Channel-2', "Wrong channel label"
    assert anim_panel.fig.texts[1].get_text() == 'Channel-1', "Wrong channel label"
    assert anim_panel.fig.texts[2].get_text() == 'Channel-0', "Wrong channel label"
    assert anim_panel.fig.texts[3].get_text() == 'Channel-2', "Wrong channel label"
    assert anim_panel.fig.texts[4].get_text() == 'Channel-1', "Wrong channel label"
    assert anim_panel.fig.texts[5].get_text() == 'Channel-0', "Wrong channel label"

    assert anim_panel.fig.texts[0].get_position()[1] > 0.8, "Wrong y position for first Channel-2 label"
    assert anim_panel.fig.texts[0].get_position()[0] < 0.5, "Wrong x position for first Channel-2 label"
    assert anim_panel.fig.texts[3].get_position()[1] > 0.8, "Wrong y position for second Channel-2 label"
    assert anim_panel.fig.texts[3].get_position()[0] > 0.5, "Wrong x position for second Channel-2 label"
    
    
def test_animationpanel_time_slider():

    anim_panel.microanims[0,0].add_time_stamp(unit='T', unit_per_frame='3', location='lower left')

    anim_panel.time_slider.value = 2
    assert anim_panel.microanims[0,0].ax.texts[0].get_text() == '00:06:00', "Panel Timer not properly updating when time slider updates"

    verify_colors(anim_panel.microanims[0,0], 'blue', 2)
    verify_colors(anim_panel.microanims[0,1], 'yellow', 2)
    
def test_animationpanel_save():
    
    anim_panel_save = microanim.Microanimpanel(rows=1, cols=2)
    anim_panel_save.add_element([0,0], anim)
    anim_panel_save.add_element([0,1], anim2) 
    
    anim_panel_save.save_movie('test_savepanelmovie.mp4')
    os.path.isfile('test_savepanelmovie.mp4')
    os.remove('test_savepanelmovie.mp4')