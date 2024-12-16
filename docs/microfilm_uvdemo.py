# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "microfilm",
#     ]
# ///


def main() -> None:
    import webbrowser
    from pathlib import Path
    import skimage
    from microfilm import microplot
    from microfilm.microplot import Micropanel

    image = skimage.io.imread('https://raw.githubusercontent.com/guiwitz/microfilm/refs/heads/master/demodata/coli_nucl_ori_ter.tif')

    multi_channel = image[:,10,:,:]

    # create images
    microim = microplot.microshow(
        images=multi_channel,
        cmaps=['cyan', 'magenta', 'yellow'])
    
    microim2 = microplot.microshow(
        images=multi_channel[0], 
        cmaps=['vispy:husl'])

    microim3 = microplot.microshow(
        images=multi_channel[1:3], 
        cmaps=['chrisluts:bop_orange', 'chrisluts:bop_purple'])
        
    # create panel
    microim.add_label('A', label_location='upper left', label_color='white', label_font_size=20)
    microim2.add_label('B', label_location='upper left', label_color='black', label_font_size=20)
    microim3.add_label('C', label_location='upper left', label_color='white', label_font_size=20)

    micropanel = Micropanel(1,3)
    micropanel.add_element([0,0], microim, copy=True)
    micropanel.add_element([0,1], microim2, copy=True)
    micropanel.add_element([0,2], microim3, copy=True)

    # add channel labels
    micropanel.add_channel_label(channel_names=
                                 [[['cyan', 'magenta', 'yellow'],
                                  ['vispy:husl'],
                                  ['chrisluts:bop_orange', 'chrisluts:bop_purple']]])
                                  
    # save images
    microim3.savefig('microfilm_demo.png')
    micropanel.savefig('micropanel_demo.png')

    # open images
    mypath = Path('microfilm_demo.png').absolute()
    webbrowser.open(f"file://{mypath}")

    mypath = Path('micropanel_demo.png').absolute()
    webbrowser.open(f"file://{mypath}")


if __name__ == "__main__":
    main()
