from microfilm import dataset
import numpy as np

data_tiffseries = dataset.TIFFSeries('microfilm/dataset/tests/test_folders/test_tifseries_good/')
data_multipagetiff = dataset.MultipageTIFF('microfilm/dataset/tests/test_folders/test_multipage_good/')
data_nd2 = dataset.ND2('microfilm/dataset/tests/test_folders/test_nd2_good/cluster.nd2')
data_h5 = dataset.H5('microfilm/dataset/tests/test_folders/test_h5_good/')

all_data = {'tiffseries': data_tiffseries,
            'multipagetiff': data_multipagetiff,
            'nd2': data_nd2,
            'h5': data_h5
           }

max_time_points = {'tiffseries': 51,
                   'multipagetiff': 51,
                   'nd2': 3,
                   'h5': 51
                  }

channel_names = {'tiffseries': ['channel1', 'channel1'],
                   'multipagetiff': ['C2-MAX_mitosis.tif', 'C1-MAX_mitosis.tif'],
                   'nd2': ['5-FAM/pH 9.0', 'FM 4-64/2% CHAPS'],
                   'h5': ['C1-MAX_mitosis.h5', 'C2-MAX_mitosis.h5']
                  }

array_shape = {'tiffseries': [196, 171],
                   'multipagetiff': [196, 171],
                   'nd2': [31, 38],
                   'h5': [196, 171]
                  }


def test_find_files():
    
    for datatype in all_data:
        assert len(all_data[datatype].channel_name) == 2, f"wrong number of channels in {datatype}"
        
def test_channel_names():
    
    for datatype in all_data:
        for ch in channel_names[datatype]:
            assert ch in all_data[datatype].channel_name, f"{ch} not in channel list of {datatype}"
            
def test_max_time():
    
    for datatype in all_data:
        assert all_data[datatype].max_time == max_time_points[datatype], f"wrong number of time points in {datatype}"
        
def test_image_size():
    for datatype in all_data:
        assert np.array_equal(np.array(all_data[datatype].load_frame(all_data[datatype].channel_name[0],0).shape),
                              np.array(array_shape[datatype])), f"wrong image dimension in {datatype}"