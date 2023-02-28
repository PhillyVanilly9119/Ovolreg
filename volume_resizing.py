import os
import cv2
import glob
import numpy as np
from tqdm.auto import tqdm
import matplotlib.pyplot as plt


def resize_volme(path_loading: str, in_dims: tuple, out_dims: tuple, noise_crop: float):
    with open(path_loading) as f:
        vol = np.fromfile(f, dtype=np.uint8).reshape(in_dims)
    out = np.zeros(out_dims)
    for i in range(vol.shape[-1]):
        out[...,i] = cv2.resize(vol[...,i], (out_dims[1],out_dims[0]))            
    # print(out.shape) #debug
    out = out-(noise_crop*out.max())
    out[out<0] = 0
    return out


def save_resized_vol(vol: np.array, path_to_volume: str) -> None:
    dims = vol.shape
    dims_str = str(dims[0]) + 'x' + str(dims[1]) + 'x' + str(dims[2])
    tail, head = os.path.split(path_to_volume)
    tmp = head.split('_')
    tmp = tmp[:-1]
    tmp = ''.join([e + '_' for e in tmp])
    full_path_saving = os.path.join(tail, 'resized', tmp + dims_str + '.bin')
    print(full_path_saving) 
    

def average_volumes_from_list(vol_list: list, dims: tuple) -> np.array:
    out_vol = np.zeros((*dims,len(vol_list)), dtype=np.uint8)
    for i, vol in tqdm(enumerate(vol_list)):
        with open(vol) as f:
            out_vol[...,i] = np.fromfile(f, dtype=np.uint8).reshape(dims)
    return np.mean(out_vol, axis=-1)


def generate_enface_images(vol: np.array) -> np.array:
    pass
    # enface_stack


def calculate_snr_for_nVols(vol_list: list, dims: tuple) -> np.array:
    snr = []
    out_vol = np.zeros((*dims,len(vol_list)), dtype=np.uint8)
    for i, vol in tqdm(enumerate(vol_list)):
        with open(vol) as f:
            out_vol[...,i] = np.fromfile(f, dtype=np.uint8).reshape(dims)
    return np.mean(out_vol, axis=-1)
    

def normalize(x: np.array) -> np.array:
    return (x-x.max()/(x.max()-x.min()))


def find_center_of_mass(path_loading: str, in_dims: tuple):
    with open(path_loading) as f:
        vol = np.fromfile(f, dtype=np.uint8).reshape(in_dims)
        a_scan = vol[:,200,200].astype(np.float32)
    plt.plot(normalize(a_scan))
    # plt.plot((c_of_mass - c_of_mass.min())/(c_of_mass.max() - c_of_mass.min()))
    plt.show()
    
    
if __name__ == '__main__':
    test_path = r"C:\Users\PhilippsLabLaptop\Desktop\20230221_110638_binaries\OctVolume_17_20230221_110638_binaries_644x391x391.bin"
    v = find_center_of_mass(test_path, (644,391,391))
    # save_resized_vol(v, test_path)