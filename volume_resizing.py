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
    """Generates 4 differently gen

    Args:
        vol (np.array): _description_

    Returns:
        np.array: _description_
    """
    enface_stack = np.zeros((*vol.shape[1:], 4))
    print(enface_stack.shape)
    enface_stack[...,0] = np.mean(vol, axis=0)
    enface_stack[...,1] = np.median(vol, axis=0)
    enface_stack[...,2] = np.max(vol, axis=0)
    enface_stack[...,3] = np.argmax(vol, axis=0)
    return enface_stack


def calculate_snr_for_nVols(vol_list: list, dims: tuple) -> np.array:
    snr = []
    out_vol = np.zeros((*dims,len(vol_list)), dtype=np.uint8)
    for i, vol in tqdm(enumerate(vol_list)):
        with open(vol) as f:
            out_vol[...,i] = np.fromfile(f, dtype=np.uint8).reshape(dims)
    return np.mean(out_vol, axis=-1)
    
    
# @staticmethod
def normalize(x: np.array) -> np.array:
    return (x-x.max()/(x.max()-x.min()))

# @staticmethod
def return_rpe_indeces(vol: np.array):
    return vol.argmax(axis=0)
    
    
if __name__ == '__main__':
    test_path = r"C:\Users\PhilippsLabLaptop\Desktop\20230221_110638_binaries\OctVolume_17_20230221_110638_binaries_644x391x391.bin"
    with open(test_path) as f:
        v = np.fromfile(f, dtype=np.uint8).reshape((644,391,391))
    enfaces = generate_enface_images(v)
    # fig, ax = plt.subplots(2,2)
    # ax[0,0].imshow(enfaces[...,0], cmap='gray')
    # ax[1,0].imshow(enfaces[...,1], cmap='gray')
    # ax[0,1].imshow(enfaces[...,2], cmap='gray')
    # ax[1,1].imshow(enfaces[...,3], cmap='gray')
    # plt.show()