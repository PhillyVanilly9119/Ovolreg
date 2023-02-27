import os
import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import zscore


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


def save_resized_vol(vol: np.array, path_saving: str) -> None:
    dims = vol.shape
    dims_str = str(dims[0]) + 'x' + str(dims[1]) + 'x' + str(dims[2])
    tail, head = os.path.split(path_saving)
    tmp = head.split('_')
    tmp = tmp[:-1]
    tmp = ''.join([e + '_' for e in tmp])
    fullpath_saving = os.path.join(tail, 'resized', tmp + dims_str + '.bin')
    print(fullpath_saving) 
    
    
def find_center_of_mass(path_loading: str, in_dims: tuple):
    with open(path_loading) as f:
        vol = np.fromfile(f, dtype=np.uint8).reshape(in_dims)
        a_scan = vol[:,200,200].astype(np.float32)
    c_of_mass = zscore(a_scan)
    plt.plot(c_of_mass)
    plt.plot((a_scan - a_scan.min())/(a_scan.max() - a_scan.min()))
    # plt.plot((c_of_mass - c_of_mass.min())/(c_of_mass.max() - c_of_mass.min()))
    plt.show()
    
    
if __name__ == '__main__':
    test_path = r"C:\Users\PhilippsLabLaptop\Desktop\20230221_110638_binaries\OctVolume_17_20230221_110638_binaries_644x391x391.bin"
    v = find_center_of_mass(test_path, (644,391,391))
    # save_resized_vol(v, test_path)