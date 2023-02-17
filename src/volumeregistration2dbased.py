import os
import glob
import numpy as np
from tqdm.auto import tqdm 
from itertools import groupby
from scipy.ndimage import shift
import matplotlib.pyplot as plt
from skimage import registration
from skimage.registration import phase_cross_correlation
from image_registration import chi2_shift, cross_correlation_shifts


class VolumeRegistration2Dbased():
    def __init__(self, params_dict) -> None:
        self.vol_files = glob.glob(params_dict["path_loading"] + "/*")
        self.path_saving = params_dict['path_saving']
        self.volume_pre_checks()
    
    def all_equal(self, list) -> bool:
        g = groupby(list)
        return next(g, True) and not next(g, False)
    
    def get_dimensions_from_file_name(self, file: str) -> list:
        dims = os.path.basename(file).split('_')[-1].split('.bin')[0].split('x')
        return [int(i) for i in dims]

    def volume_pre_checks(self):
        print("[INFO:] Asserting all file names in list to follow the naming convention \"./../<prefix>_AxBxC.bin\"\nand containing volumes of equal size")
        file_sizes = [os.path.getsize(i) for i in self.vol_files]
        if not self.all_equal(file_sizes):
            raise Exception("File sizes are different!")
        dims = [self.get_dimensions_from_file_name(i) for i in self.vol_files]
        if not self.all_equal(dims):
            raise Exception("Dimensions for files (by file name) are different!")
        self.dims = dims[0]
          
    @staticmethod  
    def signaltonoise(a, axis=0, ddof=0):
        a = np.asanyarray(a)
        m = a.mean(axis)
        sd = a.std(axis=axis, ddof=ddof)
        return np.where(sd == 0, 0, m/sd)

    @staticmethod
    def chi_squared_registration(image: np.array, offset_image: np.array, noise: float=0.1) -> np.ndarray:
        xoff, yoff, _, _ = chi2_shift(image, offset_image, noise, return_error=True, upsample_factor='auto')
        return shift(offset_image, shift=(xoff,yoff), mode='constant')
     
    @staticmethod   
    def cross_correlation_registration(image: np.array, offset_image: np.array) -> np.ndarray:
        xoff, yoff = cross_correlation_shifts(image, offset_image)
        return shift(offset_image, shift=(xoff,yoff), mode='constant')    
    
    @staticmethod
    def optical_flow_registration(image: np.array, offset_image: np.array) -> np.array:
        flow = registration.optical_flow_tvl1(image, offset_image)
        flow_x, flow_y = flow[1, :, :], flow[0, :, :]
        xoff, yoff = np.mean(flow_x), np.mean(flow_y)
        return shift(offset_image, shift=(xoff,yoff), mode='constant')
        
    @staticmethod
    def translation_registration(image: np.array, offset_image: np.array) -> np.array:
        shifted, _, _ = phase_cross_correlation(image, offset_image)
        xoff, yoff = -shifted[1], -shifted[0]
        return shift(offset_image, shift=(xoff,yoff), mode='constant')
    
    def _rigid_registration(self, method: str="chisquared") -> np.array:
        buffer_size_pxls = self.dims[0] * self.dims[1]
        out_vol = np.zeros((tuple(self.dims)), dtype=np.uint8)
        tmp_buffer_stack = np.zeros((self.dims[0], self.dims[1], len(self.vol_files)), dtype=np.uint8)
        out_buffer_stack = np.zeros((self.dims[0], self.dims[1], len(self.vol_files)))
        for i in tqdm(range(self.dims[-1])):
            for j, _ in enumerate(self.vol_files):
                with open(self.vol_files[j]) as c_file: # fill current buffer cluster (B-scans that we want to register)
                    tmp_buffer_stack[...,j] = np.fromfile(c_file, dtype=np.uint8, count=buffer_size_pxls, 
                                                          offset = i * buffer_size_pxls).reshape((self.dims[1], self.dims[0])).swapaxes(0,1)    
            out_buffer_stack[...,0] = tmp_buffer_stack[...,0]
            for k in range(len(self.vol_files)-1): # iterate through all n volumes/current frame in volume and register all against 1st/reference frame
                if method=='noreg': # average volume WITHOUT ANY REGISTRATION
                    out_vol[:,:,i] = np.mean(tmp_buffer_stack, axis=-1)
                elif method=='chisquared':
                    out_buffer_stack[:,:,k] = VolumeRegistration2Dbased.chi_squared_registration(tmp_buffer_stack[:,:,0], tmp_buffer_stack[:,:,k+1])
                elif method=='crosscorrelation':
                    out_buffer_stack[:,:,k] = VolumeRegistration2Dbased.cross_correlation_registration(tmp_buffer_stack[:,:,0], tmp_buffer_stack[:,:,k+1]) 
                elif method=='opticalflow':
                    out_buffer_stack[:,:,k] = VolumeRegistration2Dbased.optical_flow_registration(tmp_buffer_stack[:,:,0], tmp_buffer_stack[:,:,k+1]) 
                elif method=='registrationtranslation':
                    out_buffer_stack[:,:,k] = VolumeRegistration2Dbased.translation_registration(tmp_buffer_stack[:,:,0], tmp_buffer_stack[:,:,k+1])
            out_vol[:,:,i] = np.mean(out_buffer_stack, axis=-1)
        print(f"\nSNR of raw volume={VolumeRegistration2Dbased.signaltonoise(out_buffer_stack)}\nand SNR of registered volume={VolumeRegistration2Dbased.signaltonoise(out_vol)}")
        return out_vol
            
    def _save_volume_to_disk(self, out_vol):
        print("[INFO:] Saving to disk...")
        np.swapaxes(out_vol, 0, -1).astype(np.uint8).tofile(self.path_saving) #TODO: grab that from config
        print("[INFO:] Done!")