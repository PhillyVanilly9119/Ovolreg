import cv2
import scipy
import os, sys
import numpy as np
from PIL import Image
from tqdm.auto import tqdm 
from itertools import groupby
from scipy.ndimage import shift
import matplotlib.pyplot as plt
from image_registration import chi2_shift, cross_correlation_shifts

volume_dirs = (
    r"C:\Users\PhilippsLabLaptop\Desktop\recon_rasterVol01_4000x512x511.bin",
    r"C:\Users\PhilippsLabLaptop\Desktop\recon_rasterVol02_4000x512x511.bin",
    r"C:\Users\PhilippsLabLaptop\Desktop\recon_rasterVol03_4000x512x511.bin"
    )

def check_size_compatibility(input_list: list) -> bool:
    """_summary_

    Args:
        input_list (list): _description_
    Returns:
        bool: _description_
    """
    file_sizes = [os.path.getsize(i) for i in input_list]
    g = groupby(file_sizes)
    return next(g, True) and not next(g, False)


def get_dimensions_from_file_name(file: str) -> list:
    """Returns volume dimensions from the file name in which the data is stored
    Args:
        file (str): file name from which the string is split
        asserts the following form for the file path: "../../<prefix>_AxBxC.bin"
    Returns:
        list: list with the volume dimensions -> (a,b,c)
    """
    dims = os.path.basename(file).split('_')[-1].split('.bin')[0].split('x')
    return [int(i) for i in dims]


def FAST_oriented_BRIEF_2D_registration(img1: np.array, img2: np.array, n_orb_feat: int=250) -> np.array:
    """TODO: buggy -> produces crazy artifacts
        img1 is the reference frame onto which the dewarped image is overlayed -> both images are averaged
    """
    # assuming img1 and img2 have the same size and color code
    orb_detector = cv2.ORB_create(n_orb_feat)
    kp1, d1 = orb_detector.detectAndCompute(img1, None)
    kp2, d2 = orb_detector.detectAndCompute(img2, None)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(d1, d2)
    matches.sort(key = lambda x: x.distance)
    matches = matches[:int(len(matches)*0.9)]
    no_of_matches = len(matches)
    p1 = np.zeros((no_of_matches, 2))
    p2 = np.zeros((no_of_matches, 2))
    for i in range(len(matches)):
        p1[i, :] = kp1[matches[i].queryIdx].pt
        p2[i, :] = kp2[matches[i].trainIdx].pt
    homography, _ = cv2.findHomography(p1, p2, cv2.RANSAC)
    transformed_img = cv2.warpPerspective(img2, homography, (img2.shape[1],img2.shape[0]))
    # return average of reg imgs and transform
    return np.asarray(Image.blend(Image.fromarray(img1), Image.fromarray(transformed_img), alpha=0.5)) 


def rigid_registration(img1: np.array, img2: np.array, n_orb_feat: int=250) -> np.array:
    img1 = np.float32(img1)
    img2 = np.float32(img2)
    src = cv2.cornerHarris(img1, img1.shape[1]//100, 5, 0.1)
    dst = cv2.cornerHarris(img2, img2.shape[1]//100, 5, 0.1)
    #result is dilated for marking the corners, not important
    src = cv2.dilate(src, None)
    dst = cv2.dilate(dst, None)
    # ret, dst = cv2.threshold(dst,0.01*dst.max(),255,0)
    # dst = np.uint8(dst)
    M = cv2.findHomography(src, dst, cv2.RANSAC)
    # out = cv2.warpAffine(img2, M, img2.shape)
    # fig, ax = plt.subplots(1,4)
    # ax[0].imshow(img1)
    # ax[1].imshow(img2)
    # ax[2].imshow(dst)
    # ax[3].imshow(out)
    # plt.show()
    return 


def signaltonoise(a, axis=0, ddof=0):
    a = np.asanyarray(a)
    m = a.mean(axis)
    sd = a.std(axis=axis, ddof=ddof)
    return np.where(sd == 0, 0, m/sd)


def chi_squared_registration(image: np.array, offset_image: np.array, noise: float=0.1) -> np.ndarray:
    xoff, yoff, _, _ = chi2_shift(image, offset_image, noise, return_error=True, upsample_factor='auto')
    return shift(offset_image, shift=(xoff,yoff), mode='constant')
    fig, ax = plt.subplots(1, 4)
    ax[0].imshow(image[2500:3500], cmap='gray')
    ax[0].set_title(f"Input image 1: SNR={np.round(signaltonoise(img1, axis=None), 4)}")
    ax[1].imshow(offset_image[2500:3500], cmap='gray')
    ax[1].set_title(f"Input image 1: SNR={np.round(signaltonoise(offset_image, axis=None),4)}")
    ax[2].imshow(corrected_image[2500:3500], cmap='gray') 
    ax[2].set_title(f"Chi Squared corrected image\n(offset_image registered to img 1): SNR={np.round(signaltonoise(corrected_image, axis=None), 4)}")
    ax[3].imshow(((image+corrected_image)/2)[2500:3500], cmap='gray')
    ax[3].set_title(f"Average of Img1 and the\ncorrected image: SNR={np.round(signaltonoise((image+corrected_image)/2, axis=None), 4)}")
    plt.show()
    return


def cross_correlation_registration(image: np.array, offset_image: np.array) -> np.ndarray:
    xoff, yoff = cross_correlation_shifts(image, offset_image)
    corrected_image = shift(offset_image, shift=(xoff,yoff), mode='constant')    
    # return corrected_image
    fig, ax = plt.subplots(1, 4)
    ax[0].imshow(image[2500:3500], cmap='gray')
    ax[0].set_title(f"Input image 1: SNR={np.round(signaltonoise(image, axis=None), 4)}")
    ax[1].imshow(offset_image[2500:3500], cmap='gray')
    ax[1].set_title(f"Input image 1: SNR={np.round(signaltonoise(offset_image, axis=None),4)}")
    ax[2].imshow(corrected_image[2500:3500], cmap='gray') 
    ax[2].set_title(f"Chi Squared corrected image\n(img2 registered to img 1): SNR={np.round(signaltonoise(corrected_image, axis=None), 4)}")
    ax[3].imshow(((image+corrected_image)/2)[2500:3500], cmap='gray')
    ax[3].set_title(f"Average of Img1 and the\ncorrected image: SNR={np.round(signaltonoise((image+corrected_image)/2, axis=None), 4)}")
    plt.show()
    return

def _registration()

def register_n_vols_slice_wise(input_list: list, method: str='noreg', axis=-1) -> np.array:
    if not check_size_compatibility(input_list):
        return # implement error with explanation of what went wrong
    dims = get_dimensions_from_file_name(input_list[0]) # rethink reshaping
    buffer_size_pxls = dims[0] * dims[1]
    out_vol = np.zeros((tuple(dims)), dtype=np.uint8)
    tmp_buffer_stack = np.zeros((dims[0], dims[1], len(input_list)), dtype=np.uint8)
    out_buffer_stack = np.zeros((dims[0], dims[1], len(input_list)))
    for i in tqdm(range(dims[-1])):
        for j, _ in enumerate(input_list):
            with open(input_list[j]) as c_file: # fill current buffer cluster (B-scans that we want to register)
                tmp_buffer_stack[...,j] = np.fromfile(c_file, 
                                                dtype=np.uint8,
                                                count=buffer_size_pxls,
                                                offset=i * buffer_size_pxls).reshape((dims[1], dims[0])).swapaxes(0,1)    
        out_buffer_stack[...,0] = tmp_buffer_stack[...,0]
        if method=='noreg': # average volume WITHOUT ANY REGISTRATION
            out_vol[:,:,i] = np.mean(tmp_buffer_stack, axis=-1)  
        elif method=='fastbrief': # none-rigid registration -> currently produces crazy artifacts --> TODO: debug
            for k in range(len(input_list)-1): # iterate through all n volumes/current frame in volume and register all against 1st/reference frame
                out_buffer_stack[:,:,k] = FAST_oriented_BRIEF_2D_registration(tmp_buffer_stack[:,:,0], tmp_buffer_stack[:,:,k+1]) # perform registration
            out_vol[:,:,i] = np.mean(out_buffer_stack, axis=-1)
        # elif method=='rigid':
        #     rigid_registration(reg_buffer_stack[:,:,0], reg_buffer_stack[:,:,1])
        elif method=='chisquared':
            for k in range(len(input_list)-1): # iterate through all n volumes/current frame in volume and register all against 1st/reference frame
                out_buffer_stack[:,:,k] = chi_squared_registration(tmp_buffer_stack[:,:,0], tmp_buffer_stack[:,:,k+1]) # perform registration
            out_vol[:,:,i] = np.mean(out_buffer_stack, axis=-1)
        elif method=='crosscorrelation':
            for k in range(len(input_list)-1): # iterate through all n volumes/current frame in volume and register all against 1st/reference frame
                out_buffer_stack[:,:,k] = chi_squared_registration(tmp_buffer_stack[:,:,0], tmp_buffer_stack[:,:,k+1]) # perform registration
            out_vol[:,:,i] = np.mean(out_buffer_stack, axis=-1)
        
    return out_vol    


def save_averaged_volume(vol: np.array, path: str):
    assert os.path.isfile(path)
    np.swapaxes(vol, 0, -1).astype(np.uint8).tofile(path)
    return


vol = register_n_vols_slice_wise(volume_dirs, 'crosscorrelation')
save_averaged_volume(vol, r"C:\Users\PhilippsLabLaptop\Desktop\crosscorr_vol_bidirectional.bin")
# np.swapaxes(v3, 0, -1).astype(np.uint8).tofile()