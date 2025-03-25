import cv2
import numpy as np
from typing import Dict
import os
import sys

H = 540
W = 960
TYPE = None

def img_save(img: np.ndarray, save_path: str, is_BRG=True, is_int8=True):
    if len(img.shape) == 4 and img.shape[0] == 1 and img.shape[1] == 1:
        img = np.squeeze(img, axis=(0, 1))
    elif len(img.shape) == 3 and img.shape[0] == 3:
        img = np.transpose(1, 2, 0)

    if is_int8:
        img = np.clip(img*255.0, 0, 255).astype(np.uint8)
    else:
        img = np.clip(img, 0, 255).astype(np.uint8)

    if not is_BRG:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    elif len(img.shape) == 2 or img.shape[2] == 1:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    cv2.imwrite(save_path, img)


def img_read(path : str, is_single=False, is_vis=False, is_int8=False) -> np.ndarray:

    '''
    return img：灰度图像 维度(1, 1, H, W) 或 三通道图像 BGR 维度(H, W, C)
    '''

    if is_single:
        img = cv2.imread(path)
        img = cv2.resize(img, (W, H), interpolation=cv2.INTER_LANCZOS4)
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        if is_vis:
            img = cv2.imread(path)
            img = cv2.resize(img, (W, H), interpolation=cv2.INTER_LANCZOS4)
            #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (W, H), interpolation=cv2.INTER_LANCZOS4)
            img = np.expand_dims(np.expand_dims(img, axis=0), axis=0)
    if not is_int8:
        img = img.astype(np.float32) / 255.0
    return img

def get_img_list(path: str):
    return [os.path.join(path, i) for i in os.listdir(path)]


def BGR2YCrCb(img : np.ndarray):

    if img.dtype != np.float32:
        img = img.astype(np.float32)
    # if img.shape[-1] == 3:
    #     img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    Y, Cr, Cb = cv2.split(img)
    Y = np.expand_dims(np.expand_dims(Y, axis=0), axis=0)

    return Y, Cr, Cb


def YCrCb2BGR(Y: np.ndarray, Cr: np.ndarray, Cb: np.ndarray) -> np.ndarray:
    
    Y = np.squeeze(np.squeeze(Y, axis=0), axis=0)
    img = cv2.merge([Y, Cr, Cb])

    img = cv2.cvtColor(img, cv2.COLOR_YCrCb2RGB)

    return img





def ImgFuse(ir, vi):

    '''
    ir (channels, height, width)
    vi (channels, height, width)
    '''

    fuse_img = None
    return fuse_img


def PredModel(img):

    results = None
    return results
