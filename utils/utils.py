'''
Utility functions for training
'''

import SimpleITK as sitk
import numpy as np


def save_nii(img, filename, vmin=-1, vmax=1):
    img = (img + 1) / 2 * (vmax - vmin) + vmin

    x = np.copy(img) * 1000
    sitk_img = sitk.GetImageFromArray(x.astype(np.int16))
    sitk.WriteImage(sitk_img, filename)


def snapshot(img, islice=None, vmin=0.5, vmax=1):
    x = np.copy(img)

    if islice is None:
        islice = x.shape[0] // 2

    x = (x[[islice]] - vmin) / (vmax - vmin)
    return x[..., np.newaxis]
