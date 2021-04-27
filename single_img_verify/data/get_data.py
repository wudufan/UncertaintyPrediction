'''
Get the x0 for single image verification
'''
#%%
import h5py
import SimpleITK as sitk
import pandas as pd
import os
import numpy as np

#%%
tag = 'L291'
islice = 95
input_dir = '/home/dwu/trainData/deep_denoiser_ensemble/data/mayo_2d_3_layer_mean/'
manifest = pd.read_csv(os.path.join(input_dir, 'manifest.csv'))
slice_ind = manifest[manifest['Tag'] == tag].reset_index(drop = True).loc[islice, 'Index']
with h5py.File(os.path.join(input_dir, 'dose_rate_4.h5'), 'r') as f:
    img = np.copy(f['img'][slice_ind])
    spacing = np.copy(f['spacing'])

sitk_img = sitk.GetImageFromArray(img)
sitk_img.SetSpacing([spacing[1], spacing[2]])
sitk.WriteImage(sitk_img, './x0.nii')
