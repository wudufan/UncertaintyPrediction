#%%
import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy.ndimage
import pandas as pd
import SimpleITK as sitk

#%%
preds = sitk.GetArrayFromImage(sitk.ReadImage('/home/dwu/trainData/uncertainty_prediction/denoising_results/mayo_2d_3_layer_mean/l2_depth_3/dose_rate_4/uncertainty_model/l2_depth_3/valid/dose_rate_4.pred.nii'))
y = sitk.GetArrayFromImage(sitk.ReadImage('/home/dwu/trainData/uncertainty_prediction/denoising_results/mayo_2d_3_layer_mean/l2_depth_3/dose_rate_4/uncertainty_model/l2_depth_3/valid/dose_rate_4.y.nii'))

#%%
islice 

# %%
dst_datasets = ('/home/dwu/trainData/deep_denoiser_ensemble/data/mayo_2d_3_layer_mean/dose_rate_1.h5', 
                '/home/dwu/trainData/uncertainty_prediction/denoising_results/mayo_2d_3_layer_mean/l2_depth_3/dose_rate_4/dose_rate_4.h5')
manifest = pd.read_csv('/home/dwu/trainData/deep_denoiser_ensemble/data/mayo_2d_3_layer_mean/manifest.csv')
tags = ['L291','L143','L067']
inds = manifest[manifest['Tag'].isin(tags)]['Index'].values

y = []
for filename in dst_datasets:
    with h5py.File(filename, 'r') as f:
        y.append(np.copy(f['img'][inds])[..., np.newaxis].astype(np.float32) / 1000)
y = np.concatenate(y, -1)

# %%
var_map = (y[...,[0]] - y[...,[1]])**2

islice = 95
img = var_map[95, ..., 0]
smooth_img = scipy.ndimage.gaussian_filter(img, 3)

plt.figure(figsize = [16,8])
plt.subplot(121); plt.imshow(np.sqrt(img), 'gray', vmin=0, vmax=0.05)
plt.subplot(122); plt.imshow(np.sqrt(smooth_img), 'gray', vmin=0, vmax=0.05)


# %%
img2 = np.mean(var_map[90:100, ..., 0], 0)
plt.figure(figsize=(8,8))
plt.imshow(np.sqrt(img2), 'gray', vmin=0, vmax=0.05)

# %%
