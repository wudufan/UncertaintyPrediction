#%%
import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy.ndimage
import pandas as pd
import SimpleITK as sitk

#%%
# with h5py.File('/home/dwu/trainData/deep_denoiser_ensemble/data/mayo_2d_3_layer_mean/dose_rate_1.h5', 'r') as f:
#     y = np.copy(f['img'])

# with h5py.File('/home/dwu/trainData/deep_denoiser_ensemble/data/mayo_2d_3_layer_mean/dose_rate_4.h5', 'r') as f:
#     x = np.copy(f['img'])

# with h5py.File('/home/dwu/trainData/uncertainty_prediction/denoising_results/mayo_2d_3_layer_mean/l2_depth_3/dose_rate_4/dose_rate_4.h5', 'r') as f:
#     pred = np.copy(f['img'])

#%%
# plt.figure(figsize=(16,16))
# plt.subplot(221); plt.imshow(y[95, ...], 'gray', vmin=-160, vmax=240)
# plt.subplot(222); plt.imshow(x[95, ...], 'gray', vmin=-160, vmax=240)
# plt.subplot(223); plt.imshow(pred[95, ...], 'gray', vmin=-160, vmax=240)
# plt.subplot(224); plt.imshow(np.abs(y-pred)[95, ...], 'gray', vmin=0, vmax=50)

#%%
# preds = sitk.GetArrayFromImage(sitk.ReadImage('/home/dwu/trainData/uncertainty_prediction/denoising_results/mayo_2d_3_layer_mean/l2_depth_3/dose_rate_4/uncertainty_model/l2_depth_3/patch/valid/dose_rate_4.pred.nii'))
# y = sitk.GetArrayFromImage(sitk.ReadImage('/home/dwu/trainData/uncertainty_prediction/denoising_results/mayo_2d_3_layer_mean/l2_depth_3/dose_rate_4/uncertainty_model/l2_depth_3/patch/valid/dose_rate_4.y.nii'))

x = sitk.GetArrayFromImage(sitk.ReadImage('/home/dwu/trainData/uncertainty_prediction/denoising_results/mayo_2d_3_layer_mean/l2_depth_3/dose_rate_4/uncertainty_model/l2_depth_4/img/valid/dose_rate_4.x.nii'))
preds = sitk.GetArrayFromImage(sitk.ReadImage('/home/dwu/trainData/uncertainty_prediction/denoising_results/mayo_2d_3_layer_mean/l2_depth_3/dose_rate_4/uncertainty_model/l2_depth_4/img/valid/dose_rate_4.pred.nii'))
y = sitk.GetArrayFromImage(sitk.ReadImage('/home/dwu/trainData/uncertainty_prediction/denoising_results/mayo_2d_3_layer_mean/l2_depth_3/dose_rate_4/uncertainty_model/l2_depth_4/img/valid/dose_rate_4.y.nii'))

x = x.astype(np.float32)
preds = preds.astype(np.float32)
y = y.astype(np.float32)

#%%
print ((preds**2).mean()/100, (y**2).mean()/100)

#%%
plt.figure(figsize=(8,4), dpi=200)
plt.subplot(121); plt.imshow(np.sqrt(np.mean(preds[50:100]**2, 0)), 'gray', vmin=0, vmax=500); plt.title('Mean predicted 50 slices (std) \n [0, 50] HU')
plt.subplot(122); plt.imshow(np.sqrt(np.mean(y[50:100]**2, 0)), 'gray', vmin=0, vmax=500); plt.title('Mean error 50 slices (std) \n [0, 50] HU')

# %%
islice = 80
std = 10
smooth_y = np.sqrt(scipy.ndimage.gaussian_filter(y[islice]**2, std))
smooth_p = np.sqrt(scipy.ndimage.gaussian_filter(preds[islice]**2, std))

plt.figure(figsize=(16,4))
plt.subplot(141); plt.imshow(preds[islice], 'gray', vmin=0, vmax=500); plt.title('predicted')
plt.subplot(142); plt.imshow(y[islice], 'gray', vmin=0, vmax=500); plt.title('error')
plt.subplot(143); plt.imshow(smooth_p, 'gray', vmin=0, vmax=500); plt.title('smoothed predicted \n (Gaussian with std=10)')
plt.subplot(144); plt.imshow(smooth_y, 'gray', vmin=0, vmax=500)

#%%
islice = 201
plt.figure(figsize=(15,5))
plt.subplot(131); plt.imshow(x[islice], 'gray', vmin=-160, vmax=240); plt.title('x: [-160,240] HU')
plt.subplot(132); plt.imshow(preds[islice], 'gray', vmin=0, vmax=500); plt.title('uncertainty (std): [0, 50] HU')
plt.subplot(133); plt.imshow(y[islice], 'gray', vmin=0, vmax=500); plt.title('|y-x|: [0, 50] HU')

# %%
var_y_mean = np.mean((y/10)**2, (1,2))
var_pred_mean = np.mean((preds/10)**2, (1,2))

plt.figure(figsize=(4,3), dpi=200)
plt.plot(var_y_mean)
plt.plot(var_pred_mean)
plt.ylabel('Mean Variance (HU$^2$)')
plt.xlabel('#slice')
plt.legend(['$(y-f(x))^2$', 'Predicted'])

# %%
