'''
Verify the predicted images
'''

#%%
import numpy as np
import os
import matplotlib.pyplot as plt
import SimpleITK as sitk
import sys

sys.path.append('..')
import single_img_verify.noise_model as noise_model

#%%
working_dir = '/home/dwu/trainData/uncertainty_prediction/single_img_verify/std_x/square/valid/'
x0 = (sitk.GetArrayFromImage(sitk.ReadImage('./data/x0.nii')).astype(np.float32) + 1000) / 1000
var_roi_map = sitk.GetArrayFromImage(sitk.ReadImage('./data/variance.seg.nrrd'))[0]
x = (sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(working_dir, 'valid.x.nii'))).astype(np.float32) + 1000) / 1000
y = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(working_dir, 'valid.y.nii'))).astype(np.float32) / 10000
pred = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(working_dir, 'valid.pred.nii'))).astype(np.float32) / 10000
y[y < 0] = y[y<0] + 6.55536
pred[pred < 0] = pred[pred<0] + 6.55536

# %%
data_model = noise_model.ImageNoiseModel(x0, var_roi_map)

# %%
post_mean, post_std = data_model.posterior_mean_and_std(x[0])
post_sq = post_std**2 + post_mean**2

#%%
error_x0 = np.sqrt(post_sq - 2 * post_mean * x0 + x0**2)
plt.imshow(error_x0*10000, 'gray', vmin=0, vmax=1000)

# %%
