'''
Verify the predicted images
'''

#%%
import numpy as np
import os
import matplotlib.pyplot as plt
import SimpleITK as sitk
import sys
import tensorflow as tf
import tensorflow.keras.backend as K
import argparse

sys.path.append('..')
import single_img_verify.noise_model as noise_model
import utils.config_manager

#%%
# the configuration to use
cmds = ['error.cfg']
RUN_PREDICTION = True
parser = argparse.ArgumentParser()
parser.add_argument('config')
args, config, train_args = utils.config_manager.parse_config_with_extra_arguments(parser, cmds)

#%%
norm = train_args['Data']['norm']
norm_y = train_args['Data']['scale_y'] * train_args['Display']['norm_y']

working_dir = os.path.dirname(os.path.join(train_args['IO']['output_dir'], train_args['IO']['tag']).rstrip('/'))
x0 = (sitk.GetArrayFromImage(sitk.ReadImage(train_args['IO']['x0'])).astype(np.float32) + 1000) / norm
var_roi_map = sitk.GetArrayFromImage(sitk.ReadImage(train_args['IO']['var_roi_map']))[0]

x = (sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(working_dir, 'mean/valid/mean.x.nii'))).astype(np.float32) + 1000) / norm
pred_mean = (sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(working_dir, 'mean/valid/mean.pred.nii'))).astype(np.float32) + 1000) / norm
pred_error = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(working_dir, 'error/valid/error.pred.nii'))).astype(np.float32) / norm_y
y_error = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(working_dir, 'error/valid/error.y.nii'))).astype(np.float32) / norm_y
pred_error0 = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(working_dir, 'error_x0/valid/error_x0.pred.nii'))).astype(np.float32) / norm_y
y_error0 = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(working_dir, 'error_x0/valid/error_x0.y.nii'))).astype(np.float32) / norm_y

# %%
data_model = noise_model.ImageNoiseModel(x0, var_roi_map, **train_args['NoiseModel'])

np.random.seed(0)
x_train = []
y_train = []
print ('Sampling 100 samples', end='...', flush=True)
for i in range(100):
    label, imgs = data_model.forward_sample()
    x_train.append(imgs[0])
    y_train.append(label)
x_train = np.array(x_train)[..., np.newaxis] - 1000 / norm
y_train = np.array(y_train)[..., np.newaxis] - 1000 / norm
print ('done', flush=True)

#%%
# load model and do prediction
if RUN_PREDICTION:
    print ('Predicting 100 samples', end='...', flush=True)
    os.environ['CUDA_VISIBLE_DEVICES'] = train_args['Train']['device']
    K.clear_session()
    pred_model = tf.keras.models.load_model(train_args['IO']['checkpoint'])
    pred_train = pred_model.predict(x_train, batch_size = 1)
    print ('done', flush=True)
    err_mean = np.sqrt(np.mean((y_train - pred_train)**2, 0))[..., 0]

    plt.figure(figsize=(15,5))
    plt.subplot(131); plt.imshow(y_error[0], 'gray', vmin=0.025, vmax=0.075)
    plt.subplot(132); plt.imshow(pred_error[0], 'gray', vmin=0.025, vmax=0.075)
    plt.subplot(133); plt.imshow(err_mean, 'gray', vmin=0.025, vmax=0.075)

#%%
err0_mean = np.sqrt(np.mean((y_train - (x0[np.newaxis, ..., np.newaxis] - 1000 / norm))**2, 0))[..., 0]
plt.figure(figsize=(15,5))
plt.subplot(131); plt.imshow(y_error0[0], 'gray', vmin=0.0, vmax=0.1)
plt.subplot(132); plt.imshow(pred_error0[0], 'gray', vmin=0.0, vmax=0.1)
plt.subplot(133); plt.imshow(err0_mean, 'gray', vmin=0.0, vmax=0.1)

