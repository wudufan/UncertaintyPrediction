'''
Verify the predicted images
'''

# %%
import numpy as np
import os
import matplotlib.pyplot as plt
import SimpleITK as sitk
import sys
import tensorflow as tf
import tensorflow.keras.backend as K
import argparse
import imageio

sys.path.append('..')
import single_img_verify.noise_model as noise_model
import utils.config_manager

# %%
# the configuration to use
# cmds = ['error.cfg']
cmds = [
    'error.cfg',
    '--IO.x0', '"forbild/x0.nii"',
    '--IO.var_roi_map', '"forbild/variance.seg.nrrd"',
    '--IO.tag', '"forbild/error_x0"',
    '--IO.checkpoint', '"/home/dwu/trainData/uncertainty_prediction/single_img_verify/forbild/mean/100.h5"',
    '--Network.input_shape', '(512,512,1)',
]
RUN_PREDICTION = True
SAVE_RESULTS = False
parser = argparse.ArgumentParser()
parser.add_argument('config')
args, config, train_args = utils.config_manager.parse_config_with_extra_arguments(parser, cmds)

# %%
norm = train_args['Data']['norm']
norm_y = train_args['Data']['scale_y'] * train_args['Display']['norm_y']

working_dir = os.path.dirname(os.path.join(train_args['IO']['output_dir'], train_args['IO']['tag']).rstrip('/'))
x0 = (sitk.GetArrayFromImage(sitk.ReadImage(train_args['IO']['x0'])).astype(np.float32) + 1000) / norm
var_roi_map = sitk.GetArrayFromImage(sitk.ReadImage(train_args['IO']['var_roi_map']))[0]

x = (sitk.GetArrayFromImage(sitk.ReadImage(
    os.path.join(working_dir, 'mean/valid/mean.x.nii')
)).astype(np.float32) + 1000) / norm
pred_mean = (sitk.GetArrayFromImage(sitk.ReadImage(
    os.path.join(working_dir, 'mean/valid/mean.pred.nii')
)).astype(np.float32) + 1000) / norm
pred_error = sitk.GetArrayFromImage(sitk.ReadImage(
    os.path.join(working_dir, 'error/valid/error.pred.nii')
)).astype(np.float32) / norm_y
y_error = sitk.GetArrayFromImage(sitk.ReadImage(
    os.path.join(working_dir, 'error/valid/error.y.nii')
)).astype(np.float32) / norm_y
pred_error0 = sitk.GetArrayFromImage(sitk.ReadImage(
    os.path.join(working_dir, 'error_x0/valid/error_x0.pred.nii')
)).astype(np.float32) / norm_y
y_error0 = sitk.GetArrayFromImage(sitk.ReadImage(
    os.path.join(working_dir, 'error_x0/valid/error_x0.y.nii')
)).astype(np.float32) / norm_y

# %%
data_model = noise_model.ImageNoiseModel(x0, var_roi_map, **train_args['NoiseModel'])

np.random.seed(0)
x_train = []
y_train = []
print('Sampling 100 samples', end='...', flush=True)
for i in range(100):
    label, imgs = data_model.forward_sample()
    x_train.append(imgs[0])
    y_train.append(label)
x_train = np.array(x_train)[..., np.newaxis] - 1000 / norm
y_train = np.array(y_train)[..., np.newaxis] - 1000 / norm
print('done', flush=True)

# %%
# load model and do prediction
if RUN_PREDICTION:
    print('Predicting 100 samples', end='...', flush=True)
    os.environ['CUDA_VISIBLE_DEVICES'] = train_args['Train']['device']
    K.clear_session()
    pred_model = tf.keras.models.load_model(train_args['IO']['checkpoint'])
    pred_train = pred_model.predict(x_train, batch_size=1)
    print('done', flush=True)
    err_mean = np.sqrt(np.mean((y_train - pred_train)**2, 0))[..., 0]

    plt.figure(figsize=(15, 5))
    plt.subplot(131)
    plt.imshow(y_error[0], 'gray', vmin=0.025, vmax=0.075)
    plt.subplot(132)
    plt.imshow(pred_error[0], 'gray', vmin=0.025, vmax=0.075)
    plt.subplot(133)
    plt.imshow(err_mean, 'gray', vmin=0.025, vmax=0.075)

# %%
err0_mean = np.sqrt(np.mean((y_train - (x0[np.newaxis, ..., np.newaxis] - 1000 / norm))**2, 0))[..., 0]
plt.figure(figsize=(15, 5))
plt.subplot(131)
plt.imshow(y_error0[0], 'gray', vmin=0.0, vmax=0.1)
plt.subplot(132)
plt.imshow(pred_error0[0], 'gray', vmin=0.0, vmax=0.1)
plt.subplot(133)
plt.imshow(err0_mean, 'gray', vmin=0.0, vmax=0.1)

# %%
# generate figures
output_dir = os.path.join('figure', os.path.dirname(train_args['IO']['tag']))
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


def save_img(filename, img, vmin, vmax):
    img = (img - vmin) / (vmax - vmin) * 255
    img[img < 0] = 0
    img[img > 255] = 255
    img = img.astype(np.uint8)

    if filename is None:
        plt.figure()
        plt.imshow(img, 'gray')
    else:
        imageio.imwrite(filename, img)


# figure for phantom generation
if SAVE_RESULTS:
    save_img(os.path.join(output_dir, 'y0.png'), x0, 0.84, 1.24)
    std0 = np.sqrt(data_model.std_x0**2 * (1 + data_model.var_roi_map * data_model.var_roi_ratio))
    save_img(os.path.join(output_dir, 'sigma0_50_100_HU.png'), std0, 0.05, 0.1)
    save_img(os.path.join(output_dir, 'y.png'), y_train[0], -0.16, 0.24)
    save_img(os.path.join(output_dir, 'x.png'), x_train[0], -0.16, 0.24)


# %%
# profile plot
def save_profile_img(filename, img, vmin, vmax, ix=140, iy=[100, 250], icolor=0):
    color = plt.rcParams['axes.prop_cycle'].by_key()['color'][icolor]

    plt.figure(figsize=[4, 4], dpi=200)
    plt.imshow(img, 'gray', vmin=vmin, vmax=vmax)
    if ix is not None:
        plt.plot([ix, ix], [iy[0], iy[1]], color=color, linestyle='--')
    plt.axis('off')

    if filename is not None:
        plt.savefig(filename, bbox_inches='tight', pad_inches=0)
        plt.close()


def save_profile_plot(filename, pred, ref, ix=140, iy=[100, 250], ratio=1000, ymin=42, ymax=55):
    plt.figure(figsize=[4, 3], dpi=200)
    if iy is None:
        iy = [0, pred.shape[0]]
    plt.plot(pred[iy[0]:iy[1], ix] * ratio)
    plt.plot(ref[iy[0]:iy[1], ix] * ratio, '--')
    plt.ylim([ymin, ymax])

    plt.ylabel('STD (HU)')
    plt.xlabel('Pixel')
    plt.legend(['Prediction', 'Ground truth'], loc=0)

    if filename is not None:
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()


ix = 140
if SAVE_RESULTS:
    save_profile_img(os.path.join(output_dir, 'pred_err0.png'), pred_error[0], 0.025, 0.075)
    save_profile_img(os.path.join(output_dir, 'y_err0.png'), y_error[0], 0.025, 0.075, icolor=1)
    save_profile_img(os.path.join(output_dir, 'pred_err1.png'), pred_error[1], 0.025, 0.075)
    save_profile_img(os.path.join(output_dir, 'y_err1.png'), y_error[1], 0.025, 0.075, icolor=1)
    save_profile_plot(os.path.join(output_dir, 'profile0.png'), pred_error[0], y_error[0])
    save_profile_plot(os.path.join(output_dir, 'profile1.png'), pred_error[1], y_error[1])

# %%
# estimate error values
err_mats = np.zeros([len(pred_error), len(y_error)])
for i in range(len(pred_error)):
    for j in range(len(y_error)):
        err_mats[i, j] = np.sqrt(np.mean((pred_error[i] - y_error[j])**2)) * 1000

psnrs = 20 * np.log10(np.max(y_error, (1, 2)) / np.sqrt(np.mean((pred_error - y_error)**2, (1, 2))))
print('pnsr = ', np.mean(psnrs), np.std(psnrs))

# fix prediction, change the ground truth, how much is exceeded
mismatch_excession = []
for i in range(len(pred_error)):
    for j in range(len(y_error)):
        if i != j:
            mismatch_excession.append((err_mats[i, j] - err_mats[i, i]) / err_mats[i, i])
mismatch_excession = np.array(mismatch_excession)

print(mismatch_excession.min(), mismatch_excession.max(), np.median(mismatch_excession))
