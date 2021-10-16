'''
Compare the trained estimators
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
import h5py

sys.path.append('..')
import single_img_verify.noise_model as noise_model
import utils.config_manager

# %%
# testing configuration
estimators = {
    'mean': './mean.cfg',
    'error': './error.cfg',
    'gaussian': './gaussian_assumption.cfg'
}
seed = 1  # should be different from the training seed
nsamples = 50  # number of realizations for clean image y
sample_dir = './forbild/test'
device = '1'
checkpoint = '100.h5'
output_dir = './forbild/results'

GENERATE_NEW_SAMPLE = False
SAVE_RESULTS = True
CALC_ERROR_PRED = False

# %%
# load the training config to load model
estimator_args = {}
for name in estimators:
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    _, _, train_args = utils.config_manager.parse_config_with_extra_arguments(parser, [estimators[name]])
    estimator_args[name] = train_args

# %%
# generate the ground truth
# noise realizations
train_args = estimator_args['mean']  # it should be the same for all three estimators
norm = train_args['Data']['norm']

x0 = (sitk.GetArrayFromImage(sitk.ReadImage(train_args['IO']['x0'])).astype(np.float32) + 1000) / norm
var_roi_map = sitk.GetArrayFromImage(sitk.ReadImage(train_args['IO']['var_roi_map']))[0]
data_model = noise_model.ImageNoiseModel(x0=x0, var_roi_map=var_roi_map, **train_args['NoiseModel'])

# generate the sample pairs x and y
try:
    assert(not GENERATE_NEW_SAMPLE)
    with h5py.File(os.path.join(sample_dir, 'sample_pairs.h5'), 'r') as f:
        x_test = np.copy(f['x'])
        y_test = np.copy(f['y'])
        post_mean_test = np.copy(f['post_mean'])
        post_std_test = np.copy(f['post_std'])
        seed = np.copy(f['seed'])
    print('Sample pair loaded with seed = {0}'.format(seed))
except Exception:
    print('Generating new sample pairs with seed = {0}'.format(seed), flush=True)
    np.random.seed(seed)
    x_test = []
    y_test = []
    post_mean_test = []
    post_std_test = []
    print('Sampling {0} samples'.format(nsamples), end='...', flush=True)
    for i in range(nsamples):
        print(i + 1, end=',', flush=True)
        label, imgs = data_model.forward_sample()
        post_mean, post_std = data_model.posterior_mean_and_std(imgs[0])
        x_test.append(imgs[0])
        y_test.append(label)
        post_mean_test.append(post_mean)
        post_std_test.append(post_std)
    x_test = np.array(x_test) - 1000 / norm
    y_test = np.array(y_test) - 1000 / norm
    post_mean_test = np.array(post_mean_test) - 1000 / norm
    post_std_test = np.array(post_std_test)

    if not os.path.exists(sample_dir):
        os.makedirs(sample_dir)
    with h5py.File(os.path.join(sample_dir, 'sample_pairs.h5'), 'w') as f:
        f['x'] = x_test
        f['y'] = y_test
        f['post_mean'] = post_mean_test
        f['post_std'] = post_std_test
        f['seed'] = seed

    print('done', flush=True)

# %%
# display an example
plt.figure(figsize=[16, 4])
plt.subplot(141)
plt.imshow(x_test[0], 'gray', vmin=-0.16, vmax=0.24)
plt.subplot(142)
plt.imshow(y_test[0], 'gray', vmin=-0.16, vmax=0.24)
plt.subplot(143)
plt.imshow(post_mean_test[0], 'gray', vmin=-0.16, vmax=0.24)
plt.subplot(144)
plt.imshow(post_std_test[0], 'gray', vmin=0.025, vmax=0.075)

# %%
# generate predictions
preds = {}
# mean predictor
os.environ['CUDA_VISIBLE_DEVICES'] = device
K.clear_session()
for name in estimator_args:
    print('Predicting {0}'.format(name), flush=True)

    if name == 'gaussian':
        chkpt = '100.h5'
    else:
        chkpt = checkpoint

    train_args = estimator_args[name]
    model = tf.keras.models.load_model(
        os.path.join(train_args['IO']['output_dir'], train_args['IO']['tag'], chkpt),
        compile=False
    )
    if name == 'mean':
        pred = model.predict(x_test[..., np.newaxis], batch_size=1)
    elif name == 'error':
        x = np.concatenate([x_test[..., np.newaxis], preds['mean']], -1)
        pred = model.predict(x, batch_size=1)
        pred[pred < 0] = 0
        pred = np.sqrt(pred) / train_args['Data']['scale_y']
    else:
        pred = model.predict(x_test[..., np.newaxis], batch_size=1)
        pred[..., 1] = np.sqrt(np.exp(pred[..., 1]))

    preds[name] = pred

# %%
# Check the denoising error
if CALC_ERROR_PRED:
    ref = preds['mean'][..., 0]
    post_std_test = np.sqrt(post_std_test**2 + post_mean_test**2 - 2 * post_mean_test * ref + ref**2)

# %%
# check the profile
ind = 0
ix = 140
iy0 = 100
iy1 = 250


# %%
# show the results of aleatoric uncertainty
plt.figure(figsize=[12, 6])
# mean
vmin_mean = -0.16
vmax_mean = 0.24
plt.subplot(241)
plt.imshow(post_mean_test[ind], 'gray', vmin=vmin_mean, vmax=vmax_mean)
plt.axis(False)
plt.subplot(242)
plt.imshow(preds['gaussian'][ind, ..., 0], 'gray', vmin=vmin_mean, vmax=vmax_mean)
plt.axis(False)
plt.subplot(243)
plt.imshow(preds['mean'][ind, ..., 0], 'gray', vmin=vmin_mean, vmax=vmax_mean)
plt.axis(False)
plt.subplot(244)
plt.plot(preds['gaussian'][ind, iy0:iy1, ix, 0] * norm)
plt.plot(preds['mean'][ind, iy0:iy1, ix, 0] * norm)
plt.plot(post_mean_test[ind, iy0:iy1, ix] * norm)
plt.xlim([0, iy1 - iy0 + 1])
plt.ylim([-100, 200])
plt.xlabel('Pixel')
plt.ylabel('Expectation (HU)')
plt.tight_layout()
# variance
vmin_std = 0.035
vmax_std = 0.055
plt.subplot(245)
plt.imshow(post_std_test[ind], 'gray', vmin=vmin_std, vmax=vmax_std)
plt.axis(False)
plt.subplot(246)
plt.imshow(preds['gaussian'][ind, ..., 1], 'gray', vmin=vmin_std, vmax=vmax_std)
plt.axis(False)
plt.subplot(247)
plt.imshow(preds['error'][ind, ..., 0], 'gray', vmin=vmin_std, vmax=vmax_std)
plt.axis(False)
plt.subplot(248)
plt.plot(preds['gaussian'][ind, iy0:iy1, ix, 1] * norm)
plt.plot(preds['error'][ind, iy0:iy1, ix, 0] * norm)
plt.plot(post_std_test[ind, iy0:iy1, ix] * norm)
plt.xlim([0, iy1 - iy0 + 1])
plt.ylim([42, 54])
plt.xlabel('Pixel')
plt.ylabel('Std (HU)')
plt.tight_layout()


# %%
# RMSE
def rmse(x, y, norm=1000):
    err = np.sqrt(np.mean((x - y)**2, (1, 2))) * norm
    return np.mean(err), np.std(err)


rmse_mean = rmse(post_mean_test, preds['mean'][..., 0])
rmse_mean_gaussian = rmse(post_mean_test, preds['gaussian'][..., 0])

rmse_std = rmse(post_std_test, preds['error'][..., 0])
rmse_std_gaussian = rmse(post_std_test, preds['gaussian'][..., 1])

print('Mean RMSE, Gaussian = {0}+-{1}, UNet = {2}+-{3}'.format(
    rmse_mean_gaussian[0], rmse_mean_gaussian[1],
    rmse_mean[0], rmse_mean[1]
))
print('Mean RMSE, Gaussian = {0}+-{1}, UNet = {2}+-{3}'.format(
    rmse_std_gaussian[0], rmse_std_gaussian[1],
    rmse_std[0], rmse_std[1]
))


# %%
# save the results
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


def save_profile_plot(filename, imgs, ix=140, iy=[100, 250], ratio=1000, ymin=42, ymax=55):
    plt.figure(figsize=[4, 3], dpi=200)
    if iy is None:
        iy = [0, pred.shape[0]]
    for i, img in enumerate(imgs):
        plt.plot(img[iy[0]:iy[1], ix] * ratio)
    plt.ylim([ymin, ymax])

    plt.ylabel('STD (HU)')
    plt.xlabel('Pixel')

    if filename is not None:
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()


# %%
if SAVE_RESULTS:
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    save_profile_img(
        os.path.join(output_dir, 'mean_post.png'), post_mean_test[ind], vmin_mean, vmax_mean, icolor=3
    )
    save_img(
        os.path.join(output_dir, 'x.png'), x_test[ind], vmin_mean, vmax_mean
    )
    save_img(
        os.path.join(output_dir, 'y.png'), y_test[ind], vmin_mean, vmax_mean
    )
    save_img(
        os.path.join(output_dir, 'mean_gauss.png'), preds['gaussian'][ind, ..., 0], vmin_mean, vmax_mean
    )
    save_img(
        os.path.join(output_dir, 'mean_unet.png'), preds['mean'][ind, ..., 0], vmin_mean, vmax_mean
    )

    if CALC_ERROR_PRED:
        save_profile_img(
            os.path.join(output_dir, 'std_post_err_pred.png'), post_std_test[ind], vmin_std, vmax_std, icolor=3
        )
    else:
        save_profile_img(
            os.path.join(output_dir, 'std_post.png'), post_std_test[ind], vmin_std, vmax_std, icolor=3
        )
    save_img(
        os.path.join(output_dir, 'std_gauss.png'), preds['gaussian'][ind, ..., 1], vmin_std, vmax_std
    )
    save_img(
        os.path.join(output_dir, 'std_unet.png'), preds['error'][ind, ..., 0], vmin_std, vmax_std
    )

    save_profile_plot(
        os.path.join(output_dir, 'mean_plot.png'),
        [preds['gaussian'][ind, ..., 0], preds['mean'][ind, ..., 0], post_mean_test[ind]],
        ymin=-100, ymax=200
    )

    if CALC_ERROR_PRED:
        save_profile_plot(
            os.path.join(output_dir, 'std_plot_err_pred.png'),
            [preds['error'][ind, ..., 0], preds['error'][ind, ..., 0], post_std_test[ind]],
        )
    else:
        save_profile_plot(
            os.path.join(output_dir, 'std_plot.png'),
            [preds['gaussian'][ind, ..., 1], preds['error'][ind, ..., 0], post_std_test[ind]],
        )

# %%
