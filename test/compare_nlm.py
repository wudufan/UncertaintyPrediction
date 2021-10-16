'''
Compare the estimators with nlm for real patients
'''

# %%
import numpy as np
import cupy as cp
import pandas as pd
import os
import matplotlib.pyplot as plt
import sys
import tensorflow as tf
import tensorflow.keras.backend as K
import argparse
import imageio
import h5py

sys.path.append('..')
import utils.config_manager
import CTProjector.prior.recon_prior_cupy as recon_prior

# %%
# testing configuration
nlm_d = 0.05
search_size = 11
kernel_size = 5
kernel_std = 2.5

estimators = {
    'error': [
        '../train/config/uncertainty/l2_depth_4.cfg',
        '--IO.output_dir',
        '"/home/dwu/trainData/uncertainty_prediction/train/mayo_2d_3_layer_mean/dose_rate_4/uncertainty/nlm_{0}"'
        .format(nlm_d),
        '--IO.tag', '"l2_depth_4/img"'
    ],
}
filename_x = '/home/dwu/trainData/uncertainty_prediction/data/mayo_2d_3_layer_mean/dose_rate_4.h5'
filename_y = '/home/dwu/trainData/uncertainty_prediction/data/mayo_2d_3_layer_mean/dose_rate_1.h5'
filename_manifest = '/home/dwu/trainData/uncertainty_prediction/data/mayo_2d_3_layer_mean/manifest.csv'
device = '1'
checkpoint = '100.h5'
output_dir = './patients/nlm_{0}'.format(nlm_d)

SAVE_RESULTS = True

# %%
# load the training config to load model
estimator_args = {}
for name in estimators:
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    _, _, train_args = utils.config_manager.parse_config_with_extra_arguments(parser, estimators[name])
    estimator_args[name] = train_args

# %%
# load the data
train_args = estimator_args['error']  # it should be the same for all three estimators
norm = train_args['Data']['norm']

manifest = pd.read_csv(filename_manifest)
manifest = manifest[manifest['Tag'].isin(train_args['IO']['valid'])]
islices = manifest['Index'].values

with h5py.File(filename_x, 'r') as f:
    x_test = np.copy(f['img'][islices]) / norm

with h5py.File(filename_y, 'r') as f:
    y_test = np.copy(f['img'][islices]) / norm

# %%
# display an example
plt.figure(figsize=[8, 8])
plt.subplot(221)
plt.imshow(x_test[100], 'gray', vmin=-0.16, vmax=0.24)
plt.subplot(222)
plt.imshow(y_test[100], 'gray', vmin=-0.16, vmax=0.24)
plt.subplot(223)
plt.imshow(x_test[295], 'gray', vmin=-0.16, vmax=0.24)
plt.subplot(224)
plt.imshow(y_test[295], 'gray', vmin=-0.16, vmax=0.24)
plt.show()

# %%
# make predictions
os.environ['CUDA_VISIBLE_DEVICES'] = device
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)
cp.cuda.Device(int(device)).use()
K.clear_session()

# %%
preds = {}
# nlm
pred = []
print('nlm', len(x_test))
for i in range(len(x_test)):
    if (i + 1) % 50 == 0:
        print(i + 1, end=',', flush=True)
    x_cp = cp.array(x_test[i], dtype=cp.float32, order='C')[cp.newaxis, cp.newaxis, ...]
    res = recon_prior.nlm(
        x_cp,
        x_cp,
        nlm_d,
        [1, search_size, search_size],
        [1, kernel_size, kernel_size],
        [1, kernel_std, kernel_std]
    )
    pred.append(res.get()[0, 0])
print('')
pred = np.array(pred)[..., np.newaxis]
preds['nlm'] = pred

# %%
for name in estimator_args:
    print('Predicting {0}'.format(name), flush=True)

    chkpt = checkpoint

    train_args = estimator_args[name]
    model = tf.keras.models.load_model(
        os.path.join(train_args['IO']['output_dir'], train_args['IO']['tag'], checkpoint),
        compile=False
    )
    if name == 'error':
        x = np.concatenate([x_test[..., np.newaxis], preds['nlm']], -1)
        pred = model.predict(x, batch_size=1)
        pred[pred < 0] = 0
        pred = np.sqrt(pred) / train_args['Data']['scale_y']

    preds[name] = pred

# %%
# snapshots
islice1 = 101
islice2 = 295
vmin = 0
vmax = 0.05
plt.figure(figsize=[16, 8])
plt.subplot(231)
plt.imshow(x_test[islice1, 64:-64, 64:-64], 'gray', vmin=-0.16, vmax=0.24)
plt.subplot(232)
plt.imshow(preds['error'][islice1, 64:-64, 64:-64, 0], 'gray', vmin=vmin, vmax=vmax)
plt.subplot(233)
plt.imshow(
    np.abs(preds['nlm'][islice1, 64:-64, 64:-64, 0] - y_test[islice1, 64:-64, 64:-64]), 'gray', vmin=0, vmax=0.05
)
plt.subplot(234)
plt.imshow(x_test[islice2, 64:-64, 64:-64], 'gray', vmin=-0.16, vmax=0.24)
plt.subplot(235)
plt.imshow(preds['error'][islice2, 64:-64, 64:-64, 0], 'gray', vmin=vmin, vmax=vmax)
plt.subplot(236)
plt.imshow(
    np.abs(preds['nlm'][islice2, 64:-64, 64:-64, 0] - y_test[islice2, 64:-64, 64:-64]), 'gray', vmin=0, vmax=0.05
)

# %%
# profiles
# mean
plt.figure(figsize=[4, 3])
plt.plot(np.mean(y_test, (1, 2)) * 1000)
plt.plot(np.mean(preds['nlm'][..., 0], (1, 2)) * 1000)

# variance
var_test = (preds['nlm'][..., 0] - y_test)**2
plt.figure(figsize=[4, 3])
plt.plot(np.sqrt(np.mean(var_test, (1, 2))) * 1000)
plt.plot(np.sqrt(np.mean(preds['error'][..., 0]**2, (1, 2))) * 1000)


# %%
# save slices
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


# %%
if SAVE_RESULTS:
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for islice in [islice1, islice2]:
        save_img(
            os.path.join(output_dir, 'x_{0}.png'.format(islice)),
            x_test[islice, 64:-64, 64:-64],
            -0.16, 0.24
        )
        save_img(
            os.path.join(output_dir, 'y_{0}.png'.format(islice)),
            y_test[islice, 64:-64, 64:-64],
            -0.16, 0.24
        )
        save_img(
            os.path.join(output_dir, 'nlm_{0}.png'.format(islice)),
            preds['nlm'][islice, 64:-64, 64:-64],
            -0.16, 0.24
        )
        save_img(
            os.path.join(output_dir, 'unet_std_{0}.png'.format(islice)),
            preds['error'][islice, 64:-64, 64:-64, 0],
            0, 0.05
        )
        save_img(
            os.path.join(output_dir, 'err_{0}.png'.format(islice)),
            np.abs(y_test[islice, 64:-64, 64:-64] - preds['nlm'][islice, 64:-64, 64:-64, 0]),
            0, 0.05
        )

    # variance plot
    var_test = (preds['nlm'][..., 0] - y_test)**2
    plt.figure(figsize=[4, 3], dpi=200)
    plt.plot(np.sqrt(np.mean(preds['error'][..., 0]**2, (1, 2))) * 1000)
    plt.plot(np.sqrt(np.mean(preds['error'][..., 0]**2, (1, 2))) * 1000)
    plt.plot(np.sqrt(np.mean(var_test, (1, 2))) * 1000)
    plt.xlabel('# Testing Image')
    plt.ylabel('STD (HU)')
    plt.ylim([0, 40])
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'plot_std.png'))

# %%
