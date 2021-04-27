'''
Training code
'''
#%%
import sys
import os
import tensorflow as tf
import tensorflow.keras.backend as K
import shutil
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt

import argparse

sys.path.append('..')
import single_img_verify.noise_model as noise_model
import model.unet
import model.callbacks
import utils.config_manager

#%%
parser = argparse.ArgumentParser()
parser.add_argument('config')

if sys.argv[0] != 'train.py':
    # debug
    print ('debug')
    verbose = 1
    cmds = ['error_x0.cfg', 
            '--IO.ntrain', '700',
            '--IO.nvalid', '2',
            ]
else:
    print ('no debug')
    verbose = 2
    cmds = None

args, config, train_args = utils.config_manager.parse_config_with_extra_arguments(parser, cmds)

print (args.config)
for sec in train_args:
    print ('[%s]'%sec)
    for k in train_args[sec]:
        print (k, '=', train_args[sec][k])
    print ('', flush=True)

# write the config file to the output directory
output_dir = os.path.join(train_args['IO']['output_dir'], train_args['IO']['tag'])
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
with open(os.path.join(output_dir, 'config.cfg'), 'w') as f:
    config.write(f)

#%%
def postprocess_sqrt(pred):
    pred[pred < 0] = 0
    return np.sqrt(pred)

def postprocess_linear(pred):
    return pred

if train_args['IO']['target'] in ['square', 'error', 'error_x0']:
    print ('Using squared scale, postprocessing sqrt', flush=True)
    scale_y = train_args['Data']['scale_y']**2
    postprocess = postprocess_sqrt
else:
    print ('Using linear scale, postprocessing linear', flush=True)
    scale_y = train_args['Data']['scale_y']
    postprocess = postprocess_linear

#%% 
# generate data
x_train = []
y_train = []
x_valid = []
y_valid = []
ref_valid = []
x0 = (sitk.GetArrayFromImage(sitk.ReadImage(train_args['IO']['x0'])).astype(np.float32) + 1000) / train_args['Data']['norm']
try:
    var_roi_map = sitk.GetArrayFromImage(sitk.ReadImage(train_args['IO']['var_roi_map']))[0]
except Exception as e:
    var_roi_map = None
    print ('Not using variance ROI')
data_model = noise_model.ImageNoiseModel(x0=x0, var_roi_map=var_roi_map, **train_args['NoiseModel']) 

np.random.seed(train_args['IO']['seed'])
print ('Generating %d training images'%train_args['IO']['ntrain'], end=': ', flush=True)
for i in range(train_args['IO']['ntrain']):
    if (i+1) % 100 == 0:
        print (i+1, end=',', flush=True)
    label, imgs = data_model.forward_sample()
    if train_args['IO']['target'] == 'mean':
        y = label
    elif train_args['IO']['target'] == 'square':
        y = label**2
    elif train_args['IO']['target'] == 'error_x0':
        y = (label - x0)**2
    elif train_args['IO']['target'] == 'error':
        raise NotImplementedError('IO.target=error not implemented')
    else:
        raise ValueError('IO.target must be one of "mean", "square", "error"')

    x_train.append(imgs[0])
    y_train.append(y)
print ('done', flush=True)

print ('Generating %d validation images with posterior pdf'%train_args['IO']['nvalid'], end=': ', flush=True)
for i in range(train_args['IO']['nvalid']):
    if (i+1) % 1 == 0:
        print (i+1, end=',', flush=True)
    label, imgs = data_model.forward_sample()
    post_mean, post_std = data_model.posterior_mean_and_std(imgs[0])
    if train_args['IO']['target'] == 'mean':
        y = post_mean
        ref_valid.append(np.zeros_like(y))
    elif train_args['IO']['target'] == 'square':
        y = post_std**2 + post_mean**2
        ref_valid.append(post_mean**2)
    elif train_args['IO']['target'] == 'error_x0':
        y = post_std**2 + post_mean**2 - 2*post_mean*x0 + x0**2
        ref_valid.append(np.zeros_like(y))
    elif train_args['IO']['target'] == 'error':
        raise NotImplementedError('IO.target=error not implemented')
    else:
        raise ValueError('IO.target must be one of "mean", "square", "error"')

    x_valid.append(imgs[0])
    y_valid.append(y)
print ('done', flush=True)

x_train = np.array(x_train)[..., np.newaxis] - 1000 / train_args['Data']['norm']
x_valid = np.array(x_valid)[..., np.newaxis] - 1000 / train_args['Data']['norm']

y_train = (np.array(y_train)[..., np.newaxis] + train_args['Data']['offset_y']) * scale_y
y_valid = (np.array(y_valid)[..., np.newaxis] + train_args['Data']['offset_y']) * scale_y
ref_valid = np.array(ref_valid)[..., np.newaxis] * scale_y

if cmds is not None:
    plt.imshow(postprocess(y_valid[0, ..., 0] - ref_valid[0, ..., 0]) * train_args['Display']['norm_y'], 'gray', 
               vmin=train_args['Display']['vmin_y'], vmax=train_args['Display']['vmax_y'])
    plt.show()

#%%
os.environ['CUDA_VISIBLE_DEVICES'] = train_args['Train']['device']
K.clear_session()

# tensorboard
log_dir = os.path.join(output_dir, 'log')
if train_args['Train']['relog'] and os.path.exists(log_dir):
    shutil.rmtree(log_dir)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir = log_dir, histogram_freq=1)

# network
network = model.unet.UNet2D(**train_args['Network'])
_ = network.build()
network.model.summary(line_length = 120)

# learning rates
def scheduler(epoch, lr):
    epoch_list = train_args['Train']['epochs']
    lr_list = train_args['Train']['lr']
    for i in range(len(epoch_list)):
        if epoch < epoch_list[i]:
            return lr_list[i]
    return lr_list[-1]

lr_callback = tf.keras.callbacks.LearningRateScheduler(scheduler, verbose = 1)

# optimizer
optimizer = tf.keras.optimizers.Adam()
network.model.compile(optimizer, loss = tf.keras.losses.MeanSquaredError())

# saver
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    os.path.join(output_dir, '{epoch}.h5'), save_freq=len(x_train) * train_args['Train']['save_freq'], verbose = 1)
tmp_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    os.path.join(output_dir, 'tmp.h5'), save_freq=len(x_train), verbose = 0)

#%%
# snapshot callbacks
# first extract the snapshot slices
islice = train_args['Display']['islice']
snapshots_x = {train_args['IO']['target']: x_valid[[islice]]}
snapshots_y = {train_args['IO']['target']: postprocess(y_valid[[islice]] - ref_valid[[islice]])}
snapshots_ref = {train_args['IO']['target']: ref_valid[[islice]]}
snapshot_writer = tf.summary.create_file_writer(os.path.join(log_dir, 'snapshots'))
display_args = train_args['Display']

snapshot_callback = model.callbacks.TensorboardSnapshotCallback(
    network.model, snapshot_writer, snapshots_x, snapshots_y, snapshots_ref, postprocess = postprocess,
    norm_x = display_args['norm_x'], vmin_x = display_args['vmin_x'], vmax_x = display_args['vmax_x'], 
    norm_y = display_args['norm_y'], vmin_y = display_args['vmin_y'], vmax_y = display_args['vmax_y'], )

#%%
# validation callback
validation_callback = model.callbacks.SaveValid2DImageCallback(
    network.model, x={train_args['IO']['target']: x_valid}, y={train_args['IO']['target']: y_valid},
    output_dir=os.path.join(output_dir, 'valid'), interval=train_args['Train']['save_freq'], 
    postprocess = postprocess, norm_x = display_args['norm_x'], norm_y = display_args['norm_y']
)

#%%
network.model.fit(x_train, y_train, batch_size = train_args['Data']['batch_size'], 
                  epochs = train_args['Train']['epochs'][-1], shuffle = True, 
                  max_queue_size = 10, workers = 4, use_multiprocessing = False, verbose = verbose, 
                  callbacks = [lr_callback, snapshot_callback, validation_callback, tensorboard_callback, checkpoint_callback, tmp_checkpoint_callback])