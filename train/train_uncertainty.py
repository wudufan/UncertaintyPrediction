'''
Training code for uncertainty, new interface.
'''
# %%
import sys
import os
import tensorflow as tf
import tensorflow.keras.backend as K
import shutil
import pandas as pd
import numpy as np
import argparse
import importlib
import functools

sys.path.append('..')
import model.data
import model.unet
import model.callbacks
import utils.config_manager

# %%
parser = argparse.ArgumentParser()
parser.add_argument('config')

if 'ipykernel' in sys.argv[0]:
    # debug
    print('debug')
    verbose = 1
    cmds = ['config/uncertainty/l2_depth_4_filter.cfg']
else:
    print('no debug')
    verbose = 2
    cmds = None

args, config, train_args = utils.config_manager.parse_config_with_extra_arguments(parser, cmds)

print(args.config)
for sec in train_args:
    print('[{0}]'.format(sec))
    for k in train_args[sec]:
        print(k, '=', train_args[sec][k])
    print('', flush=True)

# write the config file to the output directory
output_dir = os.path.join(train_args['IO']['output_dir'], train_args['IO']['tag'])
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
with open(os.path.join(output_dir, 'config.cfg'), 'w') as f:
    config.write(f)

# %%
os.environ['CUDA_VISIBLE_DEVICES'] = train_args['Train']['device']
K.clear_session()

# tensorboard
log_dir = os.path.join(output_dir, 'log')
if train_args['Train']['relog'] and os.path.exists(log_dir):
    shutil.rmtree(log_dir)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

# network
network = model.unet.UNet2D(**train_args['Network'])
_ = network.build()
network.model.summary(line_length=120)


# learning rates
def scheduler(epoch, lr):
    epoch_list = train_args['Train']['epochs']
    lr_list = train_args['Train']['lr']
    for i in range(len(epoch_list)):
        if epoch < epoch_list[i]:
            return lr_list[i]
    return lr_list[-1]


lr_callback = tf.keras.callbacks.LearningRateScheduler(scheduler, verbose=1)

# optimizer
optimizer = tf.keras.optimizers.Adam()
network.model.compile(optimizer, loss=tf.keras.losses.MeanSquaredError())

# %%
# filter
try:
    # get package name
    package_name = train_args['Filter']['package']
    func_name = train_args['Filter']['name']
    params = {k: train_args['Filter'][k] for k in train_args['Filter'] if k not in ['package', 'name']}
    mod_filter = importlib.import_module(package_name)
    func_filter = getattr(mod_filter, func_name)
    filter_y = functools.partial(func_filter, **params)
except Exception:
    print('No filters are used')
    filter_y = None

# %%
# data
manifest = pd.read_csv(train_args['IO']['manifest'])
all_tags = pd.unique(manifest['Tag'])
if train_args['IO']['train'] is None:
    train_args['IO']['train'] = [t for t in all_tags if t not in train_args['IO']['valid']]
print('Training tags:', train_args['IO']['train'])
print('Validation tags:', train_args['IO']['valid'], flush=True)
generator = model.data.Image2DGeneratorForUncertainty(
    manifest,
    train_args['IO']['src_datasets'],
    train_args['IO']['dst_datasets'],
    train_args['IO']['train'],
    filter_y=filter_y,
    **train_args['Data']
)
valid_generator = model.data.Image2DGeneratorForUncertainty(
    manifest,
    train_args['IO']['src_datasets'],
    train_args['IO']['dst_datasets'],
    train_args['IO']['valid'],
    patch_size=(640, 640),
    num_patches_per_slice=1,
    num_slices_per_batch=1,
    shuffle=False,
    norm=train_args['Data']['norm'],
    scale_y=train_args['Data']['scale_y'],
    filter_y=filter_y,
    flip=False
)

# saver
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    os.path.join(output_dir, '{epoch}.h5'), save_freq=len(generator) * train_args['Train']['save_freq'], verbose=1
)
tmp_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    os.path.join(output_dir, 'tmp.h5'), save_freq=len(generator), verbose=0
)

# %%
# snapshot callbacks
# first extract the snapshot slices
snapshots_x = {}
snapshots_y = {}
for i in range(len(valid_generator.src_datasets)):
    name = os.path.basename(valid_generator.src_datasets[i][0][:-3])
    x, y = valid_generator.load_slices(i, [train_args['Display']['islice']])
    snapshots_x[name] = x

    if filter_y is None:
        y = (y[..., [1]] - y[..., [0]]) ** 2 * train_args['Data']['scale_y'] * train_args['Data']['scale_y']
    else:
        y = (filter_y(y[..., [1]]) - filter_y(y[..., [0]])) ** 2\
            * train_args['Data']['scale_y'] * train_args['Data']['scale_y']
    # y = scipy.ndimage.gaussian_filter1d(y, 10, 1)
    # y = scipy.ndimage.gaussian_filter1d(y, 10, 2)

    snapshots_y[name] = np.sqrt(y)

snapshot_writer = tf.summary.create_file_writer(os.path.join(log_dir, 'snapshots'))
display_args = train_args['Display']


def postprocess(pred):
    pred[pred < 0] = 0
    return np.sqrt(pred)


snapshot_callback = model.callbacks.TensorboardSnapshotCallback(
    network.model,
    snapshot_writer,
    snapshots_x,
    snapshots_y,
    postprocess=postprocess,
    norm_x=display_args['norm_x'],
    vmin_x=display_args['vmin_x'],
    vmax_x=display_args['vmax_x'],
    norm_y=display_args['norm_y'],
    vmin_y=display_args['vmin_y'],
    vmax_y=display_args['vmax_y'],
)

# %%
# validation callback
validation_callback = model.callbacks.SaveValid2DCallback(
    network.model,
    valid_generator,
    os.path.join(output_dir, 'valid'),
    train_args['Train']['save_freq'],
    postprocess=postprocess,
    norm_x=display_args['norm_x'],
    norm_y=display_args['norm_y']
)

# %%
network.model.fit(
    generator,
    epochs=train_args['Train']['epochs'][-1],
    shuffle=False,
    max_queue_size=10,
    workers=4,
    use_multiprocessing=False,
    verbose=verbose,
    callbacks=[
        lr_callback,
        snapshot_callback,
        validation_callback,
        tensorboard_callback,
        checkpoint_callback,
        tmp_checkpoint_callback
    ]
)
