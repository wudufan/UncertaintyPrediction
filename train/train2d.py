'''
Train base denoiser. This is using old config interface.
'''

# %%
import numpy as np
import os
import sys
import tensorflow as tf
from tensorflow.keras import backend as K
import pandas as pd
import SimpleITK as sitk
import shutil

# %%
sys.path.append('..')
import model.unet as unet
import utils.utils as utils

# %%
import argparse
import configparser
parser = argparse.ArgumentParser()
parser.add_argument('--config', default='./config/denoising/l2_depth_4.cfg')

# first get all the params in the cfg
args, _ = parser.parse_known_args()
cfg = configparser.ConfigParser()
cfg.read(args.config)

# then add all the parameters in the config to parser
for sec in cfg.sections():
    for val in cfg[sec]:
        parser.add_argument('--{0}'.format(sec + '.' + val), type=str, default=None)

if sys.argv[0] != 'train2d.py':
    args = parser.parse_args([
        '--Training.device', '1',
        '--Training.save_model_interval', '1',
        '--IO.train', 'L506',
        '--IO.source', 'dose_rate_4',
        '--IO.tag', 'l2_depth_4/dose_rate_4/debug',
        '--Window.vmin', '-160',
        '--Window.vmax', '240'
    ])
else:
    args = parser.parse_args()

# update the config
for k in vars(args):
    val = getattr(args, k)
    if val is not None and k != 'config':
        sec, key = k.split('.')
        cfg[sec][key] = val

# output the configuration
for sec in cfg.sections():
    for val in cfg[sec]:
        print('{0}.{1} = {2}'.format(sec, val, cfg[sec][val]))

# %%
# set device
os.environ['CUDA_VISIBLE_DEVICES'] = cfg['Training']['device']

# make output directory
outdir = os.path.join(cfg['IO']['outdir'], cfg['IO']['tag'])
logdir = os.path.join(outdir, 'log')
valid_dir = os.path.join(outdir, 'valid')
if not os.path.exists(valid_dir):
    os.makedirs(valid_dir)

# clear logdir if needed
if int(cfg['IO']['relog']):
    if os.path.exists(logdir):
        shutil.rmtree(logdir)
if not os.path.exists(logdir):
    os.makedirs(logdir)

# save the configuration to output directory
with open(os.path.join(outdir, 'config.cfg'), 'w') as f:
    cfg.write(f)


# %%
def load_img(filename, manifest, vmin=cfg['Window']['vmin'], vmax=cfg['Window']['vmax']):
    img = sitk.GetArrayFromImage(sitk.ReadImage(filename)).astype(np.float32) / 1000
    train = img[manifest[manifest.Dataset == 'train'].Index.values][..., np.newaxis]
    valid = img[manifest[manifest.Dataset == 'valid'].Index.values][..., np.newaxis]

    try:
        vmin = float(vmin) / 1000
        vmax = float(vmax) / 1000

        train = (train - vmin) / (vmax - vmin) * 2 - 1
        train[train < -1] = -1
        train[train > 1] = 1

        valid = (valid - vmin) / (vmax - vmin) * 2 - 1
        valid[valid < -1] = -1
        valid[valid > 1] = 1
    except Exception:
        pass

    return train, valid


def getshape(s):
    '''
    Convert string "nx,ny,nz" to shape [nx,ny,nz]
    '''
    try:
        return [int(i) for i in s.split(',')]
    except Exception:
        return None


def augment(img_list, flipx, flipy):
    flipx = np.random.randint(0, 2, len(img_list[0])) * flipx
    flipy = np.random.randint(0, 2, len(img_list[0])) * flipy

    for i in range(len(img_list)):
        for k in range(len(img_list[i])):
            if flipx[k]:
                img_list[i][k] = img_list[i][k][:, ::-1, :]
            if flipy[k]:
                img_list[i][k] = img_list[i][k][::-1, :, :]


# %%
# manifest
manifest = pd.read_csv(cfg['IO']['manifest'])
valid_list = cfg['IO']['valid'].split(',')
if 'train' not in cfg['IO'] or cfg['IO']['train'] == '':
    manifest['Dataset'] = 'train'
    manifest.loc[manifest.Tag.isin(valid_list), 'Dataset'] = 'valid'
else:
    train_list = cfg['IO']['train'].split(',')
    manifest.loc[manifest.Tag.isin(valid_list), 'Dataset'] = 'valid'
    manifest.loc[manifest.Tag.isin(train_list), 'Dataset'] = 'train'
# manifest.loc[manifest.Tag.isin(valid_list), 'Dataset'] = 'valid'

# kepp only train and valid
manifest = manifest[manifest.Dataset.isin(['train', 'valid'])]
assert(len(manifest[manifest.Dataset == 'train']) > 0)

manifest.to_csv(os.path.join(outdir, 'manifest.csv'), index=False)

# the training list
src_list = [os.path.join(cfg['IO']['datadir'], s + '.nii') for s in cfg['IO']['source'].split(',')]
dst_list = [os.path.join(cfg['IO']['datadir'], s + '.nii') for s in cfg['IO']['target'].split(',')]

# load the dataset
print('Loading', flush=True, end='...')
train_y, valid_y = load_img(dst_list[0], manifest)
if len(src_list) == 1:
    train_x, valid_x = load_img(src_list[0], manifest)
print('Done', flush=True)

# %%
# build model
K.clear_session()

model_cfg = cfg['Network']
model_params = {
    'down_features': getshape(model_cfg['down_features']),
    'up_features': getshape(model_cfg['up_features']),
    'bottleneck_features': int(model_cfg['bottleneck_features']),
    'lrelu': float(model_cfg['lrelu']),
    'strides': getshape(model_cfg['strides']),
    'use_adding': int(model_cfg['use_adding'])
}

imgshape = getshape(cfg['Training']['imgshape'])

lr = float(cfg['Training']['lr'])
unet_model = unet.UNet2D(input_shape=imgshape, output_channel=imgshape[-1], **model_params)
model = unet_model.build()
model.compile(optimizer=tf.keras.optimizers.Adam(lr), loss=tf.keras.losses.mean_squared_error)
model.summary()
print('', flush=True)

# tensorboard
tb_writer = tf.summary.create_file_writer(logdir)
tb_writer.set_as_default()

# %%
# load model
checkpoint = cfg['IO']['checkpoint']
if os.path.exists(checkpoint):
    model.load_weights(checkpoint)

# %%
tf.summary.trace_on(graph=True, profiler=True)

# %%
# display window
try:
    vmin = float(cfg['Window']['vmin']) / 1000
    vmax = float(cfg['Window']['vmax']) / 1000

    display_vmin = -1
    display_vmax = 1
except Exception:
    vmin = -1
    vmax = 1

    display_vmin = -0.16
    display_vmax = 0.24

# %%
np.random.seed(0)

cfg_train = cfg['Training']
nepoch = int(cfg_train['epoch'])
start_epoch = int(cfg_train['start_epoch'])
batchsize = int(cfg_train['batchsize'])
save_model_interval = int(cfg_train['save_model_interval'])
output_interval = int(cfg_train['output_interval'])

flipx = int(cfg['Augmentation']['flipx'])
flipy = int(cfg['Augmentation']['flipy'])

train_step = 0
for epoch in range(start_epoch, nepoch):
    print('Starting epoch {0}, lr = {1}'.format(epoch + 1, lr), flush=True)

    # generate the source dose sequence
    sample_src_list = np.copy(src_list)
    np.random.shuffle(sample_src_list)

    for isrc in range(len(sample_src_list)):
        # read the source image
        if len(sample_src_list) > 1:
            train_x, valid_x = load_img(sample_src_list[isrc], manifest)

        # generate the slice sequence
        islices = np.arange(len(train_x))
        np.random.shuffle(islices)

        for i in range(0, len(islices), batchsize):
            # sample the slices
            inds = islices[i:i + batchsize]

            batch_x = train_x[inds]
            batch_y = train_y[inds]
            augment([batch_x, batch_y], flipx, flipy)

            loss = model.train_on_batch(batch_x, batch_y)

            if (i // batchsize + 1) % output_interval == 0:
                print(
                    'epoch = {0}/{1}, src = {2}/{3}, batch = {4}/{5}, loss = {6}'.format(
                        epoch + 1,
                        nepoch,
                        isrc + 1,
                        len(sample_src_list),
                        i // batchsize + 1,
                        len(islices) // batchsize,
                        np.sqrt(loss)
                    ),
                    flush=True
                )

            # tensorboard
            tf.summary.scalar('train_loss', loss, step=train_step + 1)
            if train_step == 0:
                tf.summary.trace_export('graph', 1, logdir)
            train_step += 1

#             if i >= 50:
#                 break

    # save model
    model.save(os.path.join(outdir, 'tmp.h5'))
    if (epoch + 1) % save_model_interval == 0 or (epoch + 1) == nepoch:
        model.save(os.path.join(outdir, '{0}.h5'.format(epoch + 1)))

        print('Validation({0})'.format(len(src_list)), flush=True)
        # validation and testing
        l2_losses = []

        for isrc in range(len(src_list)):
            filename = os.path.basename(src_list[isrc])[:-4]
            print('{0}: {1}'.format(isrc + 1, filename), flush=True)

            if len(src_list) > 1:
                _, valid_x = load_img(src_list[isrc], manifest)

            preds = model.predict(valid_x, batch_size=1)

            l2_losses.append(np.sqrt(np.mean((preds - valid_y)**2)))

            # output
            display_silice = 99
            tf.summary.image(
                'valid-' + filename + '/pred',
                utils.snapshot(preds[..., 0], display_silice, vmin=display_vmin, vmax=display_vmax),
                step=epoch + 1
            )
            tf.summary.image(
                'valid-' + filename + '/x',
                utils.snapshot(valid_x[..., 0], display_silice, vmin=display_vmin, vmax=display_vmax),
                step=epoch + 1
            )
            tf.summary.image(
                'valid-' + filename + '/y',
                utils.snapshot(valid_y[..., 0], display_silice, vmin=display_vmin, vmax=display_vmax),
                step=epoch + 1
            )
            tb_writer.flush()

            utils.save_nii(preds[..., 0], os.path.join(valid_dir, filename + '.pred.nii'), vmin, vmax)
            utils.save_nii(valid_x[..., 0], os.path.join(valid_dir, filename + '.x.nii'), vmin, vmax)
            utils.save_nii(valid_y[..., 0], os.path.join(valid_dir, filename + '.y.nii'), vmin, vmax)

        tf.summary.scalar('valid/loss', np.mean(l2_losses), step=epoch + 1)
        tb_writer.flush()
