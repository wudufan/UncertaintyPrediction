#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import tensorflow as tf
from tensorflow.keras import backend as K
import pandas as pd
import glob
import SimpleITK as sitk
import shutil
import scipy.ndimage


# In[2]:


sys.path.append('..')
import model.unet as unet
import model.wgan_gp as wgan
import utils.utils as utils


# In[3]:


import argparse
import configparser
parser = argparse.ArgumentParser()
parser.add_argument('--config', default = './config/wgan/l2_depth_3.cfg')

# first get all the params in the cfg
args, _ = parser.parse_known_args()
cfg = configparser.ConfigParser()
cfg.read(args.config)

# then add all the parameters in the config to parser
for sec in cfg.sections():
    for val in cfg[sec]:
        parser.add_argument('--%s'%(sec + '.' + val), type=str, default = None)

if sys.argv[0] != 'train2d_wgan.py':
    args = parser.parse_args(['--Training.device', '0', 
                              '--Training.save_model_interval', '1',
#                               '--IO.train', 'L506',
                              '--IO.source', 'dose_rate_4', 
                              '--IO.tag', 'l2_depth_3_wgan/debug'])
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
        print ('%s.%s = %s'%(sec, val, cfg[sec][val]))


# In[4]:


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


# In[5]:


def load_img(filename, manifest, vmin = cfg['Window']['vmin'], vmax = cfg['Window']['vmax']):
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
    except Exception as e:
        pass
    
    return train, valid

def make_patches(imgs, n, patch_size):
    '''
    Make n patches with patch_size = [ny, nx]
    
    @params:
    @imgs: list of array of shape [ny, nx, nchannel]
    @n: number of patches
    @patch_size: [npy, npx]
    
    @return
    @list of array, each with shape of [n, npy, npx, nchannel]
    '''
    
    ny = imgs[0].shape[0]
    nx = imgs[0].shape[1]
    
    iys = np.random.randint(0, ny - patch_size[0] + 1, n)
    ixs = np.random.randint(0, nx - patch_size[1] + 1, n)
    patch_list = []
    for img in imgs:
        patches = []
        for iy, ix in zip(iys, ixs):
            patches.append(img[iy:iy+patch_size[0], ix:ix+patch_size[0], :])
        patch_list.append(np.array(patches))
    
    return patch_list

def getshape(s):
    '''
    Convert string "nx,ny,nz" to shape [nx,ny,nz]
    '''
    try:
        return [int(i) for i in s.split(',')]
    except Exception as _:
        return None

def getfloats(s):
    '''
    Convert string "x,y,z" to [x,y,z]
    '''
    try:
        return [float(i) for i in s.split(',')]
    except Exception as _:
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


# In[6]:


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
print ('Loading', flush=True, end='...')
train_y, valid_y = load_img(dst_list[0], manifest)
if len(src_list) == 1:
    train_x, valid_x = load_img(src_list[0], manifest)
print ('Done', flush=True)


# In[7]:


# build model
K.clear_session()

g_cfg = cfg['Network']
g_params = {'down_features': getshape(g_cfg['down_features']), 
            'up_features': getshape(g_cfg['up_features']), 
            'bottleneck_features': int(g_cfg['bottleneck_features']), 
            'lrelu': float(g_cfg['lrelu']), 
            'strides': getshape(g_cfg['strides']), 
            'use_adding': int(g_cfg['use_adding'])}

d_cfg = cfg['Discriminator']
d_params = {'features': getshape(d_cfg['features']), 
            'fc_features': getshape(d_cfg['fc_features']), 
            'strides': getshape(d_cfg['strides']), 
            'lrelu': float(d_cfg['lrelu']), 
            'dropouts': getfloats(d_cfg['dropouts']),
            'layer_norm': bool(d_cfg['layer_norm'])}

wgan_cfg = cfg['WGAN']
wgan_params = {'l2_weight': float(wgan_cfg['l2_weight']), 
               'gp_weight': float(wgan_cfg['gp_weight']), 
               'discriminator_steps': int(wgan_cfg['discriminator_steps'])}

# build networks
imgshape = getshape(cfg['Training']['imgshape'])

# learning rates
g_lr = float(cfg['Training']['lr'])
d_lr = float(cfg['Training']['lr_discriminator'])
g_optimizer = tf.keras.optimizers.Adam(g_lr)
d_optimizer = tf.keras.optimizers.Adam(d_lr)

unet_wrapper = unet.unet2d(input_shape = imgshape, **g_params)
g_model = unet_wrapper.build()
discriminator_wrapper = wgan.DiscriminatorResNet2D(input_shape = imgshape, **d_params)
d_model = discriminator_wrapper.build()

model = wgan.wgan_gp(generator = g_model, discriminator = d_model, **wgan_params)
model.compile(g_optimizer=g_optimizer, d_optimizer=d_optimizer)

# tensorboard
tb_writer = tf.summary.create_file_writer(logdir)
tb_writer.set_as_default()


# In[8]:


# load model
checkpoint = cfg['IO']['checkpoint']
if os.path.exists(checkpoint):
    model.load_weights(checkpoint)


# In[9]:


tf.summary.trace_on(graph = True, profiler=True)


# In[10]:


# display window
try:
    vmin = float(cfg['Window']['vmin']) / 1000
    vmax = float(cfg['Window']['vmax']) / 1000
    
    display_vmin = -1
    display_vmax = 1
except Exception as e:
    vmin = -1
    vmax = 1
    
    display_vmin = -0.16
    display_vmax = 0.24


# In[11]:


def lr_scheme(i_iter, niters, lr_start, lr_end):
    return lr_start - (lr_start - lr_end) * i_iter / niters


# In[12]:


np.random.seed(0)

cfg_train = cfg['Training']
nepoch = int(cfg_train['epoch'])
start_epoch = int(cfg_train['start_epoch'])
batchsize = int(cfg_train['batchsize'])
save_model_interval = int(cfg_train['save_model_interval'])
output_interval = int(cfg_train['output_interval'])

flipx = int(cfg['Augmentation']['flipx'])
flipy = int(cfg['Augmentation']['flipy'])

# learning rate scheme
g_lr_end = g_lr * float(cfg_train['lr_reduction_end'])
d_lr_end = d_lr * float(cfg_train['lr_reduction_end'])
lr_interval = int(cfg_train['lr_reduction_interval'])
niters_total = len(train_y) * len(src_list) * nepoch

print ('Total iterations = %g, g_lr = %g -> %g, d_lr = %g -> %g'%(niters_total, g_lr, g_lr_end, d_lr, d_lr_end))

train_step = 0
for epoch in range(start_epoch, nepoch):
    print ('Starting epoch %d'%(epoch+1), flush=True)
    
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
        
        for i in range(0, len(islices)):
            # make patch of one slice
            ind = islices[i]
            batch_x, batch_y = make_patches([train_x[ind], train_y[ind]], batchsize, (model.generator.inputs[0].shape[1], model.generator.inputs[0].shape[2]))
            
            augment([batch_x, batch_y], flipx, flipy)
            
            # set learning rate
            if (train_step + 1) % lr_interval == 0:
                g_lr_now = lr_scheme(train_step, niters_total, g_lr, g_lr_end)
                d_lr_now = lr_scheme(train_step, niters_total, d_lr, d_lr_end)
                print ('Setting g_lr = %g, d_lr = %g'%(g_lr_now, d_lr_now))
                K.set_value(model.g_optimizer.learning_rate, g_lr_now)
                K.set_value(model.d_optimizer.learning_rate, d_lr_now)
                
            loss = model.train_on_batch(batch_x, batch_y, return_dict = True)
            
            if (i+1) % output_interval == 0:
                print ('epoch = %d/%d, src = %d/%d, batch = %d/%d, d_loss = %g, g_cost = %g, l2_cost = %g'%(
                    epoch+1, nepoch, 
                    isrc+1, len(sample_src_list), 
                    i+1, len(islices), 
                    loss['d_loss'], loss['g_cost'], loss['l2_cost']), flush=True)

            # tensorboard
            tf.summary.scalar('d_loss', loss['d_loss'], step = train_step + 1)
            tf.summary.scalar('g_cost', loss['g_cost'], step = train_step + 1)
            tf.summary.scalar('l2_cost', loss['l2_cost'], step = train_step + 1)
            if train_step == 0:
                tf.summary.trace_export('graph', 1, logdir)
            train_step += 1
            
#             if i >= 50:
#                 break
        
    # save model
    model.save(os.path.join(outdir, 'wgan_tmp'), save_format='tf')
    model.generator.save(os.path.join(outdir, 'tmp.h5'))
    
    if (epoch + 1) % save_model_interval == 0 or (epoch + 1) == nepoch:
        model.save(os.path.join(outdir, 'wgan_%d'%(epoch+1)), save_format = 'tf')
        model.generator.save(os.path.join(outdir, '%d.h5'%(epoch+1)))
        
        print ('Validation(%d)'%(len(src_list)), flush=True)
        # validation and testing
        l2_losses = []

        for isrc in range(len(src_list)): 
            filename = os.path.basename(src_list[isrc])[:-4]
            print ('%d: %s'%(isrc+1, filename), flush=True)
            
            if len(src_list) > 1:
                _, valid_x = load_img(src_list[isrc], manifest)
            
            preds = model.generator.predict(valid_x, batch_size = 1)
            
            l2_losses.append(np.sqrt(np.mean((preds - valid_y)**2)))
            
            # output
            display_slice = 99
            tf.summary.image('valid-' + filename + '/pred', utils.snapshot(preds[...,0], display_slice, vmin=display_vmin, vmax=display_vmax), step = epoch + 1)
            tf.summary.image('valid-' + filename + '/x', utils.snapshot(valid_x[...,0], display_slice, vmin=display_vmin, vmax=display_vmax), step = epoch + 1)
            tf.summary.image('valid-' + filename + '/y', utils.snapshot(valid_y[...,0], display_slice, vmin=display_vmin, vmax=display_vmax), step = epoch + 1)
            tb_writer.flush()
            
            utils.save_nii(preds[...,0], os.path.join(valid_dir, filename + '.pred.nii'), vmin, vmax)
            utils.save_nii(valid_x[...,0], os.path.join(valid_dir, filename + '.x.nii'), vmin, vmax)
            utils.save_nii(valid_y[...,0], os.path.join(valid_dir, filename + '.y.nii'), vmin, vmax)
        
        tf.summary.scalar('valid/loss', np.mean(l2_losses), step = epoch+1)
        tb_writer.flush()


# In[ ]:




