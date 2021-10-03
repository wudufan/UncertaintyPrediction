'''
Test trained denoising networks on h5 dataset
'''

# %%
import tensorflow as tf
import numpy as np
import h5py
import os
import sys
import shutil
import argparse

# %%
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default=None)
parser.add_argument('--checkpoint', default=None)
parser.add_argument('--output_dir', default=None)

parser.add_argument('--device', default='0')
parser.add_argument('--norm', type=float, defaul=1000)
parser.add_argument('--batch_size', type=int, default=4)

if 'ipykernel' in sys.argv[0]:
    print('debugging', flush=True)
    args = parser.parse_args([
        '--dataset', '/home/dwu/trainData/uncertainty_prediction/data/mayo_2d_3_layer_mean/dose_rate_4.h5',
        '--checkpoint',
        '/home/dwu/trainData/uncertainty_prediction/train/mayo_2d_3_layer_mean/l2_depth_3/dose_rate_4/25.h5',
        '--output_dir',
        '/home/dwu/trainData/uncertainty_prediction/denoising_results/mayo_2d_3_layer_mean/l2_depth_3/dose_rate_4/',
    ])
    verbose = 1
else:
    args = parser.parse_args()
    verbose = 0

for k in vars(args):
    print(k, '=', getattr(args, k), flush=True)

# %%
# load model
os.environ['CUDA_VISIBLE_DEVICES'] = args.device
model = tf.keras.models.load_model(args.checkpoint)

# load data
with h5py.File(args.dataset, 'r') as f:
    x = np.copy(f['img'])[..., np.newaxis].astype(np.float32) / args.norm
    attribs = {k: np.copy(f[k]) for k in f if k != 'img'}

print('Dataset attributes')
for k in attribs:
    print(k, '=', attribs[k], flush=True)

# set output directory
if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)
# copy model information
model_dir = os.path.join(args.output_dir, 'denoising_model')
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
shutil.copyfile(args.checkpoint, os.path.join(model_dir, os.path.basename(args.checkpoint)))
cfg_file = os.path.join(os.path.dirname(args.checkpoint), 'config.cfg')
if os.path.exists(cfg_file):
    shutil.copyfile(cfg_file, os.path.join(model_dir, 'config.cfg'))

# %%
# prediction
pred = model.predict(x, batch_size=args.batch_size, verbose=verbose)

# %%
# save the results
pred = (pred[..., 0] * args.norm).astype(np.int16)
with h5py.File(os.path.join(args.output_dir, os.path.basename(args.dataset)), 'w') as f:
    f['img'] = pred
    for k in attribs:
        f[k] = attribs[k]
