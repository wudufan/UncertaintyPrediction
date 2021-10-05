'''
Test non-local mean denoiser
'''

# %%
import numpy as np
import cupy as cp
import h5py
import os
import sys
import argparse

sys.path.append('..')
import CTProjector.prior.recon_prior_cupy as recon_prior

# %%
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default=None)
parser.add_argument('--output_dir', default=None)

parser.add_argument('--device', default='1')
parser.add_argument('--norm', type=float, default=1000)

parser.add_argument('--d', type=float, default=0.02)
parser.add_argument('--search_size', type=int, default=11)
parser.add_argument('--kernel_size', type=int, default=5)
parser.add_argument('--kernel_std', type=float, default=2.5)

if 'ipykernel' in sys.argv[0]:
    print('debugging', flush=True)
    args = parser.parse_args([
        '--dataset', '/home/dwu/trainData/uncertainty_prediction/data/mayo_2d_3_layer_mean/dose_rate_4.h5',
        '--d', '0.05',
        '--output_dir',
        '/home/dwu/trainData/uncertainty_prediction/train/mayo_2d_3_layer_mean/'
        'dose_rate_4/denoising/nlm_0.05/results/',
    ])
    verbose = 1
else:
    args = parser.parse_args()
    verbose = 0

for k in vars(args):
    print(k, '=', getattr(args, k), flush=True)

# %%
# load model
cp.cuda.Device(int(args.device)).use()

# load data
with h5py.File(args.dataset, 'r') as f:
    x = np.copy(f['img']).astype(np.float32) / args.norm
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
with open(os.path.join(model_dir, 'config.cfg'), 'w') as f:
    f.write('[NLM]\n')
    f.write('norm={0}\n'.format(args.norm))
    f.write('d={0}\n'.format(args.d))
    f.write('search_size={0}\n'.format(args.search_size))
    f.write('kernel_size={0}\n'.format(args.kernel_size))
    f.write('kernel_std={0}\n'.format(args.kernel_std))

# %%
# prediction
pred = []
print(len(x))
for i in range(len(x)):
    if (i + 1) % 50 == 0:
        print(i + 1, end=',', flush=True)
    x_cp = cp.array(x[i], order='C')[cp.newaxis, cp.newaxis, ...]
    res = recon_prior.nlm(
        x_cp,
        x_cp,
        args.d,
        [1, args.search_size, args.search_size],
        [1, args.kernel_size, args.kernel_size],
        [1, args.kernel_std, args.kernel_std]
    )
    pred.append(res.get()[0, 0])
print('')
pred = np.array(pred)

# import matplotlib.pyplot as plt
# plt.figure(figsize=[16, 8])
# plt.subplot(121)
# plt.imshow(x[698, 64:-64, 64:-64], 'gray', vmin=-0.16, vmax=0.24)
# plt.subplot(122)
# plt.imshow(res.get()[0, 0, 64:-64, 64:-64], 'gray', vmin=-0.16, vmax=0.24)

# %%
# save the results
pred = (pred * args.norm).astype(np.int16)
with h5py.File(os.path.join(args.output_dir, os.path.basename(args.dataset)), 'w') as f:
    f['img'] = pred
    for k in attribs:
        f[k] = attribs[k]
