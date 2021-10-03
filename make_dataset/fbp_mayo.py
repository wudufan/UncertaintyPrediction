'''
fbp of the mayo dataset with different noise levels
'''

# %%
import numpy as np
import sys
import matplotlib.pyplot as plt
import os
import SimpleITK as sitk
import h5py
import argparse

# In[3]:
sys.path.append('..')
import CTProjector.projector.ct_projector as ct_projector

# %%
parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', default='/home/dwu/data/lowdoseCTsets/')
parser.add_argument('--output_dir', default='/home/dwu/trainData/uncertainty_prediction/data/mayo/dose_rate_1')
parser.add_argument('--name', default='L291_full_sino')
parser.add_argument('--geometry', default='/home/dwu/data/lowdoseCTsets/geometry.cfg')

parser.add_argument('--device', dest='device', type=int, default=0)

parser.add_argument('--N0', type=float, default=2e5)
parser.add_argument('--dose_rate', type=float, default=1)
parser.add_argument('--imgNorm', type=float, default=0.019)

if 'ipykernel' in sys.argv[0]:
    showplot = True
    args = parser.parse_args(['--device', '1'])
else:
    showplot = False
    args = parser.parse_args()

for v in vars(args):
    print(v, '=', getattr(args, v), flush=True)

projector = ct_projector.ct_projector()
projector.from_file(args.geometry)
print('\n[projector]')
for v in vars(projector):
    print(v, '=', getattr(projector, v), flush=True)

# %%
# load projection
print('Loading projection', end='...', flush=True)
with h5py.File(os.path.join(args.input_dir, args.name + '.mat'), 'r') as f:
    prj = np.copy(f['sino']).transpose([1, 0, 2]).astype(np.float32) / args.imgNorm
    prj = np.copy(prj, 'C')[np.newaxis]
print('Done')

# %%
# add noise
np.random.seed(0)
if args.N0 > 0:
    prj = prj + np.sqrt((1 - 1 / args.dose_rate) * args.dose_rate * np.exp(prj * args.imgNorm) / args.N0) \
        * np.random.normal(size=prj.shape) / args.imgNorm
    prj = prj.astype(np.float32)

# %%
print('Reconstruction', end='...', flush=True)
projector.set_device(args.device)
angles = projector.get_angles()
recons = []
batchsize = 32
for i in range(0, prj.shape[2], batchsize):
    current_prj = np.copy(prj[..., i:i + batchsize, :], 'C')
    projector.nz = current_prj.shape[2]
    sprj = projector.ramp_filter(current_prj)
    recon = projector.fbp_fan_bp(sprj, angles)
    recons.append(recon)
recons = np.concatenate(recons, 1)[0]
print('Done', flush=True)

# %%
if showplot:
    plt.figure(figsize=[8, 8])
    plt.imshow(recons[100], 'gray', vmin=0.84, vmax=1.24)

# %%
print('Saving', end='...', flush=True)
img = (recons * 1000 - 1000).astype(np.int16)
sitk_img = sitk.GetImageFromArray(img)
sitk_img.SetSpacing([float(projector.dx), float(projector.dy), float(projector.dz)])
if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)
sitk.WriteImage(sitk_img, os.path.join(args.output_dir, args.name.replace('sino', 'hann') + '.nii'))
print('Done', flush=True)
