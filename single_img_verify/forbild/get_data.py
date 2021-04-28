'''
Get the Forbild head phantom
'''

#%%
import scipy.io
import matplotlib.pyplot as plt
import SimpleITK as sitk
import numpy as np

#%%
x = scipy.io.loadmat('/home/dwu/data/phantoms/forbild512.mat')['ph']

# change it to a higher contrast
ref_level = 1.05
zoom_range = [1, 1.1]
zoom = 10
y = np.copy(x)
for v in np.unique(x):
    if v >= zoom_range[0] and v <= zoom_range[1]:
        target_val = ref_level + (v - ref_level) * zoom
    else:
        target_val = v
    y[x == v] = target_val

# add a few variance disks
rads = [15, 10, 5]

disk_xs = [140, 175, 200]
disk_ys = [130, 170, 210]
# create templates
masks = []
for r in rads:
    mask = np.zeros([r*2 + 1, r*2 + 1], np.uint8)
    for iy in range(mask.shape[0]):
        for ix in range(mask.shape[1]):
            if (ix-r)**2 + (iy-r)**2 <= r**2:
                mask[iy,ix] = 1
    masks.append(mask)

v_map = np.zeros(y.shape, np.uint8)
for icontrast, iy in enumerate(disk_ys):
    for irad, ix in enumerate(disk_xs):
        sy = iy - rads[irad]
        sx = ix - rads[irad]
        v_map[sy:sy+masks[irad].shape[0], sx:sx+masks[irad].shape[1]] = masks[irad] * (1 + icontrast)

plt.figure(figsize=(15,5))
plt.subplot(131); plt.imshow(y, 'gray', vmin=0.84, vmax=1.24)
plt.subplot(132); plt.imshow(v_map, 'gray')
plt.subplot(133); plt.imshow(y + v_map * 0.1, 'gray', vmin=0.84, vmax=1.24)

#%% 
# output
sitk.WriteImage(sitk.GetImageFromArray(((y * 1000) - 1000).astype(np.int16)), 'x0.nii')
sitk.WriteImage(sitk.GetImageFromArray(v_map[np.newaxis,...]), 'variance.seg.nrrd')
