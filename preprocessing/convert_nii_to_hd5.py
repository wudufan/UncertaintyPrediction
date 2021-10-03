'''
Convert the nii files to hdf5 files for better generalizability
'''

import SimpleITK as sitk
import h5py
import os
import glob

working_dir = '/home/dwu/trainData/uncertainty_prediction/data/mayo_2d_3_layer_mean/'
input_files = glob.glob(os.path.join(working_dir, '*.nii'))

print(len(input_files))
for k, filename in enumerate(input_files):
    print(k, end=',', flush=True)

    sitk_img = sitk.ReadImage(filename)
    img = sitk.GetArrayFromImage(sitk_img)

    with h5py.File(os.path.join(working_dir, os.path.basename(filename).replace('.nii', '.h5')), 'w') as f:
        f['img'] = img
        f['spacing'] = sitk_img.GetSpacing()[::-1]
