'''
generate mayo dataset for 2D. use 3 layer mean
'''

# %%
import pandas as pd
import SimpleITK as sitk
import numpy as np
import os
import glob

# %%
input_dir = '/home/dwu/trainData/uncertainty_prediction/data/mayo/'
output_dir = '/home/dwu/trainData/uncertainty_prediction/data/mayo_2d_3_layer_mean'
nslices_per_img = 100
nslices_mean = 3
spacing = [0.75, 0.75, 1]

REGEN_MANIFEST = False

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# %%
# generate manifest to sample
if REGEN_MANIFEST:
    np.random.seed(0)
    records = []
    dose_levels = [s for s in glob.glob(os.path.join(input_dir, '*')) if os.path.isdir(s)]
    filenames = glob.glob(os.path.join(dose_levels[0], '*.nii'))

    ind = 0
    for filename in filenames:
        imgname = os.path.basename(filename).split('_')[0]
        print(imgname, end=', ', flush=True)

        img = sitk.GetArrayFromImage(sitk.ReadImage(filename))

        islices = np.random.choice(np.arange(img.shape[0] - nslices_mean + 1), size=nslices_per_img, replace=False)
        islices = np.sort(islices)
        for islice in islices:
            records.append({'Index': ind, 'Tag': imgname, 'Slice': islice})
            ind += 1

    records = pd.DataFrame(records)
    records.to_csv(os.path.join(output_dir, 'manifest.csv'), index=False)
else:
    dose_levels = [s for s in glob.glob(os.path.join(input_dir, '*')) if os.path.isdir(s)]
    records = pd.read_csv(os.path.join(output_dir, 'manifest.csv'))

# %%
# generate nii images for each dose level
imgnames = records.Tag.drop_duplicates().values
for dose in dose_levels:
    print(dose, flush=True)
    dataset = []
    for imgname in imgnames:
        sub_records = records[records.Tag == imgname]
        filename = glob.glob(os.path.join(dose, imgname + '_*.nii'))[0]
        img = sitk.GetArrayFromImage(sitk.ReadImage(filename))
        for islice in sub_records.Slice.values:
            dataset.append(img[islice:islice + nslices_mean].mean(0).astype(np.int16))

    dataset = np.array(dataset)
    sitk_dataset = sitk.GetImageFromArray(dataset)
    sitk_dataset.SetSpacing(spacing)
    sitk.WriteImage(sitk_dataset, os.path.join(output_dir, os.path.basename(dose) + '.nii'))
