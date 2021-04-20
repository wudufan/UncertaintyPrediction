'''
Test the trained network
'''

import pandas as pd
import tensorflow as tf
import sys
import argparse
import os
import numpy as np
import sklearn.metrics
import matplotlib.pyplot as plt

sys.path.append('..')
import model.data

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir')
parser.add_argument('--manifest', default = None)
parser.add_argument('--patches', default = None)
parser.add_argument('--checkpoint')
parser.add_argument('--tag', default='valid,test')
parser.add_argument('--verbose', type=int, default=0)
parser.add_argument('--device', default='0')

args = parser.parse_args()

# args = parser.parse_args([
#     # '--checkpoint', '/home/dwu/trainData/aneurysm/80_20_hard_fpr_new/train_400_th_0.25_NMS_0.25/train/DenseNet/size_26x40x40/plain_focal_alpha_0.25_gamma_2/best.h5',
#     # '--checkpoint', '/home/dwu/trainData/aneurysm/80_20_hard_fpr_new/train_400_th_0.25_NMS_0.25/train/ResNet/size_32x48x48_deep/plain_focal_alpha_0.25_gamma_2/best.h5',
#     '--checkpoint', '/home/dwu/trainData/aneurysm/80_20_hard_fpr_new/train_400_th_0.25_NMS_0.25/train/ResNet/size_16x32x32_deep/hard_4/best.h5',
#     # '--checkpoint', '/home/dwu/trainData/aneurysm/80_20_hard_fpr_new/train_400_th_0.25_NMS_0.25/train/MobileNet/size_26x40x40/hard_4/best.h5',
#     # '--checkpoint', '/home/dwu/trainData/aneurysm/80_20_hard_fpr_new/train_400_th_0.25_NMS_0.25/train/PlainCNN/size_32x48x48/hard_4/best.h5',
#     # '--checkpoint', '/home/dwu/trainData/aneurysm/80_20_hard_fpr_dlca/train/train/size_32x32x32_dlca/hard_4/best.h5',
#     '--data_dir', '/home/dwu/trainData/aneurysm/80_20_hard_fpr_new',
#     '--device', '3',
#     '--tag', 'valid,test',
#     '--verbose', '1',
#     ])

if args.manifest is None:
    args.manifest = os.path.join(args.data_dir, 'manifest.csv')
if args.patches is None:
    args.patches = os.path.join(args.data_dir, 'patches.csv')

for v in vars(args):
    print (v, '=', getattr(args, v))

os.environ['CUDA_VISIBLE_DEVICES'] = args.device
tf.keras.backend.clear_session()

# get network
network = tf.keras.models.load_model(args.checkpoint, compile = False)
# new_model = tf.keras.Model(inputs = network.input, outputs = network.layers[-3].output)
# get the data shape
shape = np.array(network.input.shape[1:4]).astype(int)

# manifest 
tags = args.tag.split(',')
manifest = pd.read_csv(args.manifest)
manifest = manifest[manifest.Tag.isin(tags)].reset_index(drop=True)

# patch manifest
patch_manifest = pd.read_csv(args.patches)
patch_manifest = patch_manifest[patch_manifest.Dataset.isin(manifest.MRN.values)].reset_index(drop=True)

# data generator
generator = model.data.DataGenerator(manifest, args.data_dir, shuffle = False, preload = True, exclude_intermediate = False, 
                                     shape = shape, zoom = 0, offset = (0,0,0), flip = False, noise_prob = 0)

# prediction
pred = network.predict(generator, verbose = args.verbose)

# write into the patch_manifest
patch_manifest['fpr'] = pred
patch_manifest.to_csv(args.checkpoint.replace('.h5', '.csv'), index=False)

'''
# original bbox
detection_prob = []
for bbox in patch_manifest.bbox.values:
    s = [s for s in bbox[1:-1].split(' ') if len(s) > 0]
    detection_prob.append(float(s[0]))
detection_prob = np.array(detection_prob)

# brief evaluation
label = []
for i in range(len(generator)):
    _, y = generator[i]
    label.append(y)
label = np.concatenate(label)
label[label < 0] = 0

inds = np.where((patch_manifest.Tag.values != 'label') & (patch_manifest.Tag.values != 'fn'))

fprs, tprs, _ = sklearn.metrics.roc_curve(label[inds], pred[inds])
fprs2, tprs2, _ = sklearn.metrics.roc_curve(label[inds], detection_prob[inds])
plt.plot(fprs, tprs)
plt.plot(fprs2, tprs2)
'''