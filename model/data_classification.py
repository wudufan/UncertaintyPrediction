'''
data generator
'''

import tensorflow as tf
import numpy as np
import h5py
import pandas as pd
import os
import scipy.ndimage

class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, manifest, data_dir, tag = None, regression = False, 
                 batch_size = 32, batch_size_pos = None, shuffle = True, preload = False, exclude_intermediate = True,
                 shape = (26, 40, 40), norm = 1000, 
                 zoom = 0.1, offset = (3, 5, 5), flip = True, noise_std = 0.02, noise_prob = 0.5):
        '''
        @params
        @manifest: path to the dataset manifest, or the manifest itself (pd.DataFrame)
        @data_dir: directory to the dataset. Each h5 file is data_dir/manifest.MRN
        @regression: if true, for each sample will return [label, z, y, x, diameter]. Otherwise only return label
        @tag: tag to be included
        @batch_size: batch size
        @batch_size_pos: how many positive samples in one batch. If None, sample unifromly from the batch
        @exclude_intermediate: if true then exclude the intermediate patches (y=-1) in the indices. Set to false in the final testing. Only works when batch_size_pos = None
        '''

        # dataset
        if isinstance(manifest, str):
            self.manifest = pd.read_csv(manifest)
        else:
            self.manifest = manifest
        self.data_dir = data_dir
        self.tag = tag
        if self.tag is not None:
            self.manifest = self.manifest[self.manifest['Tag'] == tag].reset_index(drop=True)
        
        self.regression = regression

        # shape
        self.batch_size = batch_size
        self.batch_size_pos = batch_size_pos
        self.shuffle = shuffle
        self.shape = shape
        self.norm = norm

        # augmentation
        self.zoom = zoom
        self.offset = offset
        self.flip = flip
        self.noise_std = noise_std
        self.noise_prob = noise_prob

        self.preload = preload
        self.exclude_intermediate = exclude_intermediate

        if self.preload:
            self.preload_all_data()
        
        # initialize the indices
        self.on_epoch_end()
    
    def __len__(self):
        return np.sum([int(np.ceil(len(s) / self.batch_size)) for s in self.data_indices])
    
    def preload_all_data(self):
        print ('Preloading all the data (%d)'%len(self.manifest), end='...', flush=True)

        self.datasets = []
        for i, row in self.manifest.iterrows():
            print (i+1, end=',', flush=True)
            with h5py.File(os.path.join(self.data_dir, row.MRN), 'r') as f:
                x = np.copy(f['x'])
                y = np.copy(f['y'])
                center = np.copy(f['center'])
                if self.regression:
                    diameter = np.copy(f['label_diameter'])
            if self.regression:
                self.datasets.append({'x': x, 'y': y, 'center': center, 'label_diameter': diameter})
            else:
                self.datasets.append({'x': x, 'y': y, 'center': center})
        
        print ('done', flush=True)

    def on_epoch_end(self):
        super().on_epoch_end()

        # generate index for the manifest record (each h5 file)
        self.file_indices = np.arange(len(self.manifest))
        if self.shuffle:
            np.random.shuffle(self.file_indices)
        # generate index within each h5
        self.data_indices = []
        for file_ind in self.file_indices:
            if self.preload:
                y = np.copy(self.datasets[file_ind]['y'])
            else:
                mrn = self.manifest.loc[file_ind, 'MRN']
                with h5py.File(os.path.join(self.data_dir, mrn), 'r') as f:
                    y = np.copy(f['y'])
            if self.batch_size_pos != None:
                batch_size_neg = self.batch_size - self.batch_size_pos
                # balance the positive and negative samples
                pos_inds = np.where(y == 1)[0]
                neg_inds = np.where(y == 0)[0]
                if self.shuffle:
                    np.random.shuffle(pos_inds)
                # tile the negative indices
                neg_tile = int(np.ceil(len(pos_inds) / self.batch_size_pos * batch_size_neg / len(neg_inds)))
                neg_inds_all = []
                for i in range(neg_tile):
                    inds = np.copy(neg_inds)
                    if self.shuffle:
                        np.random.shuffle(inds)
                    neg_inds_all += list(inds)
                neg_inds_all = np.array(neg_inds_all)
                # merge the positive and negative inds
                data_inds = []
                for i in range(0, len(pos_inds), self.batch_size_pos):
                    i_neg = int(i / self.batch_size_pos * batch_size_neg)
                    data_inds = data_inds + list(pos_inds[i:i+self.batch_size_pos]) + list(neg_inds_all[i_neg:i_neg+batch_size_neg])
                self.data_indices.append(np.array(data_inds))
            else:
                # all data uniformly sampled
                if self.exclude_intermediate:
                    data_inds = np.where(y >= 0)[0]
                else:
                    data_inds = np.arange(len(y))
                if self.shuffle:
                    np.random.shuffle(data_inds)
                self.data_indices.append(data_inds)
    
    def __getitem__(self, index):
        '''
        Get one batch of data
        '''

        # first get the file to be opened
        total_length = 0
        for i in range(len(self.data_indices)):
            current_length = int(np.ceil(len(self.data_indices[i]) / self.batch_size))
            total_length += current_length
            if total_length > index:
                break
        
        # get the index to read
        local_index = index - (total_length - current_length)
        inds = np.unique(self.data_indices[i][local_index * self.batch_size : (local_index + 1) * self.batch_size])
        
        # load data
        if self.preload:
            k = self.file_indices[i]
            x = np.copy(self.datasets[k]['x'][inds])
            y = np.copy(self.datasets[k]['y'][inds])
            center = np.copy(self.datasets[k]['center'][inds])
            if self.regression:
                diameter = np.copy(self.datasets[k]['label_diameter'][inds])
        else:
            # open the h5 file
            # h5py only accept ascending indices
            sorted_inds = list(np.sort(inds))  
            with h5py.File(os.path.join(self.data_dir, self.manifest.loc[self.file_indices[i], 'MRN']), 'r') as f:
                x = np.copy(f['x'][sorted_inds])
                y = np.copy(f['y'][sorted_inds])
                center = np.copy(f['center'][sorted_inds])
                if self.regression:
                    diameter = np.copy(f['label_diameter'][sorted_inds])
            # sort the indices back
            origin_ind_map = np.searchsorted(sorted_inds, inds)
            x = x[origin_ind_map]
            y = y[origin_ind_map]
            center = center[origin_ind_map]
            if self.regression:
                diameter = diameter[origin_ind_map]
        
        if self.regression:
            patch = []
            y_reg = []
            for i in range(x.shape[0]):
                p, r = self.__extract_patch__(x[i], center[i], diameter[i])
                patch.append(p)
                y_reg.append(r)
            return np.array(patch), np.concatenate([y[:, np.newaxis], np.array(y_reg)], -1)
        else:
            patch = np.array([self.__extract_patch__(x[i], center[i]) for i in range(x.shape[0])])
            return patch, y

    def __extract_patch__(self, x, center, diameter = 0):
        '''
        Extract patch from the volume with augmentation

        @params
        @x: np.array of size [nz, ny, nx], the volume
        @center: vector of size 3: (z, y, x), the sampling center

        @return
        @patch: the extracted patch
        '''
        translation = np.random.uniform(-1, 1, size = 3) * np.array(self.offset)
        if self.flip:
            flip = np.random.randint(0, 2, size = 3)
        else:
            flip = np.zeros([3], int)
        zoom = np.random.uniform(1 - self.zoom, 1 + self.zoom, 3)
        
        # calculate the patch to be extracted
        new_center = np.round(center + translation).astype(int)
        new_size = np.round(np.array(self.shape) * zoom).astype(int)
        # truncate the size to make sure it can fit in
        for i in range(3):
            new_size[i] = min(new_size[i], x.shape[i])
        # get the starting coordinates
        start_coords = []
        for i in range(3):
            half_size = new_size[i] // 2
            coord = min(max(0, new_center[i] - half_size), x.shape[i] - new_size[i])
            start_coords.append(coord)
        start_coords = np.array(start_coords)

        # extract the patch
        patch = x[start_coords[0]:start_coords[0] + new_size[0], 
                  start_coords[1]:start_coords[1] + new_size[1],
                  start_coords[2]:start_coords[2] + new_size[2]]
        # zoom 
        patch_zoom = np.array(self.shape) / np.array(patch.shape)
        patch = scipy.ndimage.zoom(patch, patch_zoom, order=1, mode='nearest').astype(np.float32)

        if self.regression:
            relative_center = (center - start_coords - new_size / 2) / new_size
            relative_diameter = diameter / np.sqrt(new_size[1] * new_size[2])
            label_reg = np.array(list(relative_center) + [relative_diameter])

        # normalization
        patch = patch / self.norm

        # flip
        if flip[0]:
            patch = patch[::-1, :, :]
        if flip[1]:
            patch = patch[:, ::-1, :]
        if flip[2]:
            patch = patch[:, :, ::-1]
        
        # noise
        if np.random.uniform() < self.noise_prob:
            patch = patch + self.noise_std * np.random.normal(size = patch.shape)
        
        if self.regression:
            return patch, label_reg
        else:
            return patch


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    data_dir = '/home/dwu/trainData/aneurysm/80_20_hard_fpr/20210407'
    manifest = os.path.join(data_dir, 'manifest.csv')
    manifest = pd.read_csv(manifest)
    manifest = manifest[manifest.Tag == 'test']
    manifest = manifest.iloc[[0]].reset_index(drop = True)
    data = DataGenerator(manifest, data_dir, 'test', regression = True, batch_size_pos = 4, shuffle = True, noise_prob = 1)

    ind_data = np.random.randint(len(data))
    # x,y = data[ind_data]
    x, y = data[ind_data]
    # i1 = np.random.choice(np.where(y == 1)[0])
    # i2 = np.random.choice(np.where(y == 0)[0])
    
    # print (y[i1], y[i2])
    # plt.figure(figsize=[8,4])
    # plt.subplot(121); plt.imshow(x[i1, x.shape[1]//2, ...], 'gray', vmin=0, vmax=0.5)
    # plt.subplot(122); plt.imshow(x[i2, x.shape[1]//2, ...], 'gray', vmin=0, vmax=0.5)



    '''
    mrn = data.manifest.loc[data.file_indices[0], 'MRN']
    inds = data.data_indices[0]
    with h5py.File(os.path.join(data.data_dir, mrn), 'r') as f:
        y = np.copy(f['y'])
    print (np.unique(y))
    print (np.unique(y[inds]))
    for i in range(0, len(inds), data.batch_size):
        print (i//data.batch_size, y[inds[i:i+data.batch_size]])
    '''

    