'''
Data generator
'''

from typing import List, Tuple, Union
import tensorflow as tf
import numpy as np
import SimpleITK as sitk
import pandas as pd
import os
import copy

class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, 
                 manifest: Union[str, pd.DataFrame],
                 src_niis: List[Union[str, tuple]],
                 dst_niis: List[Union[str, tuple]],
                 tags: List[str], 
                 patch_size: Tuple[int] = (40, 40), 
                 num_patches_per_slice: int = 100,
                 num_slices_per_batch: int = 1,
                 shuffle: bool = True,
                 norm: float = 1000, 
                 flip: bool = True,
                 verbose: int = 0,
                 ):
        '''
        @params:
        @manifest: csv files recording which tag each slice belongs to
        @src_niis: list of strings or tuples. Each elements give the path to the nii file to extract x from. 
                   If the element is a tuple of strings, it will extract the same slices from each of the nii files and concatenate along the channel dimension.
        @dst_niis: list of strings or tuples. Each elements give the path to the nii file to extract y from. 
                   If the element is a tuple of strings, it will extract the same slices from each of the nii files and concatenate along the channel dimension.
                   If len(dst_niis) == 1, all the src_nii are mapped to the same dst_nii. Otherwise it should be a one-to-one mapping. 
        @tags: the list of tags to extract data based on the tags in the manifest
        @patch_size: patch size
        @num_patches_per_slice: how many patches to extract from each slice
        @num_slices_per_batch: how many slices per batch
        @shuffle: whether to shuffle slices and nii reads 
        @norm: the image value is divided by norm
        @flip: whether to do random flips along x and y
        '''

        if isinstance(manifest, str):
            self.manifest = pd.read_csv(manifest)
        else:
            self.manifest = manifest
        self.manifest = self.manifest[self.manifest['Tag'].isin(tags)]

        assert(len(src_niis) == len(dst_niis) or len(dst_niis) == 1)
        self.src_niis = copy.deepcopy(src_niis)
        if len(dst_niis) == 1:
            self.dst_niis = dst_niis * len(self.src_niis)
        else:
            self.dst_niis = copy.deepcopy(dst_niis)
        
        # convert the single string to tuples
        for i, f in enumerate(self.src_niis):
            if isinstance(f, str):
                self.src_niis[i] = tuple([f])
        
        for i, f in enumerate(self.dst_niis):
            if isinstance(f, str):
                self.dst_niis[i] = tuple([f])
        
        self.patch_size = patch_size
        self.num_patches_per_slice = num_patches_per_slice
        self.num_slices_per_batch = num_slices_per_batch
        self.shuffle = shuffle
        self.norm = norm
        self.flip = flip
        self.verbose = verbose
        
        self._current_nii_ind = -1 # current loaded nii index
        self.on_epoch_end()
    
    def batch_size(self):
        return self.num_patches_per_slice * self.num_slices_per_batch
    
    def num_batches_per_nii(self):
        return int(np.ceil(len(self.manifest) / self.num_slices_per_batch))

    def __len__(self):
        return self.num_batches_per_nii() * len(self.src_niis)
    
    def on_epoch_end(self):
        super().on_epoch_end()

        # indices
        self.nii_inds = np.arange(len(self.src_niis))
        self.slice_inds = [np.arange(len(self.manifest))] * len(self.nii_inds)
        if self.shuffle:
            np.random.shuffle(self.nii_inds)
            for inds in self.slice_inds:
                np.random.shuffle(inds)

    def __load_nii__(self, filename: str):
        img = sitk.GetArrayFromImage(sitk.ReadImage(filename)).astype(np.float32) / self.norm
        img = img[self.manifest['Index'].values][..., np.newaxis]

        return img

    def __getitem__(self, index: int):
        '''
        Get one batch
        '''
        assert(index < self.__len__() and index >= -self.__len__())

        # the nii index
        nii_ind = self.nii_inds[index // self.num_batches_per_nii()]
        # the slice index
        slice_batch = index % self.num_batches_per_nii()
        slice_ind = self.slice_inds[nii_ind][slice_batch*self.num_slices_per_batch : (slice_batch+1)*self.num_slices_per_batch]

        # reload nii images if needed
        if nii_ind != self._current_nii_ind:
            if self.verbose > 0:
                print ('Loading dataset %d'%(nii_ind))
                for filename in self.src_niis[nii_ind]:
                    print ('x: %s'%filename)
                for filename in self.dst_niis[nii_ind]:
                    print ('y: %s'%filename, flush=True)
            self.x = np.concatenate([self.__load_nii__(filename) for filename in self.src_niis[nii_ind]], -1)
            self.y = np.concatenate([self.__load_nii__(filename) for filename in self.dst_niis[nii_ind]], -1)

            self._current_nii_ind = nii_ind

        patches = self.__extract_patches__([self.x[slice_ind], self.y[slice_ind]])
        x = patches[0]
        y = patches[1]

        return x, y

    def __extract_patches__(self, imgs: list):
        '''
        For each element in imgs, generate the corresponding 2d patches
        '''

        ny = imgs[0].shape[1]
        nx = imgs[0].shape[2]
        py = ny if self.patch_size[0] < 0 else self.patch_size[0]
        px = nx if self.patch_size[1] < 0 else self.patch_size[1]

        # use [[]] * len(imgs) will shallow copy the list to each element
        patches = [[] for _ in range(len(imgs))]

        # first generate x y coordinates
        for islice in range(self.num_slices_per_batch):
            iys = np.random.randint(0, ny - py + 1, self.num_patches_per_slice)
            ixs = np.random.randint(0, nx - px + 1, self.num_patches_per_slice)

            for k, img in enumerate(imgs):
                for ii, (iy, ix) in enumerate(zip(iys, ixs)):
                    patches[k].append(img[islice, iy:iy+py, ix:ix+px, :])

        for k, p in enumerate(patches):
            patches[k] = np.array(p)
        
        # augmentation by flipping
        if self.flip:
            flip_opt = np.random.randint(0, 4, len(patches[0]))
            for k in range(len(patches)):
                for i in range(len(patches[k])):
                    if flip_opt[i] == 1:
                        patches[k][i] = patches[k][i][:, ::-1, :]
                    elif flip_opt[i] == 2:
                        patches[k][i] = patches[k][i][::-1, :, :]
                    elif flip_opt[i] == 3:
                        patches[k][i] = patches[k][i][::-1, ::-1, :]

        
        return patches
            

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    np.random.seed(0)
    
    manifest = '/home/dwu/trainData/deep_denoiser_ensemble/data/mayo_2d_3_layer_mean/manifest.csv'
    df = pd.read_csv(manifest)
    src_niis = [('/home/dwu/trainData/deep_denoiser_ensemble/data/mayo_2d_3_layer_mean/dose_rate_4.nii', 
                 '/home/dwu/trainData/deep_denoiser_ensemble/data/mayo_2d_3_layer_mean/dose_rate_2.nii'), 
                ('/home/dwu/trainData/deep_denoiser_ensemble/data/mayo_2d_3_layer_mean/dose_rate_8.nii', 
                 '/home/dwu/trainData/deep_denoiser_ensemble/data/mayo_2d_3_layer_mean/dose_rate_2.nii'),]
    dst_niis = ['/home/dwu/trainData/deep_denoiser_ensemble/data/mayo_2d_3_layer_mean/dose_rate_1.nii']
    tags = np.random.choice(pd.unique(df['Tag']), 5, False)

    print (tags)

    generator = DataGenerator(manifest, src_niis, dst_niis, tags, patch_size = (-1,-1), num_patches_per_slice=1, num_slices_per_batch=4)
    x, y = generator[0]


    