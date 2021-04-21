'''
Data generator
'''

from typing import List, Tuple, Union
import tensorflow as tf
import numpy as np
import pandas as pd
import copy
import h5py

class Image2DGenerator(tf.keras.utils.Sequence):
    def __init__(self, 
                 manifest: Union[str, pd.DataFrame],
                 src_datasets: List[Union[str, tuple]],
                 dst_datasets: List[Union[str, tuple]],
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
        @src_datasets: list of strings or tuples. Each elements give the path to the h5 file to extract x from. 
            If the element is a tuple of strings, it will extract the same slices from each of the h5 files and concatenate along the channel dimension.
        @dst_datasets: list of strings or tuples. Each elements give the path to the h5 file to extract y from. 
            If the element is a tuple of strings, it will extract the same slices from each of the h5 files and concatenate along the channel dimension.
            If len(dst_datasets) == 1, all the src_datasets are mapped to the same dst_datasets. Otherwise it should be a one-to-one mapping. 
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

        assert(len(src_datasets) == len(dst_datasets) or len(dst_datasets) == 1)
        self.src_datasets = copy.deepcopy(src_datasets)
        if len(dst_datasets) == 1:
            self.dst_datasets = dst_datasets * len(self.src_datasets)
        else:
            self.dst_datasets = copy.deepcopy(dst_datasets)
        
        # convert the single string to tuples
        for i, f in enumerate(self.src_datasets):
            if isinstance(f, str):
                self.src_datasets[i] = tuple([f])
        
        for i, f in enumerate(self.dst_datasets):
            if isinstance(f, str):
                self.dst_datasets[i] = tuple([f])
        
        self.patch_size = patch_size
        self.num_patches_per_slice = num_patches_per_slice
        self.num_slices_per_batch = num_slices_per_batch
        self.shuffle = shuffle
        self.norm = norm
        self.flip = flip
        self.verbose = verbose
        
        self._current_idataset = -1 # current loaded nii index
        self.on_epoch_end()
    
    def batch_size(self):
        return self.num_patches_per_slice * self.num_slices_per_batch
    
    def num_batches_per_dataset(self):
        return int(np.ceil(len(self.manifest) / self.num_slices_per_batch))

    def __len__(self):
        return self.num_batches_per_dataset() * len(self.src_datasets)
    
    def on_epoch_end(self):
        super().on_epoch_end()

        # indices
        self.dataset_indices = np.arange(len(self.src_datasets))
        self.slice_indices = [np.arange(len(self.manifest))] * len(self.dataset_indices)
        if self.shuffle:
            np.random.shuffle(self.dataset_indices)
            for inds in self.slice_indices:
                np.random.shuffle(inds)

    def __load_h5__(self, filename: str):
        with h5py.File(filename, 'r') as f:
            img = np.copy(f['img'][self.manifest['Index'].values])[..., np.newaxis].astype(np.float32) / self.norm
        
        return img
    
    def load_slices(self, idataset: int, islices: None):
        x = []
        y = []
        for filename in self.src_datasets[idataset]:
            with h5py.File(filename, 'r') as f:
                x.append(np.copy(f['img'][islices])[..., np.newaxis].astype(np.float32) / self.norm)
        
        for filename in self.dst_datasets[idataset]:
            with h5py.File(filename, 'r') as f:
                y.append(np.copy(f['img'][islices])[..., np.newaxis].astype(np.float32) / self.norm)
        
        x = np.concatenate(x, -1)
        y = np.concatenate(y, -1)

        return x, y

    # def __load_nii__(self, filename: str):
    #     img = sitk.GetArrayFromImage(sitk.ReadImage(filename)).astype(np.float32) / self.norm
    #     img = img[self.manifest['Index'].values][..., np.newaxis]

    #     return img

    def __getitem__(self, index: int):
        '''
        Get one batch
        '''
        assert(index < self.__len__() and index >= -self.__len__())

        # the nii index
        idataset = self.dataset_indices[index // self.num_batches_per_dataset()]
        # the slice index
        slice_batch = index % self.num_batches_per_dataset()
        islices = self.slice_indices[idataset][slice_batch*self.num_slices_per_batch : (slice_batch+1)*self.num_slices_per_batch]

        # reload nii images if needed
        if idataset != self._current_idataset:
            if self.verbose > 0:
                print ('Loading dataset %d'%(idataset))
                for filename in self.src_datasets[idataset]:
                    print ('x: %s'%filename)
                for filename in self.dst_datasets[idataset]:
                    print ('y: %s'%filename, flush=True)
            self.x = np.concatenate([self.__load_h5__(filename) for filename in self.src_datasets[idataset]], -1)
            self.y = np.concatenate([self.__load_h5__(filename) for filename in self.dst_datasets[idataset]], -1)

            self._current_idataset = idataset

        patches = self.__extract_patches__([self.x[islices], self.y[islices]])
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

class Image2DGeneratorForUncertainty(Image2DGenerator):
    '''
    Generate the target as (y - y0)^2
    '''
    def __init__(self, 
                 manifest: Union[str, pd.DataFrame],
                 src_datasets: List[Union[str, tuple]],
                 dst_datasets: List[tuple],
                 tags: List[str], 
                 patch_size: Tuple[int] = (40, 40), 
                 num_patches_per_slice: int = 100,
                 num_slices_per_batch: int = 1,
                 shuffle: bool = True,
                 norm: float = 1000, 
                 scale_y: float = 1,
                 flip: bool = True,
                 verbose: int = 0):
        
        # scale_y is used to bring y close to 1
        self.scale_y = scale_y

        assert (len(dst_datasets[0]) == 2)
        super().__init__(manifest, src_datasets, dst_datasets, tags, patch_size, num_patches_per_slice, num_slices_per_batch, shuffle, norm, flip, verbose)
    
    def __getitem__(self, index: int):
        x, y = super().__getitem__(index)

        return x, (y[..., [1]] - y[..., [0]]) ** 2 * self.scale_y * self.scale_y

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    np.random.seed(0)
    
    manifest = '/home/dwu/trainData/deep_denoiser_ensemble/data/mayo_2d_3_layer_mean/manifest.csv'
    df = pd.read_csv(manifest)
    tags = np.random.choice(pd.unique(df['Tag']), 5, False)
    print (tags)

    # src_datasets = [('/home/dwu/trainData/deep_denoiser_ensemble/data/mayo_2d_3_layer_mean/dose_rate_4.h5', 
    #                  '/home/dwu/trainData/deep_denoiser_ensemble/data/mayo_2d_3_layer_mean/dose_rate_2.h5'), 
    #                 ('/home/dwu/trainData/deep_denoiser_ensemble/data/mayo_2d_3_layer_mean/dose_rate_8.h5', 
    #                  '/home/dwu/trainData/deep_denoiser_ensemble/data/mayo_2d_3_layer_mean/dose_rate_2.h5'),]
    # dst_datasets = ['/home/dwu/trainData/deep_denoiser_ensemble/data/mayo_2d_3_layer_mean/dose_rate_1.h5']
    
    # generator = Image2DGenerator(manifest, src_datasets, dst_datasets, tags, patch_size = (64,64), num_patches_per_slice=100, num_slices_per_batch=1)
    # x, y = generator[0]

    src_datasets = [('/home/dwu/trainData/deep_denoiser_ensemble/data/mayo_2d_3_layer_mean/dose_rate_4.h5', 
                     '/home/dwu/trainData/uncertainty_prediction/denoising_results/mayo_2d_3_layer_mean/l2_depth_3/dose_rate_4/dose_rate_4.h5')]
    dst_datasets = [('/home/dwu/trainData/deep_denoiser_ensemble/data/mayo_2d_3_layer_mean/dose_rate_1.h5', 
                     '/home/dwu/trainData/uncertainty_prediction/denoising_results/mayo_2d_3_layer_mean/l2_depth_3/dose_rate_4/dose_rate_4.h5')]
    generator = Image2DGeneratorForUncertainty(manifest, src_datasets, dst_datasets, tags, 
                                               patch_size = (-1,-1), num_patches_per_slice=1, num_slices_per_batch=4)
    x,y = generator[0]
    # x, y = generator.load_slices(0, [0])

    