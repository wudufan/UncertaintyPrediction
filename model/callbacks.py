'''
Callback functions
'''

import tensorflow as tf
import numpy as np
import SimpleITK as sitk
import os

class SaveValid2DImageCallback(tf.keras.callbacks.Callback):
    def __init__(self, 
                 model: tf.keras.Model, 
                 x: dict, 
                 y: dict, 
                 output_dir: str, 
                 interval = 1, 
                 postprocess = None, 
                 norm_x = 1000,
                 norm_y = 1000
                 ):
        super().__init__()

        self.model = model
        self.x = x
        self.y = y
        self.output_dir = output_dir
        self.interval = interval
        self.postprocess = postprocess
        self.norm_x = norm_x
        self.norm_y = norm_y

    def on_epoch_end(self, epoch, logs = None):
        if (epoch + 1) % self.interval != 0:
            return
        
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        # predict all the x
        for name in self.x:
            preds = self.model.predict(self.x[name])
            if self.postprocess is not None:
                ys = self.postprocess(self.y[name])
                preds = self.postprocess(preds)

            xs = (self.x[name] * self.norm_x).astype(np.int16)
            ys = (ys * self.norm_y).astype(np.int16)
            preds = (preds * self.norm_y).astype(np.int16)

            sitk.WriteImage(sitk.GetImageFromArray(xs[...,0]), os.path.join(self.output_dir, name+'.x.nii'))
            sitk.WriteImage(sitk.GetImageFromArray(ys[...,0]), os.path.join(self.output_dir, name+'.y.nii'))
            sitk.WriteImage(sitk.GetImageFromArray(preds[...,0]), os.path.join(self.output_dir, name+'.pred.nii'))

class SaveValid2DCallback(tf.keras.callbacks.Callback):
    def __init__(self, 
                 model: tf.keras.Model, 
                 generator,
                 output_dir: str, 
                 interval = 1, 
                 postprocess = None, 
                 norm_x = 1000,
                 norm_y = 1000
                 ):
        super().__init__()

        self.model = model
        self.generator = generator
        self.output_dir = output_dir
        self.interval = interval
        self.postprocess = postprocess
        self.norm_x = norm_x
        self.norm_y = norm_y

    def on_epoch_end(self, epoch, logs = None):
        if (epoch + 1) % self.interval != 0:
            return
        
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        for idataset in range(len(self.generator.src_datasets)):
            xs = []
            ys = []
            preds = []
            nbatches = self.generator.num_batches_per_dataset()
            
            for i in range(idataset*nbatches, (idataset+1)*nbatches):
                x, y = self.generator[i]
                pred = self.model.predict(x)
                xs.append(x)
                ys.append(y)
                preds.append(pred)
            
            xs = np.concatenate(xs, 0)
            ys = np.concatenate(ys, 0)
            preds = np.concatenate(preds, 0)

            if self.postprocess is not None:
                ys = self.postprocess(ys)
                preds = self.postprocess(preds)
            
            xs = (xs * self.norm_x).astype(np.int16)
            ys = (ys * self.norm_y).astype(np.int16)
            preds = (preds * self.norm_y).astype(np.int16)

            name = os.path.basename(self.generator.src_datasets[idataset][0])[:-3]
            sitk.WriteImage(sitk.GetImageFromArray(xs[...,0]), os.path.join(self.output_dir, name+'.x.nii'))
            sitk.WriteImage(sitk.GetImageFromArray(ys[...,0]), os.path.join(self.output_dir, name+'.y.nii'))
            sitk.WriteImage(sitk.GetImageFromArray(preds[...,0]), os.path.join(self.output_dir, name+'.pred.nii'))


class TensorboardSnapshotCallback(tf.keras.callbacks.Callback):
    def __init__(self, 
                 model: tf.keras.Model, 
                 file_writer, 
                 x: dict, 
                 y: dict, 
                 ref: dict = None,
                 interval = 1,
                 postprocess = None, 
                 norm_x = 1000, 
                 vmin_x = -160,
                 vmax_x = 240, 
                 norm_y = 1000, 
                 vmin_y = -160, 
                 vmax_y = 240):
        super().__init__()

        self.model = model
        self.file_writer = file_writer
        self.x = x
        self.y = y
        self.ref = ref
        self.interval = interval
        self.norm_x = norm_x
        self.vmin_x = vmin_x
        self.vmax_x = vmax_x
        self.norm_y = norm_y
        self.vmin_y = vmin_y
        self.vmax_y = vmax_y

        # postprocessing on the prediction
        self.postprocess = postprocess
    
    def make_snapshot(self, img, norm, vmin, vmax):
        img = (img * norm - vmin) / (vmax - vmin)

        return img[..., [0]]

    def on_epoch_end(self, epoch, logs = None):
        if (epoch + 1) % self.interval != 0:
            return
        
        # predict all the x
        if self.ref is not None:
            preds = {k: self.model.predict(self.x[k]) - self.ref[k] for k in self.x}
        else:
            preds = {k: self.model.predict(self.x[k]) for k in self.x}

        if self.postprocess is not None:
            preds = {k: self.postprocess(preds[k]) for k in preds}
        
        with self.file_writer.as_default():
            for k in self.x:
                tf.summary.image(k + '/x', self.make_snapshot(self.x[k], self.norm_x, self.vmin_x, self.vmax_x), step = epoch)
                tf.summary.image(k + '/pred', self.make_snapshot(preds[k], self.norm_y, self.vmin_y, self.vmax_y), step = epoch)

                if self.y is not None and k in self.y:
                    tf.summary.image(k + '/y', self.make_snapshot(self.y[k], self.norm_y, self.vmin_y, self.vmax_y), step = epoch)


        