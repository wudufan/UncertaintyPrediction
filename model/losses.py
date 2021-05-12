'''
User defined losses for network training
'''

import tensorflow as tf

class AleatoricUncertaintyLoss(tf.keras.losses.Loss):
    def __init__(self):
        super().__init__()
    
    def __call__(self, y_true, y_pred, sample_weight=None):
        y = y_pred[..., :1]
        s = y_pred[..., 1:]

        return 0.5 * (tf.exp(-s) * (y_true - y) ** 2 + s)
