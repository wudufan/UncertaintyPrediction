'''
custom metrics
'''

import tensorflow as tf

class DetectionSpecificityAtSensitivity(tf.keras.metrics.SpecificityAtSensitivity):
    def __init__(self, sensitivity, num_thresholds = 200, name = None, dtype = None):
        super().__init__(sensitivity, num_thresholds=num_thresholds, name = name, dtype = dtype)
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        return super().update_state(y_true[:, 0], y_pred[:, 0], sample_weight)

class DetectionBinaryCrossentropy(tf.keras.metrics.BinaryCrossentropy):
    def __init__(self, name = 'detection_binary_crossentropy', dtype = None, from_logits = False, label_smoothing = 0):
        super().__init__(name = name, dtype = dtype, from_logits=from_logits, label_smoothing=label_smoothing)
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        return super().update_state(y_true[:, 0], y_pred[:, 0], sample_weight)
