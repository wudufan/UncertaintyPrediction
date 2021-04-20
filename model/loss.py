'''
Losses
'''

import tensorflow as tf
import tensorflow.keras.backend as K

class DetectionLoss(tf.losses.Loss):
    '''
    Detection loss
    '''
    def __init__(self, classification_loss, reg = 1, delta = 1):
        super().__init__()
        self.classification_loss = classification_loss        
        self.reg_loss = tf.keras.losses.Huber(delta, tf.keras.losses.Reduction.NONE)
        self.reg = reg
    
    def __call__(self, y_true, y_pred, sample_weight=None):
        loss_prob = self.classification_loss(y_true[:, :1], y_pred[:, :1])
        
        loss_reg = self.reg_loss(y_true[:, 1:], y_pred[:, 1:])
        loss_reg = loss_reg * y_true[:, 0]

        return loss_prob + self.reg * loss_reg

def binary_focal_loss(gamma=2., alpha=.25):
    """
    Binary form of focal loss.
      FL(p_t) = -alpha * (1 - p_t)**gamma * log(p_t)
      where p = sigmoid(x), p_t = p or 1 - p depending on if the label is 1 or 0, respectively.
    References:
        https://arxiv.org/pdf/1708.02002.pdf
    Usage:
     model.compile(loss=[binary_focal_loss(alpha=.25, gamma=2)], metrics=["accuracy"], optimizer=adam)
    """

    def binary_focal_loss_fixed(y_true, y_pred):
        """
        :param y_true: A tensor of the same shape as `y_pred`
        :param y_pred:  A tensor resulting from a sigmoid
        :return: Output tensor.
        """
        y_true = tf.cast(y_true, tf.float32)
        # Define epsilon so that the back-propagation will not result in NaN for 0 divisor case
        epsilon = K.epsilon()
        # Add the epsilon to prediction value
        # y_pred = y_pred + epsilon
        # Clip the prediciton value
        y_pred = K.clip(y_pred, epsilon, 1.0 - epsilon)
        # Calculate p_t
        p_t = tf.where(K.equal(y_true, 1), y_pred, 1 - y_pred)
        # Calculate alpha_t
        alpha_factor = K.ones_like(y_true) * alpha
        alpha_t = tf.where(K.equal(y_true, 1), alpha_factor, 1 - alpha_factor)
        # Calculate cross entropy
        cross_entropy = -K.log(p_t)
        weight = alpha_t * K.pow((1 - p_t), gamma)
        # Calculate focal loss
        loss = weight * cross_entropy
        # Sum the losses in mini_batch
        loss = K.mean(K.sum(loss, axis=1))
        return loss

    return binary_focal_loss_fixed

def cross_entropy_with_hard_negative_mining(y_true, y_pred, nhard):
    pos_inds = tf.where(y_true == 1)
    neg_inds = tf.where(y_true == 0)
    
    def get_pos_loss():
        y_true_pos = tf.gather_nd(y_true, pos_inds)[:, tf.newaxis]
        y_pred_pos = tf.gather_nd(y_pred, pos_inds)[:, tf.newaxis]
        pos_losses = tf.keras.losses.binary_crossentropy(y_true_pos, y_pred_pos)

        return pos_losses

    def get_hard_neg_loss():
        y_true_neg = tf.gather_nd(y_true, neg_inds)[:, tf.newaxis]
        y_pred_neg = tf.gather_nd(y_pred, neg_inds)[:, tf.newaxis]
        neg_losses = tf.keras.losses.binary_crossentropy(y_true_neg, y_pred_neg)
        hard_neg_losses, _ = tf.math.top_k(neg_losses, tf.math.minimum(nhard, tf.shape(neg_losses)[0]), False)

        return hard_neg_losses

    def get_zeros():
        return tf.zeros([1,1])

    # positive samples
    pos_losses = tf.cond(tf.shape(pos_inds)[0] > 0, get_pos_loss, get_zeros)

    # negative samples
    hard_neg_losses = tf.cond(tf.shape(neg_inds)[0] > 0, get_hard_neg_loss, get_zeros)

    return 0.5 * tf.reduce_mean(hard_neg_losses) + 0.5 * tf.reduce_mean(pos_losses)

if __name__ == '__main__':
    import numpy as np

    loss = DetectionLoss(tf.keras.losses.binary_crossentropy)

    y_true = np.array([[1, 0.1, 0, 0, 5], 
                       [0, 0, 0, 0, 0]], np.float32)
    y_pred = np.array([[1, 0.1, 0, 0, 7], 
                       [0, 0, 0, 0, 1]], np.float32)

    print (loss(y_true, y_pred))
