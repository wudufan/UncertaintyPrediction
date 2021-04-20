#!/usr/bin/env python
# coding: utf-8

# In[5]:


'''
Implementation of wgan_gp
'''


# In[6]:


import tensorflow as tf
import numpy as np


# In[7]:


class DiscriminatorResNet2D:
    def __init__(self, 
                 input_shape = [64, 64, 1], 
                 nconv_per_module = 2, 
                 features = [64, 128, 256], 
                 strides = [2, 2, 2],
                 fc_features = [1024], 
                 dropouts = [0.5],
                 lrelu = 0.2, 
                 layer_norm = True):
        
        self.input_shape = input_shape
        self.nconv_per_module = nconv_per_module
        self.features = features
        self.strides = strides
        self.fc_features = fc_features
        self.lrelu = 0.2
        self.dropouts = dropouts
        self.layer_norm = layer_norm
        
        assert(len(self.features) == len(self.strides))
    
    def build(self):
        inputs = tf.keras.Input(shape = self.input_shape, name = 'input')
        x = inputs
        
        # residual blocks with downsampling
        for i in range(len(self.features)):
            # bypass
            bypass = tf.keras.layers.Conv2D(self.features[i], 1, strides = self.strides[i], padding = 'same')(x)
            # main pass
            for k in range(self.nconv_per_module):
                if k < self.nconv_per_module - 1:
                    x = tf.keras.layers.Conv2D(self.features[i], 3, padding='same')(x)
                    if self.layer_norm:
                        x = tf.keras.layers.LayerNormalization([1,2,3], scale = False)(x)
                    x = tf.keras.layers.LeakyReLU(self.lrelu)(x)
                else:
                    x = tf.keras.layers.Conv2D(self.features[i], 3, strides = self.strides[i], padding='same')(x)
            # merge
            x = x + bypass
            if self.layer_norm:
                x = tf.keras.layers.LayerNormalization([1,2,3], scale = False)(x)
            x = tf.keras.layers.LeakyReLU(self.lrelu)(x)
        
        # fc blocks
        x = tf.keras.layers.Flatten()(x)
        for i in range(len(self.fc_features)):
            x = tf.keras.layers.Dense(self.fc_features[i])(x)
            x = tf.keras.layers.LeakyReLU(self.lrelu)(x)
            x = tf.keras.layers.Dropout(self.dropouts[i])(x)
            
        x = tf.keras.layers.Dense(1, use_bias=False)(x)
        
        self.model = tf.keras.Model(inputs = inputs, outputs = x)
        
        return self.model


# In[8]:


def discriminator_loss(real_logits, fake_logits):
    real_loss = tf.reduce_mean(real_logits)
    fake_loss = tf.reduce_mean(fake_logits)
    return fake_loss - real_loss

def generator_loss(fake_logits):
    return -tf.reduce_mean(fake_logits)


# In[9]:


class wgan_gp(tf.keras.Model):
    def __init__(self, generator, discriminator, l2_weight = 50, gp_weight = 10, discriminator_steps = 4):
        super(wgan_gp, self).__init__()
        self.generator = generator
        self.discriminator = discriminator
        self._set_inputs(generator.inputs)
        self.l2_weight = l2_weight
        self.gp_weight = gp_weight
        self.d_steps = discriminator_steps
    
    def compile(self, d_optimizer, g_optimizer, d_loss_fn = discriminator_loss, g_loss_fn = generator_loss):
        super(wgan_gp, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.d_loss_fn = d_loss_fn
        self.g_loss_fn = g_loss_fn
    
    def call(self, x):
        return self.generator(x)
    
    def gradient_penalty(self, fake_imgs, real_imgs):
        """ Calculates the gradient penalty.

        This loss is calculated on an interpolated image
        and added to the discriminator loss.
        """
        batch_size = tf.shape(real_imgs)[0]
        
        # Get the interpolated image
        alpha = tf.random.normal([batch_size, 1, 1, 1], 0.0, 1.0)
        diff = fake_imgs - real_imgs
        interpolated = real_imgs + alpha * diff

        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated)
            # 1. Get the discriminator output for this interpolated image.
            pred = self.discriminator(interpolated, training=True)

        # 2. Calculate the gradients w.r.t to this interpolated image.
        grads = gp_tape.gradient(pred, [interpolated])[0]
        # 3. Calculate the norm of the gradients.
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
        gp = tf.reduce_mean((norm - 1.0) ** 2)
        
        return gp
    
    def train_step(self, data):
        # For each batch, we are going to perform the
        # following steps as laid out in the original paper:
        # 1. Train the generator and get the generator loss
        # 2. Train the discriminator and get the discriminator loss
        # 3. Calculate the gradient penalty
        # 4. Multiply this gradient penalty with a constant weight factor
        # 5. Add the gradient penalty to the discriminator loss
        # 6. Return the generator and discriminator losses as a loss dictionary

        # Train the discriminator first. The original paper recommends training
        # the discriminator for `x` more steps (typically 5) as compared to
        # one step of the generator. Here we will train it for 4 steps.
        noisy_imgs = data[0]
        real_imgs = data[1]
        
        # the fake_imgs (denoised images) will not change for this batch
        fake_imgs = self.generator(noisy_imgs, training=False)
        
        for i in range(self.d_steps):            
            with tf.GradientTape() as tape:
                # Generate fake images from the latent vector
                # Get the logits for the fake images
                fake_logits = self.discriminator(fake_imgs, training=True)
                # Get the logits for the real images
                real_logits = self.discriminator(real_imgs, training=True)

                # Calculate the discriminator loss using the fake and real image logits
                d_cost = self.d_loss_fn(real_logits=real_logits, fake_logits=fake_logits)
                # Calculate the gradient penalty
                gp = self.gradient_penalty(fake_imgs, real_imgs)
                # Add the gradient penalty to the original discriminator loss
                d_loss = d_cost + gp * self.gp_weight

            # Get the gradients w.r.t the discriminator loss
            d_gradient = tape.gradient(d_loss, self.discriminator.trainable_variables)
            # Update the weights of the discriminator using the discriminator optimizer
            self.d_optimizer.apply_gradients(
                zip(d_gradient, self.discriminator.trainable_variables)
            )

        # Train the generator        
        with tf.GradientTape() as tape:
            # watch the gradient
            fake_imgs = self.generator(noisy_imgs, training=True)
            # Get the discriminator logits for fake images
            gen_img_logits = self.discriminator(fake_imgs, training=False)
            # Calculate the generator loss
            g_cost = self.g_loss_fn(gen_img_logits)
            # l2 cost
            l2_cost = tf.reduce_mean((fake_imgs - real_imgs)**2)
            # total generator loss
            g_loss = g_cost + l2_cost * self.l2_weight

#         print ('gen')
        # Get the gradients w.r.t the generator loss
        gen_gradient = tape.gradient(g_loss, self.generator.trainable_variables)
        # Update the weights of the generator using the generator optimizer
        self.g_optimizer.apply_gradients(
            zip(gen_gradient, self.generator.trainable_variables)
        )

        return {"d_loss": d_loss, "g_cost": g_cost, "l2_cost": l2_cost}


# In[10]:


if __name__ == '__main__':
    import unet
    import os
    import tensorflow.keras.backend as K
    
    
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    K.clear_session()
    
    gen = unet.unet2d(input_shape = [64,64,1])
    g_model = gen.build()
    
    discriminator = DiscriminatorResNet2D()
    d_model = discriminator.build()
    
    d_optimizer = tf.keras.optimizers.Adam()
    g_optimizer = tf.keras.optimizers.Adam()
    wgan = wgan_gp(g_model, d_model)
    wgan.compile(d_optimizer, g_optimizer)
    
    img = np.zeros([1,64,64,1], np.float32)
    loss = wgan.train_on_batch(img, img, return_dict=True)
    
#     wgan.save('t', save_format = 'tf')


# In[ ]:




