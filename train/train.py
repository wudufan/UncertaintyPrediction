'''
Training code
'''

import sys
import os
import tensorflow as tf
import tensorflow.keras.backend as K
import shutil

import argparse
import configparser

sys.path.append('..')
import model.model3d
import model.loss
import model.data
import utils.config_manager

parser = argparse.ArgumentParser()
parser.add_argument('config')

is_debug = False

# for debug
if is_debug:
    cmds = ['config/MobileNet/debug.cfg', 
            '--Loss.nhard', '4']
    verbose = 1
else:
    cmds = None
    verbose = 2

# first get the configuration file
args, _ = parser.parse_known_args(cmds)
config = configparser.ConfigParser()
config._interpolation = configparser.ExtendedInterpolation()
config.read(args.config)

# then modify the configuration with any additional arugments
parser = utils.config_manager.build_argparser_from_config(parser, config)
args = parser.parse_args(cmds)
config = utils.config_manager.update_config_from_args(config, args)

# write the config file to the output directory
output_dir = os.path.join(config['IO']['output_dir'], config['IO']['tag'])
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
with open(os.path.join(output_dir, 'config.cfg'), 'w') as f:
    config.write(f)

# parse arguments
train_args = utils.config_manager.get_kwargs(config)

if train_args['IO']['log_dir'] is None:
    train_args['IO']['log_dir'] = os.path.join(output_dir, 'log')
if train_args['IO']['manifest'] is None:
    train_args['IO']['manifest'] = os.path.join(train_args['IO']['data_dir'], 'manifest.csv')

# loss function names
if not 'name' in train_args['Loss']:
    train_args['Loss']['name'] = None

# output the config
print (args.config)
for sec in train_args:
    print ('[%s]'%sec)
    for k in train_args[sec]:
        print (k, '=', train_args[sec][k])
    print ('', flush=True)

os.environ['CUDA_VISIBLE_DEVICES'] = '%d'%train_args['Train']['device']
K.clear_session()

# tensorboard
log_dir = train_args['IO']['log_dir']
if train_args['Train']['relog'] and os.path.exists(log_dir):
    shutil.rmtree(log_dir)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir = log_dir, histogram_freq=1)

# loss function
if train_args['Loss']['name'] == 'focal':
    print ('Using focal loss with gamma = %g, alpha = %g'%(train_args['Loss']['gamma'], train_args['Loss']['alpha']))
    loss_fn = model.loss.binary_focal_loss(gamma = train_args['Loss']['gamma'], alpha = train_args['Loss']['alpha'])
elif train_args['Loss']['name'] is None:
    if train_args['Loss']['nhard'] is None:
        print ('Using binary cross entropy')
        def loss_fn(y_pred, y_true):
            return tf.keras.losses.binary_crossentropy(y_pred, y_true)
    else:
        print ('Using OHEM with nhard = %g'%train_args['Loss']['nhard'])
        def loss_fn(y_pred, y_true):
            return model.loss.cross_entropy_with_hard_negative_mining(y_pred, y_true, nhard = train_args['Loss']['nhard'])
else:
    raise ValueError('train_args["Loss"]["name"] is not supported. Got %s'%train_args['Loss']['name'])

# network
model_class = getattr(sys.modules['model.model3d'], train_args['Model']['name'])
print (model_class)
network = model_class(**train_args['Network'])
_ = network.build()
network.model.summary(line_length = 120)

# learning rate schedule
def scheduler(epoch, lr):
    epoch_list = train_args['Train']['epochs']
    lr_list = train_args['Train']['lr']
    for i in range(len(epoch_list)):
        if epoch < epoch_list[i]:
            return lr_list[i]
    
    return lr_list[-1]

lr_callback = tf.keras.callbacks.LearningRateScheduler(scheduler, verbose = 1)

optimizer = tf.keras.optimizers.Adam()
network.model.compile(optimizer, 
                      loss = loss_fn, 
                      metrics = [tf.keras.metrics.BinaryAccuracy(), 
                                 tf.keras.metrics.SpecificityAtSensitivity(0.95)])

# data part
generator = model.data.DataGenerator(train_args['IO']['manifest'], train_args['IO']['data_dir'], 'train', **train_args['Data'])
valid_generator = model.data.DataGenerator(train_args['IO']['manifest'], train_args['IO']['data_dir'], 'valid', 
                                           shape = train_args['Data']['shape'], 
                                           batch_size = train_args['Data']['batch_size'],
                                           shuffle = False, preload = True, 
                                           zoom = 0, offset = (0,0,0), flip = False, noise_prob = 0)

# model saver
best_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    os.path.join(output_dir, 'best.h5'), monitor = 'val_specificity_at_sensitivity', mode = 'max',save_best_only=True, verbose = 1)
regular_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    os.path.join(output_dir, '{epoch}.h5'), save_freq=len(generator) * train_args['Train']['save_freq'], verbose = 1)

network.model.fit(generator, validation_data = valid_generator, epochs = train_args['Train']['epochs'][-1], shuffle = False, 
                  max_queue_size = 10, workers = 4, use_multiprocessing = False, verbose = verbose,
                  callbacks = [tensorboard_callback, lr_callback, best_checkpoint_callback, regular_checkpoint_callback])
