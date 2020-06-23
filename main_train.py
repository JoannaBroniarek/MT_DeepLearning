import os

import tensorflow as tf

#tf.config.gpu.set_per_process_memory_fraction(0.8)
#tf.config.gpu.set_per_process_memory_growth(True)

#run_opts = tf.compat.v1.RunOptions(report_tensor_allocations_upon_oom = True


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
  try:
      tf.config.experimental.set_visible_devices(gpus[1], 'GPU')
#    tf.config.experimental.set_virtual_device_configuration(
#        gpus[0],
#        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])
#    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
#    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Virtual devices must be set before GPUs have been initialized
    print(e)


import time
import argparse

from tensorflow.keras.utils import Progbar

from models import *
from create_data import *


def parse_args():
    """ Parsing and configuration """
    
    desc = "Tensorflow 2.2 implementation of UNet (Encoder Part) for 3D DWI-MRI"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--epochs', type=int, default=3, help='The number of epochs to run')
    parser.add_argument('--batch_size', type=int, default=8, help='The size of batch')
    parser.add_argument('--lr', type=float, default=5e-7, help='Learning rate')
    parser.add_argument('--train_path', type=str, default="/myworkdir/data/small_train.tfrecords", help='path to train data')
    parser.add_argument('--val_path', type=str, default="/myworkdir/data/small_val.tfrecords", help='path to val data')
    parser.add_argument('--rand_seed', type=int, default=45, help='tf random seed')
    parser.add_argument('--restore', type=str, default=None, help='path restore model path')

    return parser.parse_args()
    
    
if __name__ == "__main__":
    args = parse_args()
    if args is None:
        exit()

    
    #########################
    ##     Load Dataset    ##
    #########################
    BUFFER_SIZE = 1
    
    tf.random.set_seed(args.rand_seed)
    train_filename = args.train_path
    train_dataset = parse_dataset(train_filename)
    train_dataset = train_dataset.repeat(args.epochs).batch(args.batch_size)


    val_dataset = args.val_path
    val_dataset = parse_dataset(val_dataset)
    val_dataset = val_dataset.batch(args.batch_size)
    
    #########################
    ##     Set Params      ##
    #########################
    
    lr = args.lr
    EPOCHS = args.epochs
    
    #TODO: add restore 
    
    #########################
    ##        Train        ##
    #########################
    tf.keras.backend.clear_session()

    # Create a MirroredStrategy.
    #strategy = tf.distribute.MirroredStrategy()
    #print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

    #with strategy.scope():

    encoder = encoder3D(kernel_size = (4, 4, 4),
                        pool_size = (3, 3, 3),
                        weight_initializer = None)
    
    #encoder.summary()
    
    encoder.compile(optimizer=tf.keras.optimizers.Adam(lr, clipvalue=0.5),
                    loss={"dense":tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                        "dense_1":tf.keras.losses.BinaryCrossentropy(from_logits=False)},
                        #"dense_2":tf.keras.losses.MeanSquaredError()},
                         loss_weights=[0.1, 1], metrics=['accuracy'])
    
    encoder.fit(train_dataset, 
                epochs=args.epochs, 
                verbose=1, 
                steps_per_epoch=6,#512 
                validation_data=val_dataset, validation_steps=3)
