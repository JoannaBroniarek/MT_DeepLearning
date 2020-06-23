import numpy as np
import os
import tensorflow as tf

from tensorflow.keras import initializers

from tensorflow.keras.layers import (Conv3D,
                                     Dense, 
                                     Input,
                                     Dropout,
                                     MaxPooling3D,
                                     Flatten, 
                                     Activation)

from tensorflow.keras import Model
#import sys


######################################
####        MultiTaskModel        ####
######################################


    
def encoder3D(kernel_size, pool_size, weight_initializer):
    """
    Encoder3D.
    # TODO : add description
    
    Parameters
    ----------
    kernel_size : {an integer or tuple/list of 3 integers} - 
            Specifies the depth, height and width of the 3D convolution window. 
            Can be a single integer to specify the same value for all spatial dimensions.. 
            Default is (4, 4, 4).
    pool_size : {an integer or tuple/list of 3 integers} - 
            Specifies the depth, height and width of the 3D max-pooling window.
            Can be a single integer to specify the same value for all spatial dimensions.
            Default is (2, 2, 2).
    weight_initializer  - Kernel initializer for a dense layer.

    """
    rate = 0.2
    input_shape = (88, 256, 256, 2)

    inputs = Input(input_shape)
    conv11 = Conv3D(8, kernel_size, padding='same')(inputs)
    drop1 = Dropout(rate=rate)(conv11)
    pool1 = MaxPooling3D(pool_size=pool_size)(drop1)
    activ1 = Activation('relu')(pool1)
#    bn1 = tf.keras.layers.BatchNormalization()(activ1) 

    conv21 = Conv3D(16, kernel_size, padding='same')(activ1)
    drop2 = Dropout(rate=rate)(conv21)
    pool2 = MaxPooling3D(pool_size=pool_size)(drop2)
   # bn2 = tf.keras.layers.BatchNormalization(pool2)
    activ2 = Activation('relu')(pool2)
    #bn2 = tf.keras.layers.BatchNormalization()(activ2)
 
    conv31 = Conv3D(32, kernel_size, padding='same')(activ2)
    drop3 = Dropout(rate=rate)(conv31)
    pool3 = MaxPooling3D(pool_size=pool_size)(drop3)
	
    flatt = Flatten()(pool3)
    age_output = Dense(12, activation='softmax')(flatt)  #activation='softmax')
    gender_output = Dense(1, activation='softmax')(flatt)
#    direction_output  = Dense(2)(flatt)
    
    model = Model(inputs=[inputs], outputs=[age_output, gender_output]) #, direction_output])

    return model


