from __future__ import print_function, division

from keras.datasets import mnist
import keras
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import  Conv2D
from keras.layers import UpSampling2D
from keras.models import Sequential, Model
from keras.optimizers import RMSprop
from keras.applications import ResNet50
from keras.engine import base_layer
from keras import layers
from keras.applications import ResNet50
import keras.backend as K
import tensorflow as tf

import matplotlib.pyplot as plt
import sys
import os
import numpy as np



def spectral_norm(w, r=5):
    w_shape = K.int_shape(w)
    in_dim = np.prod(w_shape[:-1]).astype(int)
    out_dim = w_shape[-1]
    w = K.reshape(w, (in_dim, out_dim))
    u = K.ones((1, in_dim))
    for i in range(r):
        v = K.l2_normalize(K.dot(u, w))
        u = K.l2_normalize(K.dot(v, K.transpose(w)))
    return K.sum(K.dot(K.dot(u, w), K.transpose(v)))


def spectral_normalization(w):
    return w / spectral_norm(w)

def gen_block(input_tensor, filters, scale, stage, using_sn=False, kernel_size=(3,3), strides=(1,1)):
    if using_sn:
        fn_normal=spectral_normalization
    else:
        fn_normal=None
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    filters1, filters2 = filters

    conv_name_base = 'res' + str(stage) + '_'
    bn_name_base = 'bn' + str(stage) + '_'
    if scale =="up":
        shortcut = input_tensor
        shortcut = UpSampling2D()(shortcut)
        shortcut = Conv2D(filters2, (1, 1), strides=strides,
                    name=conv_name_base + '1')(shortcut)
    else:
        shortcut = input_tensor
        shortcut = Conv2D(filters2, (1, 1), strides=strides,padding="same",
                    name=conv_name_base + '1')(shortcut)
    
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + 'a')(input_tensor)
    x = Activation('relu')(x) 
    if scale =="up":
        x = UpSampling2D()(x)

    x = Conv2D(filters1, kernel_size, strides=strides,padding="same",
               name=conv_name_base + 'a')(x)
    
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + 'b')(x)
    x = Activation('relu')(x)
    x = Conv2D(filters2, (1, 1), strides=strides,padding="same",
               name=conv_name_base + 'b')(x)
    x = layers.add([x, shortcut])
    return x

def critic_block(input_tensor, filters, scale, stage, using_sn=False, kernel_size=(3,3), strides=(2,2)):
    if using_sn:
        fn_normal=spectral_normalization
    else:
        fn_normal=None
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    filters1, filters2 = filters
    conv_name_base = 'critic_res' + str(stage) + '_'
    bn_name_base = 'critic_bn' + str(stage) + '_'
    if scale =="down":
        shortcut = input_tensor
        shortcut = Conv2D(filters2, (1, 1), strides=strides,
                    name=conv_name_base + '1',kernel_constraint=spectral_normalization)(shortcut)
    else:
        shortcut = input_tensor
        shortcut = Conv2D(filters2, (1, 1), strides=(1,1),padding="same",
                    name=conv_name_base + '1')(shortcut)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + 'a',gamma_constraint=spectral_normalization)(input_tensor)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dropout(0.25)(x)
    if scale =="down":
        x = Conv2D(filters1, kernel_size, strides=strides,padding="same",
               name=conv_name_base + 'a',kernel_constraint=spectral_normalization)(x)
    else:
        x = Conv2D(filters1, kernel_size, strides=(1,1),padding="same",
               name=conv_name_base + 'a',kernel_constraint=spectral_normalization)(x)

    x = BatchNormalization(axis=bn_axis, name=bn_name_base + 'b',gamma_constraint=spectral_normalization)(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dropout(0.25)(x)
    x = Conv2D(filters2, (1, 1), strides=(1,1),padding="same",
               name=conv_name_base + 'b',kernel_constraint=spectral_normalization)(x)
    x = layers.add([x, shortcut])
    return x




    
