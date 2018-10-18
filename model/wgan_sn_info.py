
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

import os
import numpy as np
from tools.base_block import spectral_normalization,gen_block,critic_block


class ResGAN_info():
    def __init__(self,img_rows,img_cols,channels,latent_dim,num_classes):
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.channels = channels
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = latent_dim
        self.num_classes=num_classes
    def wasserstein_loss(self, y_true, y_pred):
        return K.mean(y_true * y_pred)
    def mutual_info_loss(self, c, c_given_x):
        """The mutual information metric we aim to minimize"""
        eps = 1e-8
        conditional_entropy = K.mean(- K.sum(K.log(c_given_x + eps) * c, axis=1))
        entropy = K.mean(- K.sum(K.log(c + eps) * c, axis=1))

    def build_generator(self):
        input_noise=Input(shape=(self.latent_dim,))
        x = Dense(7*7*self.latent_dim*64,activation="relu",name="fc_noise")(input_noise)
        x = Reshape((7,7,self.latent_dim*64),name="fc_reshaped")(x)
        x = BatchNormalization(momentum=0.9)(x)
        x = Activation('relu')(x)
        #unpooling
        magic = [(256,512), (256, 256), (128, 256), (128, 128), (128, 64)]
        scales = ['up','none','up','none','none']
        for block_idx in range(5):
            x =gen_block(x,filters=magic[block_idx],scale=scales[block_idx],stage=block_idx+1,using_sn=False )
        x = BatchNormalization(name="bn_last")(x)
        x = Activation('relu')(x)
        output = Conv2D(self.channels, (1, 1), strides=(1,1),padding="same",activation="tanh")(x)
        model = Model(input_noise,output)
        model.summary()
        return model
    def build_critic(self):

        img = Input(shape=self.img_shape)
        x = img
        magic = [(64, 128),(128,128),(128,256),(256,256),(512,256)]
        scales = ['down','none','down','none','none']
        for block_idx in range(5):
            x =critic_block(x,filters=magic[block_idx],scale=scales[block_idx],stage=block_idx+1,using_sn=True)

        x = BatchNormalization(axis=3,name="bn_last",gamma_constraint=spectral_normalization)(x)
        x = Activation('relu')(x)
        x = Flatten()(x)
        pre_model = Model(img,x)
        img_embedding= pre_model(img)
        q_net = Dense(128, activation='relu',kernel_constraint=spectral_normalization)(img_embedding)
        out = Dense(1,kernel_constraint=spectral_normalization)(img_embedding)
        label = Dense(self.num_classes, activation='softmax')(q_net)
        model_class = Model(img,label)
        model_d = Model(img,out)
        model_class.summary()
        model_d.summary()
        return model_d,model_class
