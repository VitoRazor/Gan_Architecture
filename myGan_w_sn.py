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


import keras.backend as K
import tensorflow as tf

import matplotlib.pyplot as plt

from data_loader import DataLoader
from model.wgan_sn import ResGAN
import sys
import os
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
path = "wgan_sn_meizi"
if not os.path.exists(path):
    os.mkdir(path)

ouput_images_path=path+"/images"
if not os.path.exists(ouput_images_path):
    os.mkdir(ouput_images_path)
saved_model_path=path+"/saved_model"
if not os.path.exists(saved_model_path):
    os.mkdir(saved_model_path)
saved_log=path+"/saved_log"
if not os.path.exists(saved_log):
    os.mkdir(saved_log)

class MyGAN():
    def __init__(self):
        self.img_rows = 64
        self.img_cols = 64
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = 128
        self.dataset_name = 'meizi'
        self.d_loss=[]
        self.g_loss=[]
        self.n_critic = 2
        self.n_gen = 2
        resgan=ResGAN(self.img_rows,self.img_cols,self.channels,self.latent_dim)
        self.data_loader = DataLoader(dataset_name=self.dataset_name,
                                      img_res= self.img_shape)

        # Following parameter and optimizer set as recommended in paper

        self.clip_value = 0.01
        optimizer = RMSprop(lr=0.00005)

        # Build and compile the critic
        self.critic = resgan.build_critic()
        self.generator = resgan.build_generator()
        # load model weights
        if os.path.exists(saved_model_path+"/discriminator_weights.hdf5"):
            self.critic.load_weights(saved_model_path+"/discriminator_weights.hdf5")
            print("load generator weights")
        if os.path.exists(saved_model_path+"/generator_weights.hdf5"):
            self.generator.load_weights(saved_model_path+"/generator_weights.hdf5")
            print("load discriminator weights")

        
        # Noise input
        z_disc = Input(shape=(self.latent_dim,))
        # Generate image based of noise (fake sample and real sample)
        fake_img = self.generator(z_disc)
        real_img = Input(shape=self.img_shape)

        # Discriminator determines validity of the real and fake images
        fake = self.critic(fake_img)
        valid = self.critic(real_img)
        self.critic.trainable = True
        self.generator.trainable = False
        self.discriminator = Model(inputs=[real_img, z_disc],
                    outputs=[valid, fake])
        self.discriminator.compile(loss=[resgan.wasserstein_loss,resgan.wasserstein_loss],
            optimizer=optimizer,
            loss_weights=[-1, -1]
            )
        
        #keras.utils.plot_model(self.critic, to_file=saved_model_path+'/model_1.png', show_shapes=True, show_layer_names=True)
        # Build the generator
        
        # The generator takes noise as input and generated imgs
        z = Input(shape=(self.latent_dim,))
        img = self.generator(z)
        # The critic takes generated images as input and determines validity
        valid = self.critic(img)
        # The combined model  (stacked generator and critic)
        self.combined = Model(z, valid)
        self.critic.trainable = False
        self.generator.trainable = True
        self.combined.compile(loss=resgan.wasserstein_loss,
            optimizer=optimizer)
        #keras.utils.plot_model(self.combined, to_file=saved_model_path+'/model_2.png', show_shapes=True, show_layer_names=True)
    def save_imgs(self, epoch):
        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, self.latent_dim))
        gen_imgs = self.generator.predict(noise)

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5
        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                if self.channels>1:
                    axs[i,j].imshow(gen_imgs[cnt])
                else:
                    axs[i,j].imshow(gen_imgs[cnt,:,:,0],"gray")
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig(ouput_images_path+"/example_%d.png" % epoch)
        plt.close()
    def save_model(self):
        def save(model, model_name):
            model_path = saved_model_path +"/%s.json" % model_name
            weights_path = saved_model_path +"/%s_weights.hdf5" % model_name
            options = {"file_arch": model_path,
                        "file_weight": weights_path}
            json_string = model.to_json()
            open(options['file_arch'], 'w').write(json_string)
            model.save_weights(options['file_weight'])

        save(self.generator, "generator")
        save(self.critic, "discriminator")
    def save_log(self):
        d_loss=np.array(self.d_loss)
        np.save(saved_log+"/d_loss.npy",d_loss)
        # d_acc=np.array(self.acc)
        # np.save(saved_log+"/d_acc.npy",d_acc)
        g_loss=np.array(self.g_loss)
        np.save(saved_log+"/g_loss.npy",g_loss)
    def plot_log(self):
        r = 2 
        fig, axs = plt.subplots(r)
        axs[0].plot(self.d_loss)
        axs[1].plot(self.g_loss)
        fig.savefig(saved_log+"/log.png" )
        plt.close()
    def train(self, epochs, batch_size=128, save_interval=50):
        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = -np.ones((batch_size, 1))
        gene=self.data_loader.load_batch(batch_size)
        for epoch in range(epochs):
            # ---------------------
            #  Train Discriminator
            # ---------------------
            for _ in range(self.n_critic):
            # Select a random half of images
                imgs=next(gene)
                noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

                # Train the discriminator (real classified as ones and generated as zeros)
                d_loss = self.discriminator.train_on_batch([imgs, noise],
                                                    [valid, fake])
            for _ in range(self.n_gen):
                # ---------------------
                #  Train Generator
                # ---------------------
                # Train the generator (wants discriminator to mistake images as real)
                g_loss = self.combined.train_on_batch(noise, fake)
            # Plot the progress
            print ("%d [D loss: %f] [G loss: %f]" % (epoch, d_loss[0],g_loss))
            self.d_loss.append(d_loss[0])
            self.g_loss.append(g_loss)

            
            #print(d_loss,g_loss)
            # If at save interval => save generated image samples
            if epoch % save_interval == 0:
                self.save_imgs(epoch)
            if epoch % (save_interval*10) == 0:
                self.save_model()
                self.save_log()
                self.plot_log()
if __name__ == '__main__':
    mygan = MyGAN()
    mygan.train(epochs=500000, batch_size=16, save_interval=100)


    
