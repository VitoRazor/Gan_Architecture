from __future__ import print_function, division

from keras.datasets import mnist
import keras
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import  Conv2D
from keras.layers import UpSampling2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.applications import ResNet50
from keras.engine import base_layer
from keras import layers
from keras.utils import to_categorical


import keras.backend as K
import tensorflow as tf

import matplotlib.pyplot as plt

from data_loader import DataLoader
from model.wgan_sn_info import ResGAN_info
import sys
import os
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
path = "wgan_sn_info"
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
        self.img_rows = 28
        self.img_cols = 28
        self.channels = 1
        self.latent_dim = 72
        self.classes = 10
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.dataset_name = 'meizi'
        self.d_loss=[]
        self.g_loss=[]
        self.c_loss=[]
        self.n_critic = 2
        self.n_gen = 2
        resgan=ResGAN_info(self.img_rows,self.img_cols,self.channels,self.latent_dim, self.classes)
        #self.data_loader = DataLoader(dataset_name=self.dataset_name,
        #                              img_res= self.img_shape)

        # Following parameter and optimizer set as recommended in paper
        optimizer = Adam(0.0002, 0.5)

        # Build and compile the critic
        self.critic, self.classify = resgan.build_critic()
        self.generator = resgan.build_generator()
        # load model weights
        if os.path.exists(saved_model_path+"/discriminator_weights.h5"):
            self.critic.load_weights(saved_model_path+"/discriminator_weights.h5")
            print("load generator weights")
        if os.path.exists(saved_model_path+"/generator_weights.h5"):
            self.generator.load_weights(saved_model_path+"/generator_weights.h5")
            print("load discriminator weights")
        if os.path.exists(saved_model_path+"/classify_weights.h5"):
            self.classify.load_weights(saved_model_path+"/classify_weights.h5")
            print("load classify weights")

        
        # Noise input
        z_disc = Input(shape=(self.latent_dim,))
        # Generate image based of noise (fake sample and real sample)
        fake_img = self.generator(z_disc)
        real_img = Input(shape=self.img_shape)

        # Discriminator determines validity of the real and fake images
        fake = self.critic(fake_img)
        valid = self.critic(real_img)
        real_label=self.classify(real_img)

        self.critic.trainable = True
        self.classify.trainable = True
        self.generator.trainable = False
        self.discriminator = Model(inputs=[real_img, z_disc],
                    outputs=[valid, fake,real_label])
        self.discriminator.compile(loss=[resgan.wasserstein_loss,resgan.wasserstein_loss,'categorical_crossentropy'],
            optimizer=optimizer,
            loss_weights=[-1, -1, 1]
            )


        
        #keras.utils.plot_model(self.critic, to_file=saved_model_path+'/model_1.png', show_shapes=True, show_layer_names=True)
        # Build the generator
        
        # The generator takes noise as input and generated imgs
        z = Input(shape=(self.latent_dim,))
        img = self.generator(z)      
        # The critic takes generated images as input and determines validity
        valid = self.critic(img)
        label = self.classify(img)
        # The combined model  (stacked generator and critic)
        self.combined = Model(z, [valid,label])
        self.critic.trainable = False
        self.classify.trainable = False
        self.generator.trainable = True
        self.combined.compile(loss=[resgan.wasserstein_loss,"categorical_crossentropy"],
            optimizer=optimizer,
            loss_weights=[1, 1])
        #keras.utils.plot_model(self.combined, to_file=saved_model_path+'/model_2.png', show_shapes=True, show_layer_names=True)
    def save_imgs(self, epoch):
        r, c = 10, 10
        fig, axs = plt.subplots(r, c)
        for i in range(c):
            sampled_noise, _ = self.sample_generator_input(c)
            label = to_categorical(np.full(fill_value=i, shape=(r,1)), num_classes=self.classes)
            gen_input = np.concatenate((sampled_noise, label), axis=1)
            gen_imgs = self.generator.predict(gen_input)
            gen_imgs = 0.5 * gen_imgs + 0.5
            for j in range(r):
                if self.channels>1:
                    axs[i,j].imshow(gen_imgs[j])
                else:
                    axs[i,j].imshow(gen_imgs[j,:,:,0],"gray")
                axs[i,j].axis('off')
        fig.savefig(ouput_images_path+"/example_%d.png" % epoch)
        plt.close()
    def save_model(self):
        def save(model, model_name):
            model_path = saved_model_path +"/%s.json" % model_name
            weights_path = saved_model_path +"/%s_weights.h5" % model_name
            options = {"file_arch": model_path,
                        "file_weight": weights_path}
            json_string = model.to_json()
            open(options['file_arch'], 'w').write(json_string)
            model.save_weights(options['file_weight'])

        save(self.generator, "generator")
        save(self.critic, "discriminator")
        save(self.classify, "classify")
    def save_log(self):
        d_loss=np.array(self.d_loss)
        np.save(saved_log+"/d_loss.npy",d_loss)
        # d_acc=np.array(self.acc)
        # np.save(saved_log+"/d_acc.npy",d_acc)
        g_loss=np.array(self.g_loss)
        np.save(saved_log+"/g_loss.npy",g_loss)
        c_loss=np.array(self.c_loss)
        np.save(saved_log+"/c_loss.npy",c_loss)
    def plot_log(self):
        r = 3 
        fig, axs = plt.subplots(r)
        axs[0].plot(self.d_loss)
        axs[1].plot(self.g_loss)
        axs[2].plot(self.c_loss)
        fig.savefig(saved_log+"/log.png" )
        plt.close()
    def sample_generator_input(self, batch_size):
        # Generator inputs
        sampled_noise = np.random.normal(0, 1, (batch_size, 62))
        sampled_labels = np.random.randint(0, self.classes, batch_size).reshape(-1, 1)
        sampled_labels = to_categorical(sampled_labels, num_classes=self.classes)

        return sampled_noise, sampled_labels
    def train(self, epochs, batch_size=128, save_interval=50):
        # Adversarial ground truths
        (X_train, y_train), (_, _) = mnist.load_data()
        # Rescale -1 to 1
        X_train = (X_train.astype(np.float32) - 127.5) / 127.5
        X_train = np.expand_dims(X_train, axis=3)
        y_train = y_train.reshape(-1, 1)

        valid = np.ones((batch_size, 1))
        fake = -np.ones((batch_size, 1))
        #gene=self.data_loader.load_batch(batch_size)
        for epoch in range(epochs):
            # ---------------------
            #  Train Discriminator
            # ---------------------
            # Select a random half of images
            #imgs=next(gene)
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            imgs = X_train[idx]
            img_labels=to_categorical(y_train[idx],self.classes)
            sampled_noise, sampled_labels = self.sample_generator_input(batch_size)
            gen_input = np.concatenate((sampled_noise, sampled_labels), axis=1)
            #noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            # Train the discriminator (real classified as ones and generated as zeros)
            d_loss = self.discriminator.train_on_batch([imgs, gen_input],
                                                [valid, fake,img_labels])

            # ---------------------
            #  Train Generator
            # ---------------------
            # Train the generator (wants discriminator to mistake images as real)
            g_loss = self.combined.train_on_batch(gen_input, [fake, sampled_labels])
            # Plot the progress
            print ("%d [D loss: %.2f, c_loss: %.2f] [g loss: %.2f c_g_loss: %.2f]" % (epoch, d_loss[1]+d_loss[2], d_loss[3], g_loss[1],g_loss[2]))
            #print(d_loss)
            #print(g_loss)
            self.d_loss.append(d_loss[0])
            self.g_loss.append(g_loss[1])
            self.c_loss.append(g_loss[2])
            #print(d_loss,g_loss)
            # If at save interval => save generated image samples
            if epoch % save_interval == 0:
                self.save_imgs(epoch)
            if epoch % (save_interval) == 0:
                self.save_model()
                self.save_log()
                self.plot_log()
if __name__ == '__main__':
    mygan = MyGAN()
    mygan.train(epochs=500000, batch_size=4, save_interval=50)
