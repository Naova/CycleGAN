import tensorflow as tf
from tensorflow import keras
import tensorflow_addons as tfa
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Concatenate, LeakyReLU, UpSampling2D, Conv2D
from tensorflow.keras.layers import BatchNormalization, Activation, ZeroPadding2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

import numpy as np
import matplotlib.pyplot as plt
import datetime
import pickle
import os

from dataset_loader import PairGenerateur
import config as cfg

class CycleGan():
    def __init__(self, image_shape:tuple, data_generateur_train:PairGenerateur, data_generateur_test:PairGenerateur):
        super(CycleGan, self).__init__()
        self.image_shape = image_shape
        self.data_generateur_train = data_generateur_train
        self.data_generateur_test = data_generateur_test

        self.nb_filtres_g = 32
        self.nb_filtres_d = 64
        
        self.lambda_cycle = 10.0
        self.lambda_identity = 0.1 * self.lambda_cycle

        self.discriminateur_patch = (int(self.image_shape[0] / 2**4), int(self.image_shape[1] / 2**4), 1)

        self.epoch_debut = 0
        self.batch_debut = 0

        self.optimizer = Adam()

        if cfg.charger_modeles:
            self.charger_modeles(cfg.charger_epoch, cfg.charger_batch)
            self.epoch_debut = cfg.charger_epoch
            self.batch_debut = cfg.charger_batch
        else:
            self.creer_modeles()
        self.construire_combined()
        self.compiler_modeles()

    def creer_modeles(self):
        #construit les discriminateurs
        self.discriminateur_simulation = self.build_discriminateur()
        self.discriminateur_robot = self.build_discriminateur()

        self.discriminateur_simulation._name = 'discriminateur_simulation'
        self.discriminateur_robot._name = 'discriminateur_robot'

        #construit les generateurs
        self.generateur_sim_robot = self.build_generateur()
        self.generateur_sim_robot._name = 'generateur_sim_robot'
        self.generateur_robot_sim = self.build_generateur()
        self.generateur_robot_sim._name = 'generateur_robot_sim'
    def compiler_modeles(self):
        self.discriminateur_simulation.trainable = True
        self.discriminateur_robot.trainable = True
        self.discriminateur_simulation.compile(loss='mse', optimizer=self.optimizer, metrics=['accuracy'])
        self.discriminateur_robot.compile(loss='mse', optimizer=self.optimizer, metrics=['accuracy'])
        self.discriminateur_simulation.trainable = False
        self.discriminateur_robot.trainable = False
        self.combined.compile(loss=['mse', 'mse',
                                    'mae', 'mae',
                                    'mae', 'mae'],
                            loss_weights=[  1, 1,
                                            self.lambda_cycle, self.lambda_cycle,
                                            self.lambda_identity, self.lambda_identity ],
                            optimizer=self.optimizer)
    def construire_combined(self):
        optimizer = Adam(0.002)
        img_A = Input(shape=self.image_shape)
        img_B = Input(shape=self.image_shape)
        # Translate images to the other domain
        fake_B = self.generateur_sim_robot(img_A)
        fake_A = self.generateur_robot_sim(img_B)
        # Translate images back to original domain
        reconstr_A = self.generateur_sim_robot(fake_B)
        reconstr_B = self.generateur_robot_sim(fake_A)
        # Identity mapping of images
        img_A_id = self.generateur_sim_robot(img_A)
        img_B_id = self.generateur_robot_sim(img_B)
        # Discriminators determines validity of translated images
        valid_A = self.discriminateur_simulation(fake_A)
        valid_B = self.discriminateur_robot(fake_B)
        # Combined model trains generators to fool discriminators
        self.combined = Model(inputs=[img_A, img_B],
                            outputs=[ valid_A, valid_B,
                                        reconstr_A, reconstr_B,
                                        img_A_id, img_B_id ])

    def build_generateur(self):
        """U-Net Generator"""
        def conv2d(layer_input, filters, f_size=4):
            """Layers used during downsampling"""
            d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            d = tfa.layers.InstanceNormalization()(d)
            return d

        def deconv2d(layer_input, skip_input, filters, f_size=4):
            """Layers used during upsampling"""
            u = UpSampling2D(size=2)(layer_input)
            u = Conv2D(filters, kernel_size=f_size, strides=1, padding='same', activation='relu')(u)
            u = tfa.layers.InstanceNormalization()(u)
            u = Concatenate()([u, skip_input])
            return u

        # Image input
        d0 = keras.layers.Input(shape=self.image_shape)

        # Downsampling
        d1 = conv2d(d0, self.nb_filtres_g)
        d2 = conv2d(d1, self.nb_filtres_g*2)
        d3 = conv2d(d2, self.nb_filtres_g*4)
        d4 = conv2d(d3, self.nb_filtres_g*8)

        # Upsampling
        u1 = deconv2d(d4, d3, self.nb_filtres_g*4)
        u2 = deconv2d(u1, d2, self.nb_filtres_g*2)
        u3 = deconv2d(u2, d1, self.nb_filtres_g)

        u4 = UpSampling2D(size=2)(u3)
        output_img = Conv2D(self.image_shape[-1], kernel_size=4, strides=1, padding='same', activation='tanh')(u4)

        return Model(d0, output_img)

    def build_discriminateur(self):
        def d_layer(layer_input, filters, f_size=4, normalization=True):
            """Discriminator layer"""
            d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            if normalization:
                d = tfa.layers.InstanceNormalization()(d)
            return d

        img = keras.layers.Input(shape=self.image_shape)

        d1 = d_layer(img, self.nb_filtres_d, normalization=False)
        d2 = d_layer(d1, self.nb_filtres_d*2)
        d3 = d_layer(d2, self.nb_filtres_d*4)
        d4 = d_layer(d3, self.nb_filtres_d*8)

        validity = Conv2D(1, kernel_size=4, strides=1, padding='same')(d4)

        return Model(img, validity)
    
    def train(self, epochs:int, echantillon_intervalle = 10, savegarde_intervalle = 50):
        start_time = datetime.datetime.now()
        # Adversarial loss ground truths
        valid = np.ones((self.data_generateur_train.batch_size, ) + self.discriminateur_patch)
        fake = np.zeros((self.data_generateur_train.batch_size, ) + self.discriminateur_patch)
        for epoch in range(self.epoch_debut, epochs):
            for i, batch in enumerate(self.data_generateur_train.generer_paires()):
                if i + self.batch_debut > len(self.data_generateur_train):
                    break
                # ----------------------
                #  Train Discriminators
                # ----------------------

                # Translate images to opposite domain
                fake_robot = self.generateur_sim_robot.predict(batch[:,0,:,:])
                fake_sim = self.generateur_robot_sim.predict(batch[:,1,:,:])

                # Train the discriminators (original images = real / translated = Fake)
                d_sim_loss_real = self.discriminateur_simulation.train_on_batch(batch[:,0,:,:], valid)
                d_sim_loss_fake = self.discriminateur_simulation.train_on_batch(fake_sim, fake)
                d_sim_loss = 0.5 * np.add(d_sim_loss_real, d_sim_loss_fake)

                d_rob_loss_real = self.discriminateur_robot.train_on_batch(batch[:,1,:,:], valid)
                d_rob_loss_fake = self.discriminateur_robot.train_on_batch(fake_robot, fake)
                d_rob_loss = 0.5 * np.add(d_rob_loss_real, d_rob_loss_fake)

                # Total disciminator loss
                d_loss = 0.5 * np.add(d_sim_loss, d_rob_loss)

                # ------------------
                #  Train Generators
                # ------------------

                # Train the generators
                g_loss = self.combined.train_on_batch([batch[:,0,:,:], batch[:,1,:,:]],
                                                        [valid, valid,
                                                        batch[:,0,:,:], batch[:,1,:,:],
                                                        batch[:,0,:,:], batch[:,1,:,:]])

                elapsed_time = datetime.datetime.now() - start_time

                # Plot the progress
                print ("[Epoch %d/%d] [Batch %d/%d] [D loss: %f, acc: %3d%%] [G loss: %05f] time: %s " \
                                                                        % ( epoch, epochs,
                                                                            i + self.batch_debut, self.data_generateur_train.nb_batches(),
                                                                            d_loss[0], 100*d_loss[1],
                                                                            g_loss[0],
                                                                            elapsed_time))
                if i % echantillon_intervalle == 0:
                    self.sauvegarde_echantillons(epoch, i + self.batch_debut)
                if i % savegarde_intervalle == 0:
                    self.sauvegarde_modeles(epoch, i + self.batch_debut)

    def sauvegarde_echantillons(self, epoch, i):
        dossier = f'echantillons/epoch_{epoch:03}/batch_{i:04}'
        os.makedirs(dossier, exist_ok=True)
        batches = self.data_generateur_test.generer_paires()
        for paires in batches:
            for j, paire in enumerate(paires):
                image_simulation = np.array([paire[0]])
                image_robot = np.array([paire[1]])
                # Translate images to the other domain
                fake_robot = self.generateur_sim_robot.predict(image_simulation)
                fake_sim = self.generateur_robot_sim.predict(image_robot)
                # Translate back to original domain
                reconstr_sim = self.generateur_robot_sim.predict(fake_robot)
                reconstr_robot = self.generateur_sim_robot.predict(fake_sim)
                gen_imgs = np.concatenate([image_simulation, fake_robot, reconstr_sim, image_robot, fake_sim, reconstr_robot])
                # Rescale images 0 - 1
                gen_imgs = 0.5 * gen_imgs + 0.5

                titles = ['Original', 'Translated', 'Reconstructed']
                fig, axs = plt.subplots(2, 3)
                cnt = 0
                for r in range(2):
                    for c in range(3):
                        axs[r,c].imshow((gen_imgs[cnt]*255).astype(np.uint8))
                        axs[r,c].set_title(titles[c])
                        axs[r,c].axis('off')
                        cnt += 1
                print(f'{dossier}/exemple_{j:02}.png')
                fig.savefig(f'{dossier}/exemple_{j:02}.png')
                plt.close()

    def sauvegarde_modeles(self, epoch, batch):
        dossier = f'modeles/epoch_{epoch:03}/batch_{batch:04}'
        os.makedirs(dossier, exist_ok=True)
        self.discriminateur_simulation.save(f'{dossier}/discriminateur_simulation.h5')
        self.discriminateur_robot.save(f'{dossier}/discriminateur_robot.h5')
        self.generateur_sim_robot.save(f'{dossier}/generateur_sim_robot.h5')
        self.generateur_robot_sim.save(f'{dossier}/generateur_robot_sim.h5')
        with open(f'{dossier}/optimizer.pkl', 'wb') as opti_file:
            pickle.dump(self.optimizer, opti_file)

    def charger_modeles(self, epoch, batch):
        dossier = f'modeles/epoch_{epoch:03}/batch_{batch:04}'
        self.discriminateur_simulation = keras.models.load_model(f'{dossier}/discriminateur_simulation.h5')
        self.discriminateur_robot = keras.models.load_model(f'{dossier}/discriminateur_robot.h5')
        self.generateur_sim_robot = keras.models.load_model(f'{dossier}/generateur_sim_robot.h5')
        self.generateur_robot_sim = keras.models.load_model(f'{dossier}/generateur_robot_sim.h5')
        with open(f'{dossier}/optimizer.pkl', 'rb') as opti_file:
            self.optimizer = pickle.load(opti_file)
