import tensorflow as tf
from tensorflow import keras
import tensorflow_addons as tfa
from tensorflow.keras.layers import Input, Concatenate, LeakyReLU, UpSampling2D, Conv2D, Conv2DTranspose
from tensorflow.keras.layers import BatchNormalization, Activation, ZeroPadding2D, MaxPooling2D
from tensorflow.keras.initializers import RandomNormal
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
    def __init__(self, image_shape:tuple, resized_image_shape:tuple, data_generateur_train:PairGenerateur, data_generateur_validation:PairGenerateur, data_generateur_test:PairGenerateur):
        super(CycleGan, self).__init__()
        self.image_shape = image_shape
        self.resized_image_shape = resized_image_shape
        self.data_generateur_train = data_generateur_train
        self.data_generateur_validation = data_generateur_validation
        self.data_generateur_test = data_generateur_test

        self.nb_filtres_g = 24
        self.nb_filtres_d = 24
        
        self.lambda_cycle = 5.0
        self.lambda_identity = 0.1 * self.lambda_cycle

        self.discriminateur_patch = (int(self.resized_image_shape[0] / 2**4) - 1, int(self.resized_image_shape[1] / 2**4), 1)

        self.epoch_debut = 0
        self.batch_debut = 0

        self.creer_modeles()

        if cfg.charger_modeles:
            self.charger_optimizer(cfg.charger_epoch, cfg.charger_batch)
        else:
            self.optimizer = Adam(0.0002)
        
        self.construire_combined()
        self.compiler_modeles()

        if cfg.charger_modeles:
            self.charger_poids(cfg.charger_epoch, cfg.charger_batch)

            self.epoch_debut = cfg.charger_epoch
            self.batch_debut = cfg.charger_batch
    def creer_modeles(self):
        #construit les discriminateurs
        self.discriminateur_simulation = self.build_discriminateur()
        self.discriminateur_robot = self.build_discriminateur()

        self.discriminateur_simulation._name = 'discriminateur_simulation'
        self.discriminateur_robot._name = 'discriminateur_robot'

        #construit les generateurs
        self.generateur_sim2robot = self.build_generateur()
        self.generateur_sim2robot._name = 'generateur_sim2robot'
        self.generateur_robot2sim = self.build_generateur()
        self.generateur_robot2sim._name = 'generateur_robot2sim'
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
        true_simu = Input(shape=self.resized_image_shape) #vraie simu
        true_robot = Input(shape=self.resized_image_shape) #vraie robot
        # Translate images to the other domain
        fake_robot = self.generateur_sim2robot(true_simu) #fake robot
        fake_simu = self.generateur_robot2sim(true_robot) #fake simu
        # Translate images back to original domain
        reconstr_simu = self.generateur_robot2sim(fake_robot) #fake fake simu
        reconstr_robot = self.generateur_sim2robot(fake_simu) #fake fake robot
        # Identity mapping of images
        robot_identity = self.generateur_sim2robot(true_robot)
        simu_identity = self.generateur_robot2sim(true_simu)
        # Discriminators determines validity of translated images
        valid_simu = self.discriminateur_simulation(fake_simu)
        valid_robot = self.discriminateur_robot(fake_robot)
        # Combined model trains generators to fool discriminators
        self.combined = Model(inputs=[true_simu, true_robot],
                            outputs=[ valid_simu, valid_robot,
                                      reconstr_simu, reconstr_robot,
                                      simu_identity, robot_identity ])

    def build_unet(self, d0):
        """U-Net Generator"""
        def conv2d(layer_input, filters, f_size=2):
            """Layers used during downsampling"""
            d = Conv2D(filters,
                        kernel_size=f_size,
                        strides=1,
                        padding='same',
                        activation=LeakyReLU()
                    )(layer_input)
            d = tfa.layers.InstanceNormalization()(d)
            d = MaxPooling2D(2)(d)
            return d

        def deconv2d(layer_input, skip_input, filters, f_size=2):
            """Layers used during upsampling"""
            u = Conv2DTranspose(filters,
                                kernel_size=f_size,
                                strides=(2, 2),
                                padding='same',
                                activation=LeakyReLU()
                            )(layer_input)
            u = tfa.layers.InstanceNormalization()(u)
            u = Concatenate()([u, skip_input])
            return u

        # Downsampling
        d1 = conv2d(d0, self.nb_filtres_g)
        d2 = conv2d(d1, self.nb_filtres_g*2)
        d3 = conv2d(d2, self.nb_filtres_g*4)
        d4 = conv2d(d3, self.nb_filtres_g*8)

        d4 = Conv2D(self.nb_filtres_g*12, kernel_size=1, strides=1,activation=LeakyReLU())(d4)

        # Upsampling
        u1 = deconv2d(d4, d3, self.nb_filtres_g*4)
        u2 = deconv2d(u1, d2, self.nb_filtres_g*2)
        u3 = deconv2d(u2, d1, self.nb_filtres_g)
        u4 = deconv2d(u3, d0, self.nb_filtres_g)

        return u4

    def build_generateur(self):
        d0 = keras.layers.Input(shape=self.resized_image_shape)
        d1 = self.build_unet(d0)
        d1 = Concatenate()([d1, d0])
        d2 = self.build_unet(d1)
        d2 = Concatenate()([d2, d0])
        d3 = self.build_unet(d2)
        d3 = Concatenate()([d3, d0])
        output_img = Conv2D(64, kernel_size=3, strides=1, padding='same', activation=LeakyReLU())(d3)
        output_img = Conv2D(self.resized_image_shape[-1], kernel_size=1, strides=1, padding='same', activation=LeakyReLU())(output_img)
        return Model(d0, output_img)

    def build_discriminateur(self):
        def d_layer(layer_input, filters, f_size=3, normalization=True, pooling=True):
            """Discriminator layer"""
            d = Conv2D(filters, kernel_size=f_size, strides=1, padding='same', activation=LeakyReLU())(layer_input)
            if normalization:
                d = tfa.layers.InstanceNormalization()(d)
            if pooling:
                d = MaxPooling2D()(d)
            return d

        img = keras.layers.Input(shape=self.resized_image_shape)

        d1 = d_layer(img, self.nb_filtres_d, normalization=False)
        d2 = d_layer(d1, self.nb_filtres_d*2)
        d3 = d_layer(d2, self.nb_filtres_d*4)
        d4 = d_layer(d3, self.nb_filtres_d*6)
        d5 = d_layer(d4, self.nb_filtres_d*8, pooling=False)

        validity = Conv2D(1, kernel_size=3, strides=1, padding='same')(d5)

        return Model(img, validity[:,:7])
    
    def train(self, epochs:int, echantillon_intervalle, sauvegarde_intervalle):
        start_time = datetime.datetime.now()
        for epoch in range(self.epoch_debut, epochs):
            for i, batch in enumerate(self.data_generateur_train.generer_paires(self.batch_debut)):
                no_batch = i + self.batch_debut
                if no_batch > len(self.data_generateur_train):
                    break
                
                #appelle `train_on_batch` sur le modele pour faire l'entrainement
                d_loss, g_loss = self.execute_model_on_batch(batch, Model.train_on_batch)

                elapsed_time = datetime.datetime.now() - start_time

                # Plot the progress
                print ("[Epoch %d/%d] [Batch %d/%d] [D loss: %f, acc: %3d%%] [G loss: %05f] time: %s " \
                                                                        % ( epoch, epochs,
                                                                            no_batch, self.data_generateur_train.nb_batches(),
                                                                            d_loss[0], 100*d_loss[1],
                                                                            g_loss[1],
                                                                            elapsed_time))
                if no_batch % echantillon_intervalle == 0:
                    self.sauvegarde_echantillons(epoch, no_batch)
                if no_batch % sauvegarde_intervalle == 0:
                    print('sauvegarde du modele...')
                    self.sauvegarde_modeles(epoch, no_batch)
            self.batch_debut = 0
    
    def execute_model_on_batch(self, batch, fonction_execution):
        # Adversarial loss ground truths
        valid = np.ones((self.data_generateur_train.batch_size, ) + self.discriminateur_patch)
        fake = np.zeros((self.data_generateur_train.batch_size, ) + self.discriminateur_patch)

        # Translate images to opposite domain
        fake_robot = self.generateur_sim2robot.predict(batch[:,0,:,:])
        fake_sim = self.generateur_robot2sim.predict(batch[:,1,:,:])

        # Train the discriminators (original images = real / translated = Fake)
        d_sim_loss_real = fonction_execution(self.discriminateur_simulation, batch[:,0,:,:], valid)
        d_sim_loss_fake = fonction_execution(self.discriminateur_simulation, fake_sim, fake)
        d_sim_loss = 0.5 * np.add(d_sim_loss_real, d_sim_loss_fake)

        d_rob_loss_real = fonction_execution(self.discriminateur_robot, batch[:,1,:,:], valid)
        d_rob_loss_fake = fonction_execution(self.discriminateur_robot, fake_robot, fake)
        d_rob_loss = 0.5 * np.add(d_rob_loss_real, d_rob_loss_fake)

        # Total discriminator loss
        d_loss = 0.5 * np.add(d_sim_loss, d_rob_loss)

        # Train the generators
        g_loss = fonction_execution(self.combined, [batch[:,0,:,:], batch[:,1,:,:]],
                                                [valid, valid,
                                                batch[:,0,:,:], batch[:,1,:,:],
                                                batch[:,0,:,:], batch[:,1,:,:]])
        return d_loss, g_loss

    def ycbcr2rgb(self, img_ycbcr:np.array):
        # Rescale image 0 - 1
        img_ycbcr = 0.5 * img_ycbcr + 0.5
        #convertion en RGB
        img = img_ycbcr*255
        xform = np.array([[1, 0, 1.402], [1, -0.34414, -.71414], [1, 1.772, 0]])
        img[0][:,:,1:] -= 128
        img[0] = img[0].dot(xform.T)
        np.putmask(img, img > 255, 255)
        np.putmask(img, img < 0, 0)
        return img

    def sauvegarde_echantillons(self, epoch, i):
        dossier = f'echantillons/epoch_{epoch:03}/batch_{i:04}'
        os.makedirs(dossier, exist_ok=True)
        batches = self.data_generateur_test.generer_paires()
        for paires in batches:
            for j, paire in enumerate(paires):
                image_simulation = np.array([paire[0]])
                image_robot = np.array([paire[1]])
                # Translate images to the other domain
                fake_robot = self.generateur_sim2robot.predict(image_simulation)
                fake_sim = self.generateur_robot2sim.predict(image_robot)
                # Translate back to original domain
                reconstr_sim = self.generateur_robot2sim.predict(fake_robot)
                reconstr_robot = self.generateur_sim2robot.predict(fake_sim)
                gen_imgs = np.concatenate([self.ycbcr2rgb(image_simulation), self.ycbcr2rgb(fake_robot), 
                                            self.ycbcr2rgb(reconstr_sim), self.ycbcr2rgb(image_robot), 
                                            self.ycbcr2rgb(fake_sim), self.ycbcr2rgb(reconstr_robot)])
                
                titles = ['Original', 'Translated', 'Reconstructed']
                fig, axs = plt.subplots(2, 3)
                cnt = 0
                for r in range(2):
                    for c in range(3):
                        axs[r,c].imshow((gen_imgs[cnt]).astype(np.uint8))
                        axs[r,c].set_title(titles[c])
                        axs[r,c].axis('off')
                        cnt += 1
                print(f'{dossier}/exemple_{j:02}.png')
                fig.savefig(f'{dossier}/exemple_{j:02}.png')
                plt.close()

    def sauvegarde_modeles(self, epoch, batch):
        dossier = f'modeles/epoch_{epoch:03}/batch_{batch:04}'
        os.makedirs(dossier, exist_ok=True)
        if os.path.exists(f'{dossier}/discriminateur_simulation.h5'):
            return
        self.discriminateur_simulation.save_weights(f'{dossier}/discriminateur_simulation.tf')
        self.discriminateur_robot.save_weights(f'{dossier}/discriminateur_robot.tf')
        self.generateur_sim2robot.save_weights(f'{dossier}/generateur_sim2robot.tf')
        self.generateur_robot2sim.save_weights(f'{dossier}/generateur_robot2sim.tf')
        with open(f'{dossier}/optimizer.pkl', 'wb') as opti_file:
            pickle.dump(self.optimizer, opti_file)

    def charger_poids(self, epoch, batch):
        dossier = f'modeles/epoch_{epoch:03}/batch_{batch:04}'
        self.discriminateur_simulation.load_weights(f'{dossier}/discriminateur_simulation.tf')
        self.discriminateur_robot.load_weights(f'{dossier}/discriminateur_robot.tf')
        self.generateur_sim2robot.load_weights(f'{dossier}/generateur_sim2robot.tf')
        self.generateur_robot2sim.load_weights(f'{dossier}/generateur_robot2sim.tf')
    
    def charger_optimizer(self, epoch, batch):
        dossier = f'modeles/epoch_{epoch:03}/batch_{batch:04}'
        with open(f'{dossier}/optimizer.pkl', 'rb') as opti_file:
            self.optimizer = pickle.load(opti_file)
