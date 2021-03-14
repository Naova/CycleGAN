import numpy as np
from pathlib import Path
import random
import tensorflow.keras as keras

import config as cfg

class Entree:
    def __init__(self, path_simulation:str, path_robot:str, flipper_simulation:bool, flipper_robot:bool):
        self.path_simulation = path_simulation
        self.path_robot = path_robot
        self.flipper_simulation = flipper_simulation
        self.flipper_robot = flipper_robot
    
    def charger_paire(self):
        image_simulation = np.fromfile(self.path_simulation, dtype=np.float32)
        image_simulation = np.reshape(image_simulation, (cfg.image_height, cfg.image_width, 3))
        image_robot = np.fromfile(self.path_robot, dtype=np.float32)
        image_robot = np.reshape(image_robot, (cfg.image_height, cfg.image_width, 3))
        if self.flipper_simulation:
            image_simulation = np.fliplr(image_simulation)
        if self.flipper_robot:
            image_robot = np.fliplr(image_robot)
        return [image_simulation, image_robot]
    
    def __repr__(self):
        return '[Entree] : ' + self.path_simulation + ' ' + self.path_robot + ', ' + str(self.flipper_simulation) + ' ' + str(self.flipper_robot)

class KerasSequence(keras.utils.Sequence):
    def __init__(self, entries:list, batch_size:int, x_function):
        self.entries = entries
        self.batch_size = batch_size
        self.x_function = x_function
    
    def __len__(self):
        return (np.ceil(len(self.entries) / float(self.batch_size))).astype(np.int)
    
    def __getitem__(self, idx):
        batch_x = [self.x_function(entry) for entry in self.entries[idx * self.batch_size : (idx+1) * self.batch_size]]
        return np.array(batch_x)

def lire_entrees(dossier_simulation:str, dossier_robot:str):
    images_simulation = list(Path(dossier_simulation).glob('*'))
    images_robot = list(Path(dossier_robot).glob('*'))

    images_simulation = images_simulation * 2
    images_robot = images_robot * 2

    entrees = []

    for i, image_sim in enumerate(images_simulation):
        flip_sim = True if i > len(images_simulation) / 2 else False
        image_rob_index = random.randint(0, len(images_robot) - 1)
        image_rob = images_robot[image_rob_index]
        flip_rob = True if image_rob_index > len(images_simulation) / 2 else False
        entrees.append(Entree(str(image_sim), str(image_rob), flip_sim, flip_rob))
    
    return entrees

def split_dataset(entrees, batch_size=16):
    random.shuffle(entrees)
    ratio_train = 0.95 #90%
    ratio_test = 20 / len(entrees) #nombre fixe, pas besoin de plus
    #ratio_validation = 10% - 20

    i = int(len(entrees) * ratio_train)#train
    j = int(len(entrees) * (ratio_train + ratio_test))#test

    train = KerasSequence(entrees[:i], batch_size, Entree.charger_paire)
    validation = KerasSequence(entrees[j:], batch_size, Entree.charger_paire)
    test = entrees[i:j]

    return train, validation, test

def create_dataset(batch_size):
    entrees = lire_entrees(cfg.dossier_brut_simulation, cfg.dossier_brut_robot)
    train, validation, test = split_dataset(entrees, batch_size)
    return train, validation, test
