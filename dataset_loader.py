import numpy as np
from pathlib import Path
import random

import config as cfg

class Entree:
    def __init__(self, path_image:str, flipper:bool):
        self.path_image = path_image
        self.flipper = flipper
    
    def charger_image(self):
        image = np.fromfile(self.path_image, dtype=np.float32)
        image = np.reshape(image, (cfg.image_height, cfg.image_width, 3))
        if self.flipper:
            image = np.fliplr(image)
        return image
    
    def __repr__(self):
        return '[Image] : ' + self.path_image + ', ' + str(self.flipper)

class PairGenerateur:
    def __init__(self, entrees_simulation:list, entrees_robot:list, batch_size:int):
        self.entrees_simulation = entrees_simulation
        self.entrees_robot = entrees_robot
        self.batch_size = batch_size
    
    def generer_paire(self, nb_batches=1):
        for _ in range(nb_batches):
            entrees_simulation = random.choices(self.entrees_simulation, k=self.batch_size)
            entrees_robot = random.choices(self.entrees_robot, k=self.batch_size)
            batch = [[i.charger_image(), j.charger_image()] for i, j in zip(entrees_simulation, entrees_robot)]

            yield np.array(batch)

def lire_entrees(dossier:str):
    images = list(Path(dossier).glob('*'))
    images = images * 2

    entrees = []

    for i, path_image in enumerate(images):
        flip = True if i > len(images) / 2 else False
        entrees.append(Entree(str(path_image), flip))
    
    return entrees

def split_dataset(entrees_simulation, entrees_robot, batch_size=16):
    random.shuffle(entrees_simulation)
    random.shuffle(entrees_robot)

    train = PairGenerateur(entrees_simulation[:-20], entrees_robot[:-20], batch_size)
    test = PairGenerateur(entrees_simulation[-20:], entrees_robot[-20:], batch_size)

    return train, test

def create_dataset(batch_size):
    entrees_simulation = lire_entrees(cfg.dossier_brut_simulation)
    entrees_robot = lire_entrees(cfg.dossier_brut_robot)

    train, test = split_dataset(entrees_simulation, entrees_robot, batch_size)
    return train, test
