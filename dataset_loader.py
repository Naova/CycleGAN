import numpy as np
from pathlib import Path
import random
from PIL import Image
import matplotlib.pyplot as plt

import fiftyone as fo
import fiftyone.zoo as foz

import config as cfg

dataset = foz.load_zoo_dataset("coco-2017", split="train", max_samples=1000)
coco_images = [info for info in dataset]

class Entree:
    def __init__(self, path_image:str, flipper:bool, augmentation:bool):
        self.path_image = path_image
        self.flipper = flipper
        self.augmentation = augmentation
    
    def rgb2ycbcr(self, x:np.array):
        y = 16 + x[:,:,0] * 65.738 / 256 + 129.057 * x[:,:,1]/256 + 25.064 * x[:,:,2] / 256
        cb = 128 - x[:,:,0] * 37.945 / 256 - 74.494 * x[:,:,1]/256 + 112.439 * x[:,:,2] / 256
        cr = 128 + x[:,:,0] * 112.439 / 256 - 94.154 * x[:,:,1]/256 - 18.285 * x[:,:,2] / 256
        img = np.stack([y, cb, cr], axis=-1)
        return img
    
    def charger_gazon(self):
        no_texture = random.randint(0, 10)
        img_texture = Image.open(f'textures/texture_{no_texture}.png')
        img_texture = img_texture.resize((cfg.resized_image_width, cfg.resized_image_height))
        img_texture = np.array(img_texture)[:,:,:-1]
        if random.random() < 0.5:
            img_texture = np.fliplr(img_texture)
        if random.random() < 0.5:
            img_texture = np.flipud(img_texture)
        return img_texture

    def ajouter_gazon(self, image):
        gazon = self.charger_gazon()
        gazon = self.rgb2ycbcr(gazon)
        dist = np.absolute(np.full((cfg.resized_image_height, cfg.resized_image_width, 3), (84, 90, 83)) - image)
        dist = np.sum(dist, -1)
        image[np.where(dist < 20)] = gazon[np.where(dist < 20)]
        return image

    def ajouter_coco(self, image):
        mask = image == [167, 137, 127] #couleur du background gris
        mask = mask[...,0] + mask[...,1] + mask[...,2]
        coco_image = random.choice(coco_images)
        i = Image.open(coco_image.filepath)
        i = i.resize((cfg.resized_image_width, cfg.resized_image_height))
        i = np.array(i)
        if len(i.shape) == 2:
            i = np.stack((i,)*3, axis=-1)
        i = self.rgb2ycbcr(i)
        image[mask] = i[mask]
        return image

    def charger_image(self):
        image = Image.open(self.path_image)
        image = image.resize((cfg.resized_image_width, cfg.resized_image_height), Image.NEAREST)
        image = np.array(image)
        if self.augmentation:
            image = self.ajouter_gazon(image)
            image = self.ajouter_coco(image)
            image = image + np.random.normal(scale=0.1)
        image = image / 255.
        if self.flipper:
            image = np.fliplr(image)
        return image * 2 - 1.0
    
    def __repr__(self):
        return '[Image] : ' + self.path_image + ', ' + str(self.flipper)

class PairGenerateur:
    def __init__(self, entrees_simulation:list, entrees_robot:list, batch_size:int):
        self.entrees_simulation = entrees_simulation
        self.entrees_robot = entrees_robot
        self.batch_size = batch_size
    
    def __len__(self):
        return len(self.entrees_simulation)
    
    def nb_batches(self):
        return int(len(self.entrees_simulation)/self.batch_size)

    def generer_paires(self, depart=0):
        for i in range(depart, self.nb_batches()):
            j = i + self.batch_size if i + self.batch_size <= len(self.entrees_robot) else len(self.entrees_robot)
            robot = self.entrees_robot[i:j]
            simu = random.choices(self.entrees_simulation, k=self.batch_size)
            batch = [[i.charger_image(), j.charger_image()] for i, j in zip(simu, robot)]
            yield np.array(batch)

def lire_entrees(dossier:str, augmentation:bool = False, flipper:bool = True):
    images = list(Path(dossier).glob('*/*'))
    if flipper:
        images = images * 2

    entrees = []

    for i, path_image in enumerate(images):
        if flipper:
            flip = True if i > len(images) / 2 else False
        else:
            flip = False
        entrees.append(Entree(str(path_image), flip, augmentation))
    
    return entrees

def split_dataset(entrees_simulation, entrees_robot, batch_size):
    random.shuffle(entrees_simulation)
    random.shuffle(entrees_robot)

    train = PairGenerateur(entrees_simulation[:-batch_size*2], entrees_robot[:-batch_size*2], batch_size)
    validation = PairGenerateur(entrees_simulation[-batch_size*2:-batch_size], entrees_robot[-batch_size*2:-batch_size], batch_size)
    test = PairGenerateur(entrees_simulation[-batch_size:], entrees_robot[-batch_size:], batch_size)

    return train, validation, test

def create_dataset(batch_size):
    entrees_simulation = lire_entrees(cfg.dossier_brut_simulation, True, True)
    entrees_robot = lire_entrees(cfg.dossier_brut_robot, False, False)

    train, validation, test = split_dataset(entrees_simulation, entrees_robot, batch_size)
    return train, validation, test
