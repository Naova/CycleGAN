import numpy as np
import matplotlib.pyplot as plt
import random
import tqdm
import os
from PIL import Image

import config as cfg
from cycle_gan import CycleGan
import dataset_loader

if not cfg.charger_modeles:
    print('erreur de config?')
    exit()

def ycbcr2rgb(img_ycbcr:np.array):
    #convertion en RGB
    img = img_ycbcr*255
    xform = np.array([[1, 0, 1.402], [1, -0.34414, -.71414], [1, 1.772, 0]])
    img[0][:,:,1:] -= 128
    img[0] = img[0].dot(xform.T)
    np.putmask(img, img > 255, 255)
    np.putmask(img, img < 0, 0)
    return img / 255.0

def generer_dataset(generateur, dataset_simulation, to_rgb=False):
    for entree in tqdm.tqdm(dataset_simulation):
        image = entree.charger_image()
        image_generee = generateur.predict(np.array([image]))
        image = image * 0.5 + 0.5
        image_generee = image_generee * 0.5 + 0.5
        image_generee[image_generee < 0] = 0
        image_generee[image_generee > 1] = 1
        if to_rgb:
            image = ycbcr2rgb(np.array([image]))[0]
            image_generee = ycbcr2rgb(image_generee)[0]
        else:
            image_generee = image_generee[0]
            image_generee = (image_generee*255).astype(np.uint8)
        path = entree.path_image.replace('Simulation', 'Genere')
        index = path.index('batch_')
        path = path[:index] + f'epoch_{cfg.charger_epoch}_batch_000/' + path[index:]
        if not os.path.exists('/'.join(path.split('/')[:-1])):
                os.makedirs('/'.join(path.split('/')[:-1]))
        Image.fromarray(image_generee).save(path)

def main():
    dataset_simulation = dataset_loader.lire_entrees(cfg.dossier_brut_simulation, True, False)
    image_shape = (cfg.image_height, cfg.image_width, cfg.image_channels)
    resized_image_shape = (cfg.resized_image_height, cfg.resized_image_width, cfg.image_channels)
    cycleGan = CycleGan(image_shape, resized_image_shape, None, None, None)
    cycleGan.generateur_sim2robot.summary()
    generateur = cycleGan.generateur_sim2robot
    del cycleGan
    generer_dataset(generateur, dataset_simulation)

if __name__ == '__main__':
    main()
