import numpy as np
import matplotlib.pyplot as plt
import random
import tqdm

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
    return img

def generer_dataset(generateur, dataset_simulation, to_rgb=False):
    for entree in tqdm.tqdm(dataset_simulation):
        image = entree.charger_image()
        image_generee = generateur.predict(np.array([image]))
        image = image / 2 + 0.5
        image_generee = image_generee / 2 + 0.5
        if to_rgb:
            image = ycbcr2rgb(np.array([image]))[0]
            image_generee = ycbcr2rgb(image_generee)[0]
        else:
            image_generee = image_generee[0]
        path = entree.path_image.replace('Simulation', 'Genere')
        image_generee[image_generee < 0] = 0
        image_generee[image_generee > 255] = 255
        with open(path, 'wb') as f:
            image_generee.tofile(f)
        # plt.imshow((image * 255).astype(np.uint8))
        # plt.show()
        # plt.imshow((image_generee * 255).astype(np.uint8))
        # plt.show()

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
