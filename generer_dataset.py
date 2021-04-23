import numpy as np
import matplotlib.pyplot as plt
import random

import config as cfg
from cycle_gan import CycleGan
import dataset_loader

if not cfg.charger_modeles:
    print('erreur de config?')
    exit()

def ycbcr2rgb(img_ycbcr:np.array):
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

def generer_dataset(generateur, dataset_simulation):
    for entree in dataset_simulation:
        image = entree.charger_image()
        image = ycbcr2rgb(np.array([image]))[0]
        image_generee = generateur.predict(np.array([image]))
        image_generee = ycbcr2rgb(image_generee)[0]
        plt.imshow(image.astype(np.uint8))
        plt.show()
        plt.imshow(image_generee.astype(np.uint8))
        plt.show()

def main():
    dataset_simulation = dataset_loader.lire_entrees(cfg.dossier_brut_simulation)
    random.shuffle(dataset_simulation)
    cycleGan = CycleGan((cfg.image_height, cfg.image_width, cfg.image_channels), None, None, None)
    cycleGan.generateur_sim2robot.summary()
    generateur = cycleGan.generateur_sim2robot
    del cycleGan
    generer_dataset(generateur, dataset_simulation)

if __name__ == '__main__':
    main()
