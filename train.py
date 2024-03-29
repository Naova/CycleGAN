import matplotlib.pyplot as plt

from cycle_gan import CycleGan
import dataset_loader
import config as cfg


def main():
    train, validation, test = dataset_loader.create_dataset(cfg.batch_size)
    image_shape = (cfg.image_height, cfg.image_width, cfg.image_channels)
    resized_image_shape = (cfg.resized_image_height, cfg.resized_image_width, cfg.image_channels)
    cycleGan = CycleGan(image_shape, resized_image_shape, train, validation, test)
    cycleGan.combined.summary()
    cycleGan.train(cfg.max_epoch, cfg.echantillon_intervalle, cfg.sauvegarde_intervalle)

if __name__ == '__main__':
    main()
