import matplotlib.pyplot as plt

from cycle_gan import CycleGan
import dataset_loader
import config as cfg


def main():
    train, validation, test = dataset_loader.create_dataset(cfg.batch_size)

    cycleGan = CycleGan((cfg.image_height, cfg.image_width, cfg.image_channels), train, validation, test)
    cycleGan.combined.summary()
    cycleGan.train(cfg.max_epoch, cfg.echantillon_intervalle, cfg.sauvegarde_intervalle, cfg.tensorboard_intervalle)
    breakpoint()

if __name__ == '__main__':
    main()
