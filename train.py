import matplotlib.pyplot as plt

from cycle_gan import CycleGan
import dataset_loader
import config as cfg


def main():
    train, test = dataset_loader.create_dataset(8)

    cycleGan = CycleGan((cfg.image_height, cfg.image_width, cfg.image_channels), train, test)
    cycleGan.combined.summary()
    cycleGan.train(200)
    breakpoint()

if __name__ == '__main__':
    main()
