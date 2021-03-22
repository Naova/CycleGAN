import matplotlib.pyplot as plt

import dataset_loader
import config as cfg


def main():
    train, test = dataset_loader.create_dataset(16)

    #verification rapide du dataset
    for i in train.generer_paire(3):
        plt.imshow(i[0][0])
        plt.show()
        plt.imshow(i[0][1])
        plt.show()

if __name__ == '__main__':
    main()
