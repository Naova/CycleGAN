import dataset_loader

import matplotlib.pyplot as plt


def main():
    train, validation, test = dataset_loader.create_dataset(16)
    
    #verification rapide du dataset
    print(train[0].shape)
    for i in range(3):
        plt.imshow(train[0][i][0])
        plt.show()
        plt.imshow(train[0][i][1])
        plt.show()

if __name__ == '__main__':
    main()
