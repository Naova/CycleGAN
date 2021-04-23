## Parametre du dataset
#chemins d'acces vers le dataset (simulation)
dataset_simulation_root = 'Dataset/Simulation/'
dossier_brut_simulation = dataset_simulation_root + 'Brut/'

#chemins d'acces vers le dataset (robot)
dataset_robot_root = 'Dataset/Robot/'
dossier_brut_robot = dataset_robot_root + 'Brut/'

#chemin d'acces vers le dataset (genere)
dataset_genere_root = 'Dataset/Genere/'
dossier_brut_genere = dataset_genere_root + 'Brut/'

## Parametres d'entrainement
#taille des batches du dataset
batch_size = 10

#resolution de l'image d'entree
image_height = 240
image_width = 320
image_channels = 3

#double virtuellement la taille du dataset
flipper_images = True

#boucle d'entrainement
max_epoch = 200
echantillon_intervalle = 25
sauvegarde_intervalle = 50
tensorboard_intervalle = 5

#reprendre un ancien entrainement?
charger_modeles = False
#si oui, quelle version continuer?
charger_epoch = 0
charger_batch = 0
