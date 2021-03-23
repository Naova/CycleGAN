
#chemins d'acces vers le dataset (simulation)
dataset_simulation_root = 'Dataset/Simulation/'
dossier_brut_simulation = dataset_simulation_root + 'Brut/'
dossier_PNG_simulation = dataset_simulation_root + 'PNG/'

#chemins d'acces vers le dataset (robot)
dataset_robot_root = 'Dataset/Robot/'
dossier_brut_robot = dataset_robot_root + 'Brut/'
dossier_PNG_robot = dataset_robot_root + 'PNG/'


#resolution de l'image d'entree
image_height = 240
image_width = 320
image_channels = 3

#double virtuellement la taille du dataset
flipper_images = True

#continuer un entrainement ?
charger_modeles = True
#quelle version continuer?
charger_epoch = 0
charger_batch = 100
