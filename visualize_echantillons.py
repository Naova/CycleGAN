from PIL import Image
from pathlib import Path
import shutil
import config as cfg

def main():
    #dossiers = list(Path('echantillons/').glob('*'))
    for no_dossier in range(200):
        src = f'echantillons/epoch_{no_dossier:03}/batch_0000/exemple_00.png'#dossier.as_posix() + '/batch_0000'
        print(src)
        dest = f'visualisation/epoch_{no_dossier:03}_batch_0000.png'
        shutil.copyfile(src, dest)


if __name__ == '__main__':
    main()
