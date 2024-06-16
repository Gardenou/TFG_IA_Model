#######################################################
# carrega les metadades i les dades i crea els splits
# de conjunts d'entrenament i validació
#######################################################

import dataLoader as dl
from torch.utils.data import DataLoader, Dataset, random_split
import torch
import pandas as pd
from pathlib import Path

# Path on tenim la carpeta del conjunt de dades
download_path = 'C:\\Users\\denou\\Downloads\\UrbanSound8K\\UrbanSound8K'
# ESC-50
#download_path = 'C:\\Users\\denou\\Downloads\\ESC-50-master\\ESC-50-master'

# Path amb l'arxiu de les metadades
metadata_file = 'C:\\Users\\denou\\Downloads\\UrbanSound8K\\UrbanSound8K\\metadata\\UrbanSound8K.csv'
# ESC-50
#metadata_file = 'C:\\Users\\denou\\Downloads\\ESC-50-master\\ESC-50-master\\meta\\esc50.csv'

df = pd.read_csv(metadata_file)

# Construeix la ruta al fitxer
df['relative_path'] = '/fold' + df['fold'].astype(str) + '/' + df['slice_file_name'].astype(str)
# ESC-50
#df['relative_path'] = '/' + df['filename'].astype(str)

# Busquem al .csv la classe de l'arxiu 
df = df[['relative_path', 'classID']]
# ESC-50
#df = df[['relative_path', 'target']]

# Path on tenim els fitxers d'audio
data_path = 'C:\\Users\\denou\\Downloads\\UrbanSound8K\\UrbanSound8K\\audio'
# ESC-50
#data_path = 'C:\\Users\\denou\\Downloads\\ESC-50-master\\ESC-50-master\\audio'


myds = dl.SoundDS(df, data_path)

# Random split dels conjunts 80/20
num_items = len(myds)
num_train = round(num_items * 0.8)
num_val = num_items - num_train
train_ds, val_ds = random_split(myds, [num_train, num_val])

# variables amb les dades carregades
train_dl = torch.utils.data.DataLoader(train_ds, batch_size=32, shuffle=True)
val_dl = torch.utils.data.DataLoader(val_ds, batch_size=32, shuffle=False)
