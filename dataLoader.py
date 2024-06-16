#######################################################################
# Classe per carregar els fitxers d'audio. Crida la classe loadAudio
# per fer el preprocessament dels arxius
#######################################################################

import loadAudio as la
from torch.utils.data import DataLoader, Dataset


class SoundDS(Dataset):
    def __init__(self, df, data_path):
        self.df = df
        self.data_path = str(data_path)
        self.duration = 4000

        #self.duration = 5000

        self.sr = 44100
        self.channel = 2
        self.shift_pct = 0.4

    def __len__(self):
        return len(self.df)

    # Agafa un element del conjunt
    def __getitem__(self, idx):
        # Ruta a la carpeta amb les dades
        audio_file = self.data_path + self.df.loc[idx, 'relative_path']
        # Classe definida pel label
        class_id = self.df.loc[idx, 'classID']

        # Classe per ESC-50
        #class_id = self.df.loc[idx, 'target']

        aud = la.AudioUtil.open(audio_file)
       
        # Tots els processos d'audio
        reaud = la.AudioUtil.resample(aud, self.sr)
        rechan = la.AudioUtil.rechannel(reaud, self.channel)

        dur_aud = la.AudioUtil.pad_trunc(rechan, self.duration)
        shift_aud = la.AudioUtil.time_shift(dur_aud, self.shift_pct)
        sgram = la.AudioUtil.spectro_gram(shift_aud, n_mels=64, n_fft=1024, hop_len=None)
        #aug_sgram = sgram
        aug_sgram = la.AudioUtil.spectro_augment(sgram, max_mask_pct=0.1, n_freq_masks=2, n_time_masks=2)

        return aug_sgram, class_id
