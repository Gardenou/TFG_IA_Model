
import matplotlib.pyplot as plt
import loadAudio as la
from torchaudio import load

duration = 4000
sr = 44100
channel = 2
shift_pct = 0.4

# Carrega el fitxer i fes el rpeprcoessament
audio_file = 'C:\\Users\\denou\\Downloads\\UrbanSound8K\\UrbanSound8K\\audio\\fold4\\74723-3-0-0.wav'

aud = la.AudioUtil.open(audio_file)
sgram_orig = la.AudioUtil.spectro_gram(aud, n_mels=64, n_fft=1024, hop_len=None)

reaud = la.AudioUtil.resample(aud, sr)
sgram_re = la.AudioUtil.spectro_gram(reaud, n_mels=64, n_fft=1024, hop_len=None)

rechan = la.AudioUtil.rechannel(reaud, channel)
sgram_rechan = la.AudioUtil.spectro_gram(rechan, n_mels=64, n_fft=1024, hop_len=None)

dur_aud = la.AudioUtil.pad_trunc(rechan, duration)
sgram_dur = la.AudioUtil.spectro_gram(dur_aud, n_mels=64, n_fft=1024, hop_len=None)

shift_aud = la.AudioUtil.time_shift(dur_aud, shift_pct)
sgram = la.AudioUtil.spectro_gram(shift_aud, n_mels=64, n_fft=1024, hop_len=None)

aug_sgram = la.AudioUtil.spectro_augment(sgram, max_mask_pct=0.1, n_freq_masks=2, n_time_masks=2)

# Funció base
def plot_spectrogram(spectrogram, channel=0, title="Spectrograma", ylabel="Freqüència (Hz)", aspect='auto', cmap='rainbow'):
    
    channel_spectrogram = spectrogram[channel].numpy()
    
    fig, ax = plt.subplots()
    im = ax.imshow(channel_spectrogram, origin='lower', aspect=aspect, cmap=cmap)
    fig.colorbar(im, ax=ax)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Temps (frames)")
    plt.show()

# Printa l'espectrograma, només un canal
for channel in range(1):
    plot_spectrogram(sgram_orig, channel = channel, title="Espectrograma Original ")

for channel in range(1):
    plot_spectrogram(sgram, channel = channel, title="Espectrograma Preprocessat ")    

for channel in range(1):
    plot_spectrogram(aug_sgram, channel = channel, title="Espectrograma Augmentat ")

        