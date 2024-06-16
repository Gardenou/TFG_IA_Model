####################################
# Classe per carregar un arxiu d'audio i aplicar-hi les transformacions
# adequades (convertir a estereo, fer que totes les pistes tinguin la
# mateixa longitud, augment de l'espectrograma, etc.)
#####################################

import random
import torch
from torchaudio import transforms, load

class AudioUtil():
  # Carrega un fitxer d'audio i n'extreu el tensor (sig) i el mostreig (sr)
  @staticmethod
  def open(audio_file):
    sig, sr = load(audio_file)
    return (sig, sr)

  # Convertir a dos canals
  @staticmethod
  def rechannel(aud, new_channel):
    sig, sr = aud

    if (sig.shape[0] == new_channel):
      return aud

    if (new_channel == 1):
      resig = sig[:1, :]
    else:
      # Convertir de mono a stereo replicant el canal que tenim
      resig = torch.cat([sig, sig])

    return ((resig, sr))

  # Resamplejar al valor donat
  @staticmethod
  def resample(aud, newsr):
    sig, sr = aud

    if (sr == newsr):     
      return aud

    num_channels = sig.shape[0]
    # Resampleja un canal
    resig = transforms.Resample(sr, newsr)(sig[:1, :])
    if (num_channels > 1):
      # Resample el segon canal
      retwo = transforms.Resample(sr, newsr)(sig[1:, :])
      # Uneix de nou els dos canals
      resig = torch.cat([resig, retwo])

    return ((resig, newsr))

  # Trunquem la pista o afegim zero-padding
  @staticmethod
  def pad_trunc(aud, max_ms):
    sig, sr = aud
    num_rows, sig_len = sig.shape
    max_len = sr // 1000 * max_ms

    if (sig_len > max_len):
      sig = sig[:, :max_len]

    elif (sig_len < max_len):
  
      pad_begin_len = random.randint(0, max_len - sig_len)
      pad_end_len = max_len - sig_len - pad_begin_len

      # Padding
      pad_begin = torch.zeros((num_rows, pad_begin_len))
      pad_end = torch.zeros((num_rows, pad_end_len))

      sig = torch.cat((pad_begin, sig, pad_end), 1)

    return (sig, sr)

  # Mètode TimeShift (desplaçament temporal en l'eix x)
  @staticmethod
  def time_shift(aud, shift_limit):
    sig, sr = aud
    _, sig_len = sig.shape
    shift_amt = int(random.random() * shift_limit * sig_len)
    return (sig.roll(shift_amt), sr)


  # ----------------------------
  # Implementació de la tècnica SpecAugment explicada en la memòria
  # Triem el numero d'emmascaraments i generem les línies horitzaontals
  # i verticals.
  # ----------------------------
  @staticmethod
  def spectro_augment(spec, max_mask_pct=0.1, n_freq_masks=2, n_time_masks=2):
    _, n_mels, n_steps = spec.shape
    mask_value = spec.mean()
    aug_spec = spec

    freq_mask_param = max_mask_pct * n_mels
    for _ in range(n_freq_masks):
      aug_spec = transforms.FrequencyMasking(freq_mask_param)(aug_spec, mask_value)

    time_mask_param = max_mask_pct * n_steps
    for _ in range(n_time_masks):
      aug_spec = transforms.TimeMasking(time_mask_param)(aug_spec, mask_value)

    return aug_spec


  # Genera l'espectrograma
  @staticmethod
  def spectro_gram(aud, n_mels=64, n_fft=1024, hop_len=None):
    sig, sr = aud
    # Nivell màxim de decibels segons conjunt
    top_db = 80
    # Funció espectrograma de Mel, passem els paràmetres de finestra, nombre de frequències i hop(no utilitzat aqui)
    spec = transforms.MelSpectrogram(sr, n_fft=n_fft, hop_length=hop_len, n_mels=n_mels)(sig)

    # Escala logarítmica per passa-ho a decibels
    spec = transforms.AmplitudeToDB(top_db=top_db)(spec)
    return (spec)
