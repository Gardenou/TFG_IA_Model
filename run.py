#################################################
# Executem el model per fer inferència
#################################################

import Model as mo
import Inference as inf
import TrainingData as td
from torch import load

# Carreguem els pesos d'algun model previament entrenat
mo.myModel.load_state_dict(load('C:\\Users\\denou\\PycharmProjects\\pythonProject\\venv\\audio_classifier_weights_ESC50_32m_512w_4s.pth'))
mo.myModel.eval()  # Mode avaluacio

# Cridem la classe inferencia
inf.inference(mo.myModel, td.val_dl)