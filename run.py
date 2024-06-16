#################################################
# Executem el model per fer inferència
#################################################

import Model as mo
import Inference as inf
import TrainingData as td
from torch import load,device

# Carreguem els pesos d'algun model previament entrenat
mo.myModel.load_state_dict(load('C:\\Users\\denou\\OneDrive\\Documents\\TFG_IA_Model\\weights\\audio_classifier_weights_64m_1024w_RMSprop.pth',map_location=device('cpu')))
mo.myModel.eval()  # Mode avaluacio

# Cridem la classe inferencia
inf.inference(mo.myModel, td.val_dl)