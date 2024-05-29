#################################################
# Executem el model per fer inferència
#################################################

import Model as mo
import Inference as inf
import TrainingData as td
from torch import load


# Carreguem els pesos del model previament entrenat
mo.myModel.load_state_dict(load('C:\\Users\\denou\\PycharmProjects\\pythonProject\\venv\\audio_classifier_weights.pth'))
mo.myModel.eval()  # Mode evaluació

# Cridem la classe inference
inf.inference(mo.myModel, td.val_dl)