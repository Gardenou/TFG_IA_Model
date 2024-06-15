############################################
# Entrenament del model
############################################

import TrainingLoop as tl
import Model as mo
import TrainingData as td
import torch

# Comprovem si tenim GPU disponible
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No s'ha torbat cap GPU al sistema")

# Entrenament
max_epochs = 10
tl.training(mo.myModel, td.train_dl, max_epochs)

# GUardem els pesos resultants de l'entrenament
#torch.save(mo.myModel.state_dict(), 'audio_classifier_weights_test_Scores.pth')
torch.save(mo.myModel.state_dict(), 'xxxxxx.pth')