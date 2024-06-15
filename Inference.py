###############################################
# Usem el model per fer inferència
###############################################

import Model as mo
import TrainingData as td
import torch

def inference(model, val_dl):
    correct_prediction = 0
    total_prediction = 0

    # Disable gradient updates
    with torch.no_grad():
        for data in td.val_dl:
            # Get the input features and target labels, and put them on the GPU
            inputs, labels = data[0].to(mo.device), data[1].to(mo.device)

            # Normalize the inputs
            inputs_m, inputs_s = inputs.mean(), inputs.std()
            inputs = (inputs - inputs_m) / inputs_s

            # Get current predictions
            outputs = model(inputs)

            # Get the predicted class with the highest score
            _, prediction = torch.max(outputs, 1)

            # Count of predictions that matched the target label
            correct_prediction += (prediction == labels).sum().item()
            total_prediction += prediction.shape[0]

    acc = correct_prediction / total_prediction
    print(f'Precisió: {acc:.2f}, Total items: {total_prediction}')
