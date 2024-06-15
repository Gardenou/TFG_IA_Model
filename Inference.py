###############################################
# Usem el model per fer inferència
###############################################

import Model as mo
import TrainingData as td
import torch
import torchmetrics

def inference(model, val_dl):
    correct_prediction = 0
    total_prediction = 0
    all_preds = []
    all_labels = []

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

            all_preds.extend(preds.cpu())
            all_labels.extend(labels.cpu())

            # Count of predictions that matched the target label
            correct_prediction += (prediction == labels).sum().item()
            total_prediction += prediction.shape[0]

    acc = correct_prediction / total_prediction
    print(f'Precisió: {acc:.2f}, Total items: {total_prediction}')

    all_preds = torch.tensor(all_preds)
    all_labels = torch.tensor(all_labels)

    precision_value = precision(all_preds, all_labels)
    recall_value = recall(all_preds, all_labels)
    f1_value = f1(all_preds, all_labels)
    conf_matrix = confusion_matrix(all_preds, all_labels)

    print(f'Precision: {precision_value:.4f}')
    print(f'Recall: {recall_value:.4f}')
    print(f'F1 Score: {f1_value:.4f}')
    print('Confusion Matrix:')
    print(conf_matrix.numpy())