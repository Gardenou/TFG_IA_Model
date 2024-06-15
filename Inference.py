###############################################
# Usem el model per fer inferència
###############################################

import Model as mo
import TrainingData as td
import calculateMetrics as cm
import torch

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

            all_preds.extend(prediction.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            # Count of predictions that matched the target label
            correct_prediction += (prediction == labels).sum().item()
            total_prediction += prediction.shape[0]

    acc = correct_prediction / total_prediction
    print(f'Precisió: {acc:.2f}, Total items: {total_prediction}')

    precision, recall, f1_score, conf_matrix = cm.calculate_metrics(torch.tensor(all_preds), torch.tensor(all_labels), 10)

    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1 Score: {f1_score:.4f}')
    print('Confusion Matrix:')
    print(conf_matrix.numpy())