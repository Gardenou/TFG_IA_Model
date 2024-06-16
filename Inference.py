###############################################
# Usem el model per fer inferència
###############################################

import Model as mo
import TrainingData as td
import torch
import calculateMetrics as cm

def inference(model, val_dl):
    correct_prediction = 0
    total_prediction = 0
    all_preds = []
    all_labels = []

    # Traiem l'actualització dels gradients perquè no té una afectació important en 
    # la inferència i ens permet executar els experiments molt més ràpidament
    with torch.no_grad():
        for data in td.val_dl:
            # Separa audio i text i passa-ho al model
            inputs, labels = data[0].to(mo.device), data[1].to(mo.device)

            # Normalització bàsica dels arrays
            inputs_m, inputs_s = inputs.mean(), inputs.std()
            inputs = (inputs - inputs_m) / inputs_s

            # Prediccions
            outputs = model(inputs)

            # Agafa la classe amb el valor més alt
            _, prediction = torch.max(outputs, 1)

            # Mètode simple de recompte bàsic de prediccions correctes (substituit per CalculateMetrius)
            correct_prediction += (prediction == labels).sum().item()
            total_prediction += prediction.shape[0]

            all_preds.extend(prediction.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = correct_prediction / total_prediction
    print(f'Accuracy: {acc:.2f}, Total items: {total_prediction}')

    # Cridem a CalculateMetrics
    precision, recall, f1_score, accuracy, conf_matrix = cm.calculate_metrics(torch.tensor(all_preds), torch.tensor(all_labels), 10)
    
    # Mostrem els resultats
    print(f"Accuracy: {accuracy}")
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1 Score: {f1_score:.4f}')
    print('Confusion Matrix:')
    print(conf_matrix)