###################################################
# Bucle d'entrenament del model
###################################################

import dataLoader as dl
import Model as mo
import TrainingData as td
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
import csv


def training(model, train_dl, max_epochs):
    # Loss Function, Optimizer and Scheduler
    criterion = nn.CrossEntropyLoss()
    #optimizer = torch.optim.SGD(model.parameters(), lr=0.04, momentum=0.9)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.001,
                                                    steps_per_epoch=int(len(train_dl)),
                                                    epochs=max_epochs,
                                                    anneal_strategy='linear')

    # Paràmetres per indicar al model quan ha d'aturar l'entrenament
    patience = 3
    best_acc = 0.0
    epochs_no_improve = 0

    # Guardem les dades en un fitxer .csv
    with open('training_log.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        # Capçalera
        writer.writerow(['Epoch', 'Batch', 'Loss'])

        # Bucle segons èpoques
        for epoch in range(max_epochs):
            running_loss = 0.0
            correct_prediction = 0
            total_prediction = 0

            # Repeat for each batch in the training set
            for i, data in enumerate(train_dl):
                # Get the input features and target labels, and put them on the GPU
                inputs, labels = data[0].to(mo.device), data[1].to(mo.device)

                # Normalize the inputs
                inputs_m, inputs_s = inputs.mean(), inputs.std()
                inputs = (inputs - inputs_m) / inputs_s

                # Zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                scheduler.step()

                # Keep stats for Loss and Accuracy
                running_loss += loss.item()

                # Get the predicted class with the highest score
                _, prediction = torch.max(outputs, 1)
                # Count of predictions that matched the target label
                correct_prediction += (prediction == labels).sum().item()
                total_prediction += prediction.shape[0]

                if i % 100 == 0:    # cada 100 batches imprimim els resultats per controlar com va
                    print('[%d, %5d] perdua: %.3f' % (epoch + 1, i + 1, running_loss / 10))
                    writer.writerow([epoch + 1, i + 1, running_loss / 10])

            # Imprimim resum de mètriques bàsiques
            num_batches = len(train_dl)
            avg_loss = running_loss / num_batches
            acc = correct_prediction / total_prediction
            print(f'Epoca: {epoch}, Pèrdua: {avg_loss:.2f}, Accuracy: {acc:.2f}, Quantitat arxius processats: {total_prediction}')
            writer.writerow([epoch + 1, 'summary', avg_loss, acc, total_prediction])

            # Mirem si la precisió és major o menor que a l'època anterior
            if acc > best_acc:
                best_acc = acc
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            # Ho comparem amb la paciència que hem determinat
            if epochs_no_improve >= patience:
                print(f'Convergència en època {epoch + 1}')
                break

    print('Fi de l`entrenament')
