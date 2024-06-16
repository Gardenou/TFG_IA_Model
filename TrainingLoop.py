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
    # Loss Function, Optimizer i Scheduler
    criterion = nn.CrossEntropyLoss()
    #optimizer = torch.optim.RMSprop(model.parameters(), lr=0.001)
    #optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.001,
                                                    steps_per_epoch=int(len(train_dl)),
                                                    epochs=max_epochs,
                                                    anneal_strategy='linear')

    # Paràmetres per indicar al model quan ha d'aturar l'entrenament
    patience = 3
    best_acc = 0.0
    epochs_no_improve = 0

    # Guardem les dades en un fitxer .csv
    with open('xxxxx.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        # Capçalera
        writer.writerow(['Epoca', 'Perdua', 'Accuracy','Items'])

        # Bucle segons èpoques
        for epoch in range(max_epochs):
            running_loss = 0.0
            correct_prediction = 0
            total_prediction = 0
            
            for i, data in enumerate(train_dl):
                # Inputs d'entrades, separem text i audio
                inputs, labels = data[0].to(mo.device), data[1].to(mo.device)

                # Normalitzar
                inputs_m, inputs_s = inputs.mean(), inputs.std()
                inputs = (inputs - inputs_m) / inputs_s

                # posem el gradient de l'optimitzador a zero abans de començar el procés
                optimizer.zero_grad()

                # forward + backward + optimització
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                scheduler.step()
                
                running_loss += loss.item()

                # Agafem la classe amb millor puntuació
                _, prediction = torch.max(outputs, 1)

                # Mètode simple de recompte bàsic de prediccions correctes 
                correct_prediction += (prediction == labels).sum().item()
                total_prediction += prediction.shape[0]

                if i % 100 == 0:    # cada 100 batches imprimim els resultats per controlar com va
                    print('[%d, %5d] perdua: %.3f' % (epoch + 1, i + 1, running_loss / 10))
                    #writer.writerow([epoch + 1, i + 1, running_loss / 10])

            # Imprimim resum de mètriques bàsiques
            num_batches = len(train_dl)
            avg_loss = running_loss / num_batches
            acc = correct_prediction / total_prediction
            print(f'Epoca: {epoch}, Pèrdua: {avg_loss:.2f}, Accuracy: {acc:.2f}, Quantitat arxius processats: {total_prediction}')
            writer.writerow([epoch + 1, avg_loss, acc, total_prediction])

            # Mirem si la precisió és major o menor que a l'època anterior
            if acc > best_acc:
                best_acc = acc
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            # Ho comparem amb la paciència que hem determinat
            if epochs_no_improve >= patience:
                print(f'Convergència en època {epoch - 3}')
                break

    print('Fi de l`entrenament')
