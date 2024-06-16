from torch import zeros

def calculate_metrics(all_preds, all_labels, num_classes):
    confusion_matrix = zeros(num_classes, num_classes)
    for t, p in zip(all_labels, all_preds):
        confusion_matrix[t, p] += 1

    TP = confusion_matrix.diag()
    FP = confusion_matrix.sum(0) - TP
    FN = confusion_matrix.sum(1) - TP
    TN = confusion_matrix.sum() - (FP + FN + TP)

    accuracy = (TP+TN) / (TP+TN+FP+FN)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1_score = 2 * (precision * recall) / (precision + recall)

    accuracy[accuracy != accuracy] = 0  # Set NaN to 0
    precision[precision != precision] = 0  # Set NaN to 0
    recall[recall != recall] = 0  # Set NaN to 0
    f1_score[f1_score != f1_score] = 0  # Set NaN to 0

    return precision.mean().item(), recall.mean().item(), f1_score.mean().item(), accuracy.mean().item(), confusion_matrix
