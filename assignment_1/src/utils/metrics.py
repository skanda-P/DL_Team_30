# Metrics utilities

import numpy as np

def confusion_matrix(y_true, y_pred, num_classes):
    """
    Generates a confusion matrix from scratch.
    Rows represent True labels, Columns represent Predicted labels.
    """
    matrix = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        matrix[int(t)][int(p)] += 1
    return matrix

def classification_metrics(y_true, y_pred, num_classes):
    """
    Calculates overall accuracy, along with class-wise, macro, and micro 
    accuracy, precision, recall, and f-measure.
    """
    cm = confusion_matrix(y_true, y_pred, num_classes)
    total_samples = len(y_true)
    
    # Arrays to hold individual class TP, FP, FN, TN
    tp = np.zeros(num_classes)
    fp = np.zeros(num_classes)
    fn = np.zeros(num_classes)
    tn = np.zeros(num_classes)
    
    # Arrays to hold class-wise metrics
    class_accuracies = np.zeros(num_classes)
    precisions = np.zeros(num_classes)
    recalls = np.zeros(num_classes)
    f_measures = np.zeros(num_classes)
    
    for i in range(num_classes):
        tp[i] = cm[i, i]
        fp[i] = np.sum(cm[:, i]) - tp[i]
        fn[i] = np.sum(cm[i, :]) - tp[i]
        
        # True Negatives: Total samples minus everything that involves class i
        tn[i] = total_samples - (tp[i] + fp[i] + fn[i])

        # classwise accuracies
        class_accuracies[i] = (tp[i] + tn[i]) / total_samples if total_samples > 0 else 0.0

        #classwise precision
        precisions[i] = tp[i] / (tp[i] + fp[i]) if (tp[i] + fp[i]) > 0 else 0.0

        #classwise recall
        recalls[i] = tp[i] / (tp[i] + fn[i]) if (tp[i] + fn[i]) > 0 else 0.0
        
        #classwise F-measure
        f_measures[i] = 2 * (precisions[i] * recalls[i]) / (precisions[i] + recalls[i])

    overall_accuracy = np.trace(cm) / np.sum(cm)

    # Macro Metrics (Unweighted average of class-wise metrics)
    macro_precision = np.mean(precisions)
    macro_recall = np.mean(recalls)
    macro_f_measure = np.mean(f_measures)
    macro_accuracy = np.mean(class_accuracies)
    
    # Micro Metrics (Global aggregates)
    total_tp = np.sum(tp)
    total_fp = np.sum(fp)
    total_fn = np.sum(fn)
    
    micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    micro_f_measure = 2 * (micro_precision * micro_recall) / (micro_precision + micro_recall)
        
    return {
        "confusion_matrix": cm,
        "overall_accuracy": overall_accuracy,
        "class_accuracy": class_accuracies,
        "class_precision": precisions,
        "class_recall": recalls,
        "class_f_measure": f_measures,
        "macro_accuracy": macro_accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f_measure": macro_f_measure,
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "micro_f_measure": micro_f_measure
    }

def print_classification_report(metrics_dict):
    print("Confusion Matrix:")
    print(metrics_dict["confusion_matrix"])
    print(f"\nOverall Accuracy: {metrics_dict['overall_accuracy']:.4f}")
    
    for i in range(len(metrics_dict["class_precision"])):
        print(f"\n--- Class {i} ---")
        print(f"Accuracy:  {metrics_dict['class_accuracy'][i]:.4f}")
        print(f"Precision: {metrics_dict['class_precision'][i]:.4f}")
        print(f"Recall:    {metrics_dict['class_recall'][i]:.4f}")
        print(f"F-Measure: {metrics_dict['class_f_measure'][i]:.4f}")
        
    print("\n--- Macro Metrics (Average) ---")
    print(f"Macro Accuracy:  {metrics_dict['macro_accuracy']:.4f}")
    print(f"Macro Precision: {metrics_dict['macro_precision']:.4f}")
    print(f"Macro Recall:    {metrics_dict['macro_recall']:.4f}")
    print(f"Macro F-Measure: {metrics_dict['macro_f_measure']:.4f}")
    
    print("\n--- Micro Metrics (Global) ---")
    print(f"Micro Precision: {metrics_dict['micro_precision']:.4f}")
    print(f"Micro Recall:    {metrics_dict['micro_recall']:.4f}")
    print(f"Micro F-Measure: {metrics_dict['micro_f_measure']:.4f}")