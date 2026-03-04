from typing import List, Tuple
from sklearn.metrics import (
    confusion_matrix as sk_confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)


class Evaluator:
    def __init__(self) -> None:
        pass

    def evaluate(
        self, predictions: List[str], gold_labels: List[str]
    ) -> Tuple:        
        cm = sk_confusion_matrix(gold_labels, predictions)
        accuracy = accuracy_score(gold_labels, predictions)
        precision = precision_score(gold_labels, predictions, average="macro")
        recall = recall_score(gold_labels, predictions, average="macro")
        f1 = f1_score(gold_labels, predictions, average="macro")
        return cm, accuracy, precision, recall, f1