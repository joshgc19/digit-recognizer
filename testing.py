import pickle
import matplotlib.pyplot as plt
from os.path import join

import numpy as np
import seaborn as sn

from classes.DigitsClassifier import DigitsClassifier


def load_features(save_path):
    """
    Function that load the list of features from disk to python
    :param save_path: path in which the features are saved
    :return: list of tuples containing observations and labels
    """
    with open(join(save_path, "features_dump.pkl"), "rb") as reader:
        return pickle.load(reader)


def show_confusion_matrix(confusion_matrix):
    sn.heatmap(confusion_matrix, annot=True)
    plt.title("Matriz de confusión para la clasificación de numeros del 0 al 9 escritos a mano")
    plt.xlabel("Predicciones")
    plt.ylabel("Objetivo")
    plt.show()


def main():
    """
    Procedure that retrieves features, trains the classifier and tests it
    """
    digitClassifier = DigitsClassifier()
    vectors = load_features("./features/")
    train_dataset, test_dataset = digitClassifier.split(vectors)
    digitClassifier.fit(train_dataset)
    score, confusion_matrix = digitClassifier.score(test_dataset)

    show_confusion_matrix(confusion_matrix)

    for i in range(confusion_matrix.shape[0]):
        print(f"Class {i}: accuracy: {confusion_matrix[i, i] / np.sum(confusion_matrix[i])}")

    print(score, confusion_matrix)


if __name__ == '__main__':
    main()
