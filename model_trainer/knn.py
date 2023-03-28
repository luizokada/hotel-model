from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn import model_selection
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
class KNNTrainer:
    def __init__(self, dataBase,n_neighbors, weights):
        self.knn = KNeighborsClassifier(weights=weights,n_neighbors=n_neighbors)
        self.scores = model_selection.cross_val_score(self.knn, dataBase.X, dataBase.Y, cv=10, scoring='accuracy')
        self.y_pred = cross_val_predict(self.knn, dataBase.X, dataBase.Y ,cv=10)
        self.conf_mat = confusion_matrix(dataBase.Y, self.y_pred)
        class_names = np.array(['0', '1'])
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(self.conf_mat, annot=True, cmap='Blues', fmt='d', xticklabels=class_names, yticklabels=class_names, ax=ax)
        ax.set_xlabel('Predicted labels')
        ax.set_ylabel('True labels')
        ax.set_title('Confusion Matrix n={} w={}'.format(n_neighbors, weights))
        plt.show()