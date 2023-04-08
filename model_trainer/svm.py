import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn import model_selection

class SVMTrainer:
    def __init__(self, dataBase) -> None:
        self.svm = SVC()
        self.cv = StratifiedKFold(n_splits=10)
        self.scores = model_selection.cross_val_score(self.svm, dataBase.X, dataBase.Y, cv=self.cv, scoring='f1')
        self.y_pred = cross_val_predict(self.svm, dataBase.X, dataBase.Y ,cv=self.cv)
        self.conf_mat = confusion_matrix(dataBase.Y, self.y_pred)
        class_names = np.array(['0', '1'])
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(self.conf_mat, annot=True, cmap='Blues', fmt='d', xticklabels=class_names, yticklabels=class_names, ax=ax)
        ax.set_xlabel('Predicted labels')
        ax.set_ylabel('True labels')
        ax.set_title('Confusion Matrix')
        plt.show()