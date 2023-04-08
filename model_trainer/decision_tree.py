from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn import model_selection
from plot.plot_confusion_matrix import plot_confusion_matrix

class DTTrainer:
    def __init__(self, dataBase) -> None:
        print("Training DT...")
        self.svm = DecisionTreeClassifier()
        self.cv = StratifiedKFold(n_splits=10)
        self.scores = model_selection.cross_val_score(self.svm, dataBase.X, dataBase.Y, cv=self.cv, scoring='f1')
        self.y_pred = cross_val_predict(self.svm, dataBase.X, dataBase.Y ,cv=self.cv)
        self.conf_mat = confusion_matrix(dataBase.Y, self.y_pred)
    
    def plot(self):
        plot_confusion_matrix(self.conf_mat, 'Confusion Matrix DT')