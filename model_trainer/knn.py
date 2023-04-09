from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn import model_selection
from plot.plot_confusion_matrix import plot_confusion_matrix

class KNNTrainer:
    def __init__(self, dataBase, n_neighbors, weights):
        self.knn = KNeighborsClassifier(weights=weights,n_neighbors=n_neighbors)
        self.cv = StratifiedKFold(n_splits=10)
        self.scores = model_selection.cross_val_score(self.knn, dataBase.X, dataBase.Y, cv=self.cv, scoring='f1')
        self.y_pred = cross_val_predict(self.knn, dataBase.X, dataBase.Y ,cv=self.cv)
        self.conf_mat = confusion_matrix(dataBase.Y, self.y_pred)
        self.neighbors = n_neighbors
        self.weights = weights
    
    def plot(self):
        plot_confusion_matrix(self.conf_mat, 'Confusion Matrix n={} w={}'.format(self.neighbors,self.weights))