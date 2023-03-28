from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn import model_selection
import matplotlib.pyplot as plt
import seaborn as sns
class KNNTrainer:
    def __init__(self, dataBase,n_neighbors, weights):
        self.knn = KNeighborsClassifier(weights=weights,n_neighbors=n_neighbors)
        self.scores = model_selection.cross_val_score(self.knn, dataBase.X, dataBase.Y, cv=10, scoring='accuracy')
        
        