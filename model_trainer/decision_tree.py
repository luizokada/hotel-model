from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn import model_selection
from plot.plot_confusion_matrix import plot_confusion_matrix

class DTTrainer:
    def __init__(self, dataBase, max_depth, splitter) -> None:
        self.dt = DecisionTreeClassifier(splitter=splitter, max_depth=max_depth)
        self.cv = StratifiedKFold(n_splits=10)
        self.scores = model_selection.cross_val_score(self.dt, dataBase.X, dataBase.Y, cv=self.cv, scoring='f1')
        self.y_pred = cross_val_predict(self.dt, dataBase.X, dataBase.Y,cv=self.cv)
        self.conf_mat = confusion_matrix(dataBase.Y, self.y_pred)
        self.max_depth = max_depth
        self.splitter = splitter
    
    def plot(self):
        plot_confusion_matrix(self.conf_mat, 'Confusion Matrix max_depth={} splitter={}'.format(self.max_depth,self.splitter))