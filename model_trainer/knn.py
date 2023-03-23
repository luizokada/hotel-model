from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn import model_selection
import matplotlib.pyplot as plt
import seaborn as sns
class KNNTrainer:
    def __init__(self, dataBase):
        knn = KNeighborsClassifier()
        scores = model_selection.cross_val_score(knn, dataBase.X, dataBase.Y, cv=10, scoring='accuracy')
        print("Acurácia média:", scores.mean())
        print("Desvio padrão:", scores.std())
        
    def printConfusionMatrix(self):
        # Configura as labels dos eixos x e y
        labels = [0, 1]
        sns.set(font_scale=1.4)
        sns.heatmap(self.matrix, annot=True, cmap='Blues', xticklabels=labels, yticklabels=labels)

        # Configura o título do gráfico e dos eixos
        plt.xlabel('Previsão')
        plt.ylabel('Valor Real')
        plt.title('Matriz de Confusão')

        # Exibe o gráfico
        plt.show()