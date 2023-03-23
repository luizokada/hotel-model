from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
class KNNTrainer:
    def __init__(self, dataBase,n_neighbors=5):
        knn = KNeighborsClassifier()
        knn.fit(dataBase.X_train, dataBase.Y_train)
        y_pred = knn.predict(dataBase.X_test)

        accuracy_knn = accuracy_score(dataBase.Y_test, y_pred)
        f1_knn = f1_score(dataBase.Y_test, y_pred)
        self.matrix = confusion_matrix(dataBase.Y_test, y_pred)
        self.printConfusionMatrix()
        print(accuracy_knn, f1_knn)
        
    def printConfusionMatrix(self):
        # Configura as labels dos eixos x e y
        labels = ['0', '1']
        sns.set(font_scale=1.4)
        sns.heatmap(self.matrix, annot=True, cmap='Blues', xticklabels=labels, yticklabels=labels)

        # Configura o título do gráfico e dos eixos
        plt.xlabel('Previsão')
        plt.ylabel('Valor Real')
        plt.title('Matriz de Confusão')

        # Exibe o gráfico
        plt.show()