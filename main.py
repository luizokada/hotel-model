import pandas as pd
from convert_data.convertData import ConvertData 
from model_trainer.knn import KNNTrainer
from model_trainer.decision_tree import DTTrainer
from bases.baseSpliter import BaseSpliter
from db_analysis.analytic import DbAnalyser
import matplotlib.pyplot as plt
import sys

def main():
    
    arg = sys.argv[1]
    data_base = pd.read_csv('./db/HotelReservations.csv')

    if(arg == 'analytic'):
        #Faz a analise de dados
        DbAnalyser(data_base)
        
    elif(arg == 'knn'):
        print("KNN\n")
        #Transforma a base de dados em uma base de dados para o KNN convertendo as variáveis categóricas em variáveis numéricas
        db = ConvertData(data_base)

        #Divide a Base de Dados
        data_knn = BaseSpliter(db.data)

        #treina o KNN
        number_of_neighbors = [1,3,5,7,9,11,13]
        type_of_weight = ['uniform', 'distance']
        f1_scores = [[],[]]

        i=0
        for neighbors in number_of_neighbors:
            i=0
            for weight in type_of_weight:
                print("Número de vizinhos: ", neighbors)
                print("Tipo de peso: ", weight)
                knn = KNNTrainer(data_knn, neighbors, weight)
                f1_scores[i].append(knn.scores.mean())
                print("F1_Score médio:", knn.scores.mean())
                print("Desvio padrão:", knn.scores.std())
                print("Matriz de confusão:")
                print("---------------------------------------------------------")
                knn.plot()
                i+=1
        
        knn.plot()
        j = f1_scores[0]
        k = f1_scores[1]

        plt.plot(number_of_neighbors,j)
        plt.title('F1_Score médio W = uniform')
        plt.ylabel('F1_Score')
        plt.xlabel('N neighbors')
        plt.show()

        plt.plot(number_of_neighbors,k)
        plt.title('F1_Score médio W = Distance')
        plt.ylabel('F1_Score')
        plt.xlabel('N neighbors')
        plt.show()

                
        print("F1_Score médio uniform:", f1_scores[0])
        print("F1_Score médio distance:", f1_scores[1])
    elif (arg == 'dt'):
        print("Decision Tree\n")
        db = ConvertData(data_base)
        data = BaseSpliter(db.data)
    
        max_depths = [30, 40, 50, None]
        splitters = ['best', 'random']
        f1_scores = [[],[]]

        i=0
        for depth in max_depths:
            i=0
            for splitter in splitters:
                print("Profundidade máxima: ", depth)
                print("Splitter: ", splitter)
                dt = DTTrainer(data, depth, splitter)
                f1_scores[i].append(dt.scores.mean())
                print("F1_Score médio:", dt.scores.mean())
                print("Desvio padrão:", dt.scores.std())
                print("Matriz de confusão:")
                print("---------------------------------------------------------")
                dt.plot()
                i+=1

        dt.plot()
        j = f1_scores[0]
        k = f1_scores[1]

        plt.plot(max_depths,j)
        plt.title('F1_Score médio Splitter = best')
        plt.ylabel('F1_Score')
        plt.xlabel('Max Depths')
        plt.show()

        plt.plot(max_depths,k)
        plt.title('F1_Score médio Splitter = random')
        plt.ylabel('F1_Score')
        plt.xlabel('Max Depths')
        plt.show()

        print("F1_Score médio best:", f1_scores[0])
        print("F1_Score médio random:", f1_scores[1])
    else:
        print("ARGUMENTO INVÁLIDO!")

if __name__ == "__main__":
    main()