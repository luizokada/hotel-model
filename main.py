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
        analytic = DbAnalyser(data_base)
        
    elif(arg == 'knn'):
        #Transforma a base de dados em uma base de dados para o KNN convertendo as variáveis categóricas em variáveis numéricas
        data_base_to_knn = ConvertData(data_base)

        #Divide a Base de Dados
        data_knn = BaseSpliter(data_base_to_knn.data)

        #treina o KNN
        number_of_neighbors = [1,3,5,7,9,11,13]
        f1_scores = [[],[]]
        type_of_weight = ['uniform', 'distance']
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
                
        plt.show()

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
        data_base_to_dt = ConvertData(data_base)
        data = BaseSpliter(data_base_to_dt.data)
        dt = DTTrainer(data)

        print("F1_Score médio:", dt.scores.mean())
        print("Desvio padrão:", dt.scores.std())
        print("Matriz de confusão:")
        dt.plot()
    else:
        print("ARGUMENTO INVÁLIDO!")

if __name__ == "__main__":
    main()