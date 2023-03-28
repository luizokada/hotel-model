import pandas as pd
from convert_data.convertToKNN import ConverterDataToKNN 
from model_trainer.knn import KNNTrainer
from bases.baseSpliter import BaseSpliter
from db_analysis.analytic import DbAnalyser

data_base = pd.read_csv('./db/HotelReservations.csv')

data_base_to_knn = ConverterDataToKNN(data_base)

data = BaseSpliter(data_base_to_knn.data)



analytic = DbAnalyser(data_base)


number_of_neighbors = [1,3,5,7,9]
type_of_weight = ['uniform', 'distance']
for neighbors in number_of_neighbors:
    for weight in type_of_weight:
        print("Número de vizinhos: ", neighbors)
        print("Tipo de peso: ", weight)
        knn = KNNTrainer(data, neighbors, weight)
        print("Acurácia média:", knn.scores.mean())
        print("Desvio padrão:", knn.scores.std())
        print("Matriz de confusão:")
        print("---------------------------------------------------------")

