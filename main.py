import pandas as pd
from convert_data.convertToKNN import ConverterDataToKNN 
from model_trainer.knn import KNNTrainer
from bases.baseSpliter import BaseSpliter

data_base = pd.read_csv('./db/HotelReservations.csv')

data_base_to_knn = ConverterDataToKNN(data_base)

data = BaseSpliter(data_base_to_knn.data)

knn = KNNTrainer(data)

