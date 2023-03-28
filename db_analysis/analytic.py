import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
import seaborn as sns
import matplotlib.pyplot as plt

class DbAnalyser:
    def __init__(self, db:pd.DataFrame):
        self.db = db
        db.info()
        values={}
        for collum in db.columns:
            values[collum] = db[collum].value_counts().shape[0]
        print("---------------------------------------------------------")
        print("Quantidade de valores únicos por coluna:")
        print(pd.DataFrame(values, index=['values count']).transpose())
        #COMO NAO USA BOOKING_ID A COLUNA DEVE SER REMOVIDA
        db = db.drop('Booking_ID',axis=1)
        db['booking_status'].value_counts().plot(kind='bar', title='booking_status').set_xticklabels(['Not_Canceled','Canceled'],rotation=0)
        plt.show()

        
        columns_to_exclude = ['avg_price_per_room','lead_time','arrival_date','booking_status']

        # Calcula o numero de colunas da figura
        num_cols = len(db.columns) - len(columns_to_exclude)
        num_rows = int(num_cols / 2) + (num_cols % 2)

        # Faz as figura das colunas com dados discretos

        for j in range(2):
            fig, axs = plt.subplots(4, 2, figsize=(10, 4*num_rows))
            fig.subplots_adjust(hspace=0.8)
            init = j * 8
            fim = (j + 1) * 8
            plot_num = 0
            for i, column in enumerate(db.columns[init:fim]):
                if column in columns_to_exclude:
                    continue
                row = plot_num // 2
                col = plot_num % 2
                sns.countplot(x=column, hue='booking_status', data=db, ax=axs[row, col])
                axs[row, col].set_title(column)
                axs[row, col].set_xticklabels(axs[row, col].get_xticklabels(), rotation=0)
                plot_num += 1
            plt.show()
        #Valores continuos
        sns.scatterplot(data = db, x = 'no_of_adults', y = 'avg_price_per_room',  hue  = 'booking_status')
        plt.show()
        sns.scatterplot(data = db, y = 'avg_price_per_room', x = 'lead_time',  hue  = 'booking_status')
        plt.show()
        sns.scatterplot(data = db, y = 'avg_price_per_room', x = 'arrival_date',  hue  = 'booking_status')
        plt.show()


        
        
        