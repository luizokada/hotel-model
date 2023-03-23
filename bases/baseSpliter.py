
from sklearn.model_selection import train_test_split
class BaseSpliter:
    def __init__(self, base):
        X = base.iloc[:,:17]
        Y = base.iloc[:,17]
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(X, Y, test_size=0.3, random_state = 0)
        