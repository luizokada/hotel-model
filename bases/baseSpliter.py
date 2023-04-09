
from sklearn.model_selection import train_test_split
class BaseSpliter:
    def __init__(self, base):
        self.X = base.iloc[:,:17]
        self.Y = base.iloc[:,17]