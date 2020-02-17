import pandas as pd
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from collections import defaultdict
class Dataset:
    def __init__(self, inputFile):
        self.data = pd.read_csv(inputFile, encoding='latin-1')  
        self.X_train = []
        self.y_train = []
        self.X_test = []
        self.y_test = []
        self.client_splits_X = defaultdict(list)
        self.client_splits_y = defaultdict(list)
        self.train_step_splits_X = defaultdict(lambda: defaultdict(list))
        self.train_step_splits_y = defaultdict(lambda: defaultdict(list))

    def train_test_split(self, test_percent):
        y = list(self.data["v1"])
        X = list(self.data["v2"])
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=test_percent, random_state=42)

    def client_split(self, n):
        kf = KFold(n_splits=n)
        i=0
        for train_index, test_index in kf.split(self.X_train):
            for idx in test_index:
                self.client_splits_X[i].append(self.X_train[idx])
                self.client_splits_y[i].append(self.y_train[idx])
            i+=1
        # print(len(self.client_splits_X[3]), len(self.client_splits_y[3]))

    def train_step_split(self, m):
        kf = KFold(n_splits=m)
        for client, data in self.client_splits_X.items():
            i=0
            for train_index, test_index in kf.split(data):
                for idx in test_index:
                    self.train_step_splits_X[client][i].append(data[idx])
                    self.train_step_splits_y[client][i].append(self.train_step_splits_y[client][idx])   
                i+=1
            

d = Dataset("../data/spam.csv")
d.train_test_split(0.2)
d.client_split(5)
d.train_step_split(3)