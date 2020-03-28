import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from collections import defaultdict
from IPython import embed
from itertools import islice 
import os
import csv
import random

class Dataset:
    def __init__(self, inputFile, test_percent, n, m):
        self.data = pd.read_csv(inputFile, encoding='latin-1')
        self.n = n
        self.m = m
        self.test_percent = test_percent
        self.X_train = []
        self.y_train = []
        self.X_test = []
        self.y_test = []
        self.client_splits_X = defaultdict(list)
        self.client_splits_y = defaultdict(list)
        self.train_step_splits_X = defaultdict(lambda: defaultdict(list))
        self.train_step_splits_y = defaultdict(lambda: defaultdict(list))

    def train_test_split(self):
        y = list([1 if item == 'spam' else 0 for item in self.data["v1"]])
        X = list(self.data["v2"])
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=self.test_percent, random_state=42)
        return self.X_train, self.X_test, self.y_train, self.y_test

    def client_split(self):
        train_mapped = list(zip(self.X_train, self.y_train))
        random.shuffle(train_mapped)
        self.X_train, self.y_train = zip(*train_mapped)
        
        client_data_sizes = []
        while np.sum(client_data_sizes)!=len(self.X_train):
            rand_dist = np.random.dirichlet(np.ones(self.n))*len(self.X_train)
            client_data_sizes = [round(val) for val in rand_dist]
        
        temp = iter(train_mapped) 
        res = [list(islice(temp, 0, int(ele))) for ele in client_data_sizes]
        
        for i in range(self.n):
            self.client_splits_X[i] = [tuple[0] for tuple in res[i]]
            self.client_splits_y[i] = [tuple[1] for tuple in res[i]]

        return self.client_splits_X, self.client_splits_y

    def train_step_split(self):
        kf = KFold(n_splits=self.m, shuffle = True)
        for client, data in self.client_splits_X.items():
            i=0
            for train_index, test_index in kf.split(data):
                for idx in test_index:
                    self.train_step_splits_X[client][i].append(data[idx])
                    self.train_step_splits_y[client][i].append(self.client_splits_y[client][idx])
                i+=1
        return self.train_step_splits_X, self.train_step_splits_y

    def write_data_to_file(self):
        fields = ["v1", "v2"]
        for client in range(self.n):
            for roundno in range(self.m):
                filename = "data" + "/" + str(client)+"/"+str(roundno)+".csv"
                os.makedirs(os.path.dirname(filename), exist_ok=True)
                with open(filename, 'w', encoding="utf-8", newline='') as csvfile:  
                    csvwriter = csv.writer(csvfile) 
                    csvwriter.writerow(fields)
                    for i in range(len(self.train_step_splits_X[client][roundno])):
                        csvwriter.writerow([self.train_step_splits_X[client][roundno][i], self.train_step_splits_y[client][roundno][i]])

    def get_round_data(self, roundNo):
        filename = "data" + "/" + str(roundno) + ".csv"
        data = pd.read_csv(filename, encoding='latin-1')
        return data["v1"], data["v2"]
    
if __name__ == "__main__":
    d = Dataset("../data/spam.csv", 0.2, 5, 3)
    d.train_test_split()
    d.client_split()
    d.train_step_split()
    d.write_data_to_file()     