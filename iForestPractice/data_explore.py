#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   data_explore.py
@Time    :   2019/09/11 09:38:13
@Author  :   fanmin 
'''

import os, random 
import pandas as pd
from sklearn import preprocessing 
from sklearn.model_selection import train_test_split


def load_data():
    current_path = os.getcwd()
    print(current_path)
    filename = "/credit_card.csv"
    data = pd.read_csv(current_path+filename)

    # data.head()
    # data.describe()
    # data.columns
    print("anomoly sample nums:", data['Class'].sum())
    # distinctCounter = data.apply(lambda x: len(x.unique()))

    dataX = data.copy().drop(['Class'],axis=1)
    dataY = data['Class'].copy()

    # featuresToScale = dataX.drop(['Time', 'Amount'],axis=1).columns

    olist = [i for i in range(1, 29)]
    slist = random.sample(olist, 28)
    featuresToScale = ['V'+str(i) for i in slist]

    # sX = preprocessing.StandardScaler(copy=True)
    # res = pd.DataFrame(sX.fit_transform(dataX[featuresToScale]), columns=featuresToScale)
    res = pd.DataFrame(dataX[featuresToScale], columns=featuresToScale)
    
    X_train, X_test, y_train, y_test = train_test_split(res, dataY, \
                                        test_size=0.1, \
                                        random_state=2019, stratify=dataY)
    
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":

    X_train, X_test, y_train, y_test = load_data()
    print(type(X_test))
    print(X_test.columns)


