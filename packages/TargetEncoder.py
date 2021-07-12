#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd

class targetencoder:
    def __init__(self, handle_missing='values', handle_unknown='values'):
        self.handle_missing = handle_missing
        self.handle_unknown = handle_unknown
        
    def fit_transform(self, X, y):
        self.mapping = dict()
        if type(X) == np.ndarray:
            if len(X.shape)<2:
                X = X.reshape(-1, 1)
            for col in range(X.shape[1]):
                dic = dict()
                uniq, counts = np.unique(X[:, col], return_counts=True)
                for u in uniq:
                    array = X[:, col].flatten()
                    idx = np.where(array==u)[0]
                    dic[u] = (y[idx]).mean()
                #values, counts = np.unique(array)
                values_num = np.array([dic[i] for i in uniq])
                imputer = (values_num * counts).sum()/counts.sum()
                dic['missing'] = imputer
                dic['unknown'] = imputer
                self.mapping[col] = dic
        
            X1 = X.copy()
            for col in range(X1.shape[1]):
                X1[:,col:col+1] = np.vectorize(lambda x: self.mapping[col][x])(X1[:,col:col+1])
               
            return X1
        
        else:
            if len(X.shape)<2:
                X = pd.DataFrame(X)
            for col in X.columns:
                dic = dict()
                uniq, counts = np.unique(X[col], return_counts=True)
                for u in uniq:
                    array = X[col].values
                    idx = np.where(array==u)[0]
                    dic[u] = (y.values[idx]).mean()
                    
                #values, counts = np.unique(array, return_counts=True)
                values_num = np.array([dic[i] for i in uniq])
                imputer = (values_num * counts).sum()/counts.sum()
                dic['missing'] = imputer
                dic['unknown'] = imputer
                self.mapping[col] = dic
            
            X1 = X.copy()
            for col in X1.columns:
                X1[col] = X1[col].apply(lambda x: self.mapping[col][x])
            
            return X1
    
    def _apply(self, col, x):
        if x in self.mapping[col]:
            return self.mapping[col][x]
        elif x!=x:
            if self.handle_missing == 'values':
                return self.mapping[col]['missing']
            else:
                return x
        else:
            if self.handle_unknown == 'values':
                return self.mapping[col]['unknown']
            else:
                return np.nan
        
    def transform(self, X,):
        if type(X) == np.ndarray:
            if len(X.shape)<2:
                X = X.reshape(-1, 1)
            X1 = X.copy()
            
            for col in range(X1.shape[1]):
                X1[:,col:col+1] = np.vectorize(lambda x: self._apply(col, x) )(X1[:,col:col+1])
               
            return X1
        else:
            if len(X.shape)<2:
                X = pd.DataFrame(X)
            X1 = X.copy()
            for col in X1.columns:
                X1[col] = X1[col].apply(lambda x: self._apply(col, x))
            
            return X1
