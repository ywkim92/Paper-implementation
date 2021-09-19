import numpy as np
import pandas as pd
from itertools import product
from sklearn.model_selection import KFold, StratifiedKFold
import sklearn.metrics as metrics
import copy

class grid_cv:
    def __init__(self, estimator, param_grid, scoring, scoring_average = 'binary', cv = 5, ):
        self.estimator = estimator
        self.param_grid = param_grid
        self.scoring = scoring
        self.scoring_average = scoring_average
        self.cv = cv
        
    def __repr__(self):
        return 'grid_cv(estimator={},\n        param_grid={},\n        cv={}, scoring={})'.format(self.estimator, self.param_grid, self.cv, self.scoring)
    
    def fit(self, X_train, y_train):
        
        self.result = dict()
        
        candidates = list(product(*self.param_grid.values()))
        
        for cand_idx in range(len(candidates)):
            kf = StratifiedKFold(n_splits = self.cv)
            kf_ = kf.split(X_train, y_train)
            scores = []

            for t, v in kf_:
                X_tr = X_train.iloc[t]
                X_va = X_train.iloc[v]
                y_tr = y_train.iloc[t]
                y_va = y_train.iloc[v]
                
                model = copy.deepcopy(self.estimator)
                
                for idx, key in enumerate(self.param_grid.keys()):
                    setattr(model, key, candidates[cand_idx][idx])
                
                model.fit(X_tr, y_tr)

                Pred = model.predict(X_va)
                
                if self.scoring_average == 'binary':
                    score = getattr(metrics, self.scoring)((y_va), Pred, )
                    #print(score)
                else:
                    score = getattr(metrics, self.scoring)((y_va), Pred, average = self.scoring_average)
                    #print(score)
                scores.append(score)

            self.result[candidates[cand_idx]] = np.mean(scores)
        
        self.best_params_ = dict(zip(self.param_grid.keys(),  max(self.result, key = lambda x: self.result[x])))
        self.best_score_ = max(self.result.values())
        self.best_estimator_ = copy.deepcopy(self.estimator)
        for key, value in self.best_params_.items():
            setattr(self.best_estimator_, key, value)
        
        return self
    
    def predict(self, X_train, y_train, X_test):
        self.best_estimator_.fit(X_train, y_train)
        result = self.best_estimator_.predict(X_test)
        return result
    
    def predict_proba(self, X_train, y_train, X_test):
        self.best_estimator_.fit(X_train, y_train)
        try:
            result = self.best_estimator_.predict_proba(X_test)
            return result
        except: 
            raise AttributeError("'{}' object has no attribute 'predict_proba'".format(self.best_estimator_)) from None