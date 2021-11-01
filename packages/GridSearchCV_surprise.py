import numpy as np
import pandas as pd
from itertools import product
from surprise.model_selection import split
from surprise import Reader, Dataset
from sklearn.metrics import mean_absolute_error, mean_squared_error, ndcg_score
import copy
        
def rec_predict(model, testset, scoring='mae', ndcg_k = 3, fill_unrated = 1.):
    df = pd.DataFrame(model.test(testset)).drop('details', axis=1)
    df1 = df[df['r_ui']!=fill_unrated].copy()
    
    if scoring == 'mae':
        true = df['r_ui']
        pred = df['est']
        return mean_absolute_error(true, pred)

    elif scoring == 'mae_without_unrated':
        true = df1['r_ui']
        pred = df1['est']
        return mean_absolute_error(true, pred)
    
    elif scoring == 'rmse':
        true = df['r_ui']
        pred = df['est']
        return mean_squared_error(true, pred, squared=False)
    
    elif scoring == 'rmse_without_unrated':
        true = df1['r_ui']
        pred = df1['est']
        return mean_squared_error(true, pred, squared=False)
    
    elif scoring == 'ndcg':
        result = []
        user_ids = np.unique(df['uid'])
        df_ndcg = df.groupby(['uid','iid']).first()
        
        for u in user_ids:
            df_ndcg1 = df_ndcg.loc[u].copy()
            true = df_ndcg1[['r_ui']].T
            pred = df_ndcg1[['est']].T
            score = ndcg_score(true, pred, k=ndcg_k)
            result.append(score)
        return np.mean(result)
    else:
        raise ValueError('Not supported.') from None
        
        
def trainset_to_df(trainset, column_names = None):
    array_ = np.array([]).reshape(-1, 3)
    for ky in trainset.ur.keys():
        l = len(trainset.ur[ky])
        array = np.hstack((np.array([ky]*l).reshape(-1,1), np.array(trainset.ur[ky])))
        array_ = np.vstack((array_, array))
        
    if column_names is None:
        column_names = ['users', 'items', 'ratings']
    df = pd.DataFrame(array_, columns=column_names)
    df.iloc[:, 0] = df.iloc[:, 0].apply(lambda x: trainset.to_raw_uid(x))
    df.iloc[:, 1] = df.iloc[:, 1].apply(lambda x: trainset.to_raw_iid(x))
    
    return df

def cross_validate_surp(estimator, data, cv, reader, scoring, ndcg_k = 3, include_unrated=False, fill_unrated=1.):
    score=[]
    for trainset, testset in cv.split(data):
        if include_unrated:
            estimator.fit(trainset)
        else:
            trainset_df = trainset_to_df(trainset, )
            trainset_df = trainset_df[trainset_df.iloc[:, -1]!=fill_unrated]
            trainset_ = Dataset.load_from_df(trainset_df, reader)
            trainset_ = trainset_.build_full_trainset()
            estimator.fit(trainset_)
            
        score.append(rec_predict(estimator, testset, scoring=scoring, ndcg_k = ndcg_k, fill_unrated=fill_unrated))
    return score

class grid_cv_surp:
    ''''scoring' parameter support 'mae', 'rmse', 'ndcg' only.
    '''
    def __init__(self, estimator, param_grid, scoring='ndcg', ndcg_k = 3, cv = 5, random_state=None, refit = True):
        self.estimator = estimator
        self.param_grid = param_grid
        self.scoring = scoring
        self.ndcg_k = ndcg_k
        self.cv = cv
        self.random_state = random_state
        self.refit = refit
        
    def __repr__(self):
        return 'grid_cv(estimator={},\n        param_grid={},\n        cv={}, scoring={})'.format(self.estimator, self.param_grid, self.cv, self.scoring)
    
    def fit(self, X_train, include_unrated=True, fill_unrated=1.):
        
        self.result = dict()
        candidates = list( product(*self.param_grid.values()) )
        reader = Reader(rating_scale= X_train.reader.rating_scale)
        
        for cand_idx in range(len(candidates)):
            if type(self.cv) == int:
                kf_ = split.KFold(n_splits= self.cv, random_state=self.random_state, shuffle=True,)
            else:
                kf_ = self.cv
            scores = []

            for t, v in kf_.split(X_train):
                model = copy.deepcopy(self.estimator)
                
                for idx, key in enumerate(self.param_grid.keys()):
                    setattr(model, key, candidates[cand_idx][idx])

                if include_unrated:
                    model.fit(t)
                else:
                    trainset_df = trainset_to_df(t, )
                    trainset_df = trainset_df[trainset_df.iloc[:, -1]!=fill_unrated]
                    trainset_ = Dataset.load_from_df(trainset_df, reader)
                    trainset_ = trainset_.build_full_trainset()
                    model.fit(trainset_)
                
                if self.scoring == 'ndcg':
                    score = rec_predict(model, v, scoring=self.scoring, ndcg_k = self.ndcg_k, fill_unrated = fill_unrated)
                else:
                    score = -rec_predict(model, v, scoring=self.scoring, ndcg_k = self.ndcg_k, fill_unrated = fill_unrated)
                
                scores.append(score)

            self.result[candidates[cand_idx]] = np.mean(scores)
        
        self.best_params_ = dict(zip(self.param_grid.keys(),  max(self.result, key = lambda x: self.result[x])))
        self.best_score_ = max(self.result.values())
        self.best_estimator_ = copy.deepcopy(self.estimator)
        for key, value in self.best_params_.items():
            setattr(self.best_estimator_, key, value)
        
        if self.refit:
            if include_unrated:
                self.best_estimator_.fit(X_train.build_full_trainset())
            else:
                X_train_df = X_train.df
                X_train_df = X_train_df[X_train_df.iloc[:, -1]!=fill_unrated]
                X_train_ = Dataset.load_from_df(X_train_df, reader)
                X_train_ = X_train_.build_full_trainset()
                self.best_estimator_.fit(X_train_)
        return self
    
    def test(self, X_test):
        #self.best_estimator_.fit(X_train, y_train)
        result = self.best_estimator_.test(X_test)
        return result