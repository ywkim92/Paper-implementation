import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_error, mean_squared_error, ndcg_score
def rec_predict(model, testset, scoring='mae', ndcg_k = 3):
    result = []
    if scoring == 'mae':
        for idx in range(testset.shape[0]):
            pred_value = model.predict(str(testset.iloc[idx, 0]), str(testset.iloc[idx, 1])).est
            result.append(pred_value)
        return mean_absolute_error(testset.iloc[:, -1].values, result)
    
    elif scoring == 'rmse':
        for idx in range(testset.shape[0]):
            pred_value = model.predict(str(testset.iloc[idx, 0]), str(testset.iloc[idx, 1])).est
            result.append(pred_value)
        return mean_squared_error(testset.iloc[:, -1].values, result, squared=False)
    
    elif scoring == 'ndcg':
        col_rating = testset.columns[-1]
        #print(testset.columns[:2])
        testset_ = testset.groupby(testset.columns[:2].tolist()).first()
        user_ids = np.unique(testset.iloc[:, 0])
        
        for u in user_ids:
            subset = testset_.loc[u, col_rating]
            true = subset.values.reshape(1, -1)
            pred = []
            items = subset.index
            for i in items:
                pred.append(model.predict(str(u), str(i)).est)
            pred = np.array(pred).reshape(1, -1)
            
            if pred.size==1:
                continue
                #true = np.append(true, 0).reshape(1, -1)
                #pred = np.append(pred, 0).reshape(1, -1)
            else:
                #print(true, pred)
                score = ndcg_score(true, pred, k = ndcg_k)
                result.append(score)
        return np.mean(result)
    else:
        raise ValueError('Not supported.') from None