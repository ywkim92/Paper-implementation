import numpy as np
import pandas as pd

def train_test_split_rec(data_input, user_vars, traning_only, test_for_training, random_state=None):
    data = data_input.copy()
    data.reset_index(inplace=True)
    data.drop('index', axis=1, inplace=True)
    
    users = np.unique(data[user_vars])
    rng = np.random.default_rng(random_state)
    users_training = rng.choice(users, int(users.size*traning_only), replace=False)
    users_test = np.setdiff1d(users, users_training)
    
    training_idx = data[data[user_vars].isin(users_training)==True].index.tolist()
    for test in users_test:
        test_df = data[data[user_vars]==test]
        test_idx = test_df.index.tolist()
        n_samples = test_df.shape[0]
        if (n_samples*test_for_training) >= .5:
            rng = np.random.default_rng(random_state+n_samples)
            training_idx += rng.choice(test_idx, int(n_samples*test_for_training)+1, replace=False).tolist()
   
    train = data.loc[training_idx]
    test = data[data.index.isin(train.index)==False]
    
    return train, test
