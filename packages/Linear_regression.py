#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
from scipy.stats import t, f, norm


class linear_regression:
    
    def fit(self, X, y):
        X_ = np.append(np.ones((X.shape[0], 1)), X, axis=1)
        Q = np.linalg.inv(X_.T.dot(X_))
        self.beta = Q.dot(X_.T).dot(y)
        
        n = X.shape[0]
        p = X.shape[1]
        self.n_samples = n
        self.n_features = p
        y_hat = X_.dot(self.beta)
        
        self.residual = y - y_hat
        sse = np.square(self.residual).sum()
        ssr = np.square(y_hat - y.mean()).sum()
        sst = np.square(y - y.mean()).sum()
        self.mse = sse/(n-p-1)
        
        self.r_sq = 1 - sse/sst
        self.adj_r = 1 - (sse/(n-p-1))/(sst/(n-1))
        
        self.f_value = (ssr/p) / (sse/(n-p-1))
        self.f_pvalue = 1 - f.cdf(self.f_value, dfn = p, dfd = n - p - 1)
        
        self.std_error = np.sqrt((sse/(n-p-1)) * Q.diagonal())
        self.t_values = self.beta / self.std_error
        self.t_pvalues = [2*(1 - t.cdf(abs(i), n-p-1)) for i in self.t_values]
        self.log_likelihood = np.log(norm.pdf(y, y_hat, self.residual.std(ddof=0))).sum()
        self.aic = -2 * self.log_likelihood + 2*self.beta.size
        self.bic = -2 * self.log_likelihood + self.beta.size * np.log(self.n_samples)
        
    def predict(self, X):
        X_ = np.append(np.ones((X.shape[0], 1)), X, axis=1)
        return X_.dot(self.beta)
    
    def summary(self, feature_names = None, round_decimals=3, alpha=0.05):
        res_min = self.residual.min()
        res_max = self.residual.max()
        res_1q = np.quantile(self.residual, .25)
        res_med = np.median(self.residual)
        res_3q = np.quantile(self.residual, .75)
        res = pd.DataFrame(np.array([res_min, res_1q, res_med, res_3q, res_max]).reshape(1,-1), columns = ['Residuals: Min', '1Q', 'Median', '3Q', 'Max'])
        
        ci_lower = np.round(self.beta - self.std_error * t.ppf(1-alpha/2, self.n_samples - self.n_features -1), round_decimals)
        ci_upper = np.round(self.beta + self.std_error * t.ppf(1-alpha/2, self.n_samples - self.n_features -1), round_decimals)
        if feature_names is None:
            feat_idx = ['x0(INTCP)'] + ['x{}'.format(i) for i in range(1, self.n_features+1)]
            coef = pd.DataFrame(np.array([self.beta, self.std_error, self.t_values, np.round(self.t_pvalues, round_decimals), ci_lower, ci_upper]).T, index = feat_idx, columns= 
                               ['coef.', 'std. error', 't_value', 'Pr(>|t|)', '{}% LWR'.format(int(100*(1-alpha))), '{}% UPR'.format(int(100*(1-alpha)))])
        else:
            feat_idx = np.insert(feature_names, 0, '(INTCP)')
            coef = pd.DataFrame(np.array([self.beta, self.std_error, self.t_values, np.round(self.t_pvalues, round_decimals), ci_lower, ci_upper]).T, index = feat_idx, columns= 
                               ['coef.', 'std. error', 't_value', 'Pr(>|t|)', '{}% LWR'.format(int(100*(1-alpha))), '{}% UPR'.format(int(100*(1-alpha)))])
        
        r2 = 'R-squared: {}, Adjusted R-squared: {}'.format(self.r_sq, self.adj_r)
        res_std_err = 'Residual standard error: {} on {} degrees of freedom'.format(self.mse, self.n_samples - self.n_features -1)
        F_stats = 'F-statistic: {} on {} and {} DF, p-value: {}'.format(self.f_value, self.n_features, self.n_samples - self.n_features -1, self.f_pvalue)
        criterions = 'Log-likelihood: {}, AIC: {}, BIC: {}'.format(self.log_likelihood, self.aic, self.bic)
        print(res, coef, res_std_err, r2, F_stats, criterions, sep='\n'+'='*70+'\n')
