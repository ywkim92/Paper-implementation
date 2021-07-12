#!/usr/bin/env python
# coding: utf-8


import numpy as np
import pandas as pd

class gram_schmidt:
    '''def __init__(self, X):
        self.X = X.astype(float)'''
        
    def _proj(self, u, v):
        return (np.vdot(v, u) / np.vdot(u,u))*u
    
    def fit_transform(self, X, col_vec = True, normal = True):
        X = X.astype(float)
        if col_vec:
            mat = X.copy()
        else:
            mat = (X.T).copy()
        
        N = mat.shape[1]
        mat_orth = np.array([]).reshape(mat.shape[0], -1)
        for n in range(N):
            u = mat[:, n:n+1].copy()
            if n ==0:
                mat_orth = np.hstack((mat_orth,u))
            else:
                for i in range(n):
                    u -= self._proj(mat_orth[:, i:i+1], mat[:, n:n+1])
                mat_orth = np.hstack((mat_orth,u))
        
        if normal:
            result = mat_orth / np.linalg.norm(mat_orth, axis=0)
            if col_vec:
                return result
            else:
                return result.T
        else:
            if col_vec:
                return mat_orth
            else:
                return mat_orth.T

