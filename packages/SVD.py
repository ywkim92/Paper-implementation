#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd

from packages.GramSchmidt import gram_schmidt

def svd_thin(data, ):
    dim = data.shape[1]
    rank = np.linalg.matrix_rank(data)
    relu = np.vectorize(lambda x: np.real(x) if np.real(x)>=0 else .0)
    
    #eval_u, evec_u = np.linalg.eig(data.dot(data.T), )
    eval_v, evec_v = np.linalg.eig(data.T.dot(data), )
    
    gs = gram_schmidt()
    #evec_u_gs = gs.fit_transform(evec_u)
    evec_v_gs = gs.fit_transform(evec_v)
        
    s = eval_v.copy()
    s = np.sqrt(relu(s))
    s1 = np.sort(s)[::-1]
    if dim > rank:
        s1[-(dim-rank):] = 0
    
    S = np.eye(dim)*s1
    
    v_idx = np.sqrt(relu(np.real(eval_v))).argsort()[-dim:][::-1]
    v = evec_v_gs[:, v_idx ]
    
    u = data.dot(v)/s1
    
    return u, S,  v
