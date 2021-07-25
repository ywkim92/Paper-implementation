#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from matplotlib import pyplot as plt
from scipy.special import ndtri


# In[3]:


def qqplot(data, color = '#1f77b4', figsize = (10,10), axis_scaling = 'equal', return_plot = True, return_xy = False):
    
    def _filliben(n):
        result = np.zeros(n)
        result[-1] = 0.5**(1./n)
        result[0] = 1 - 0.5**(1./n)
        for i in range(2, n):
            result[i-1] = (i - 0.3175) / (n + 0.365)
        return result
    
    array = np.sort(data)
    nd_ppf = ndtri(_filliben(array.size))
    
    if return_plot:
        fig, ax = plt.subplots(1,1, figsize=figsize)
        ax.set_aspect(axis_scaling)
        ax.scatter(nd_ppf, array, s=5, color = color, label='data')
        ax.axline((0,0), (1,1), color='#ff7f0e', label='line $y=x$')

        ax.axline((0,0), (0,1), linestyle=':', linewidth=1, color='k')
        ax.axline((0,0), (1,0), linestyle=':', linewidth=1, color='k')
        plt.title('Q-Q plot', fontsize=16)
        plt.legend(fontsize=12)
        plt.show()
    else:
        None
        
    if return_xy:
        return (nd_ppf, array)
    else:
        None

