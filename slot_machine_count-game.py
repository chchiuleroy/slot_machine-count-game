# -*- coding: utf-8 -*-
"""
Created on Sun Feb 25 21:47:21 2018

@author: roy
"""

import numpy as np; import pandas as pd; from joblib import Parallel, delayed; import multiprocess

A = 10; y = range(1, A); size = 9; p = [1/(A-1)]*(A-1)

price = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
                  [0, 0, 0, 1, 1, 1, 2, 2, 2, 3], [0, 0, 0, 2, 2, 2, 3, 3, 3, 4], [0, 0, 0, 3, 3, 3, 4, 4, 4, 5],
                  [0, 0, 0, 4, 4, 4, 5, 5, 5, 6], [0, 0, 0, 5, 5, 5, 6, 6, 6, 7], [0, 0, 0, 6, 6, 6, 7, 7, 7, 8], 
                  [0, 0, 0, 7, 7, 7, 8, 8, 8, 9], [0, 0, 0, 8, 8, 8, 9, 9, 9, 10], [0, 0, 0, 9, 9, 9, 10, 10, 10, 11]])

def block(A, y, size, p, price):
    
    r = np.random.choice(y, size = size, replace = True, p = p)
    #r = np.array([1, 1, 1, 1, 1, 2, 2, 2, 2])
    pic, count = np.unique(r, return_counts = True)
    
    payoff_i = []
    for i in range(len(pic)):
        payoff_i.append(price[pic[i], count[i]])
        
    payoff = sum(payoff_i)
    
    free = np.array(np.where(count >= 5))
    
    if free > -1:
        run_f = 10
        
        def free_game(y, size, p):
            f = np.random.choice(y, size, replace = True, p = p)
            pic, count = np.unique(f, return_counts = True)
            payoff_i = []
            for i in range(len(pic)):
                payoff_i.append(price[pic[i], count[i]])
            payoff = sum(payoff_i)
            return payoff
        payfree_i = []
        for j in range(run_f):
            payfree_i.append(free_game(y, size, p))
    else:
        payfree_i = 0
    
    condition = np.sum(payfree_i)
    
    if condition > 0:
        pay = payoff + sum(payfree_i)
    else:
        pay = payoff
    
    return pay


# block(A, y, size, p, price)
    
# simulations 
cost = 5
sim = 1000000
n_cores = multiprocess.cpu_count(); n = n_cores - 4

results = Parallel(n_jobs = n, backend = 'threading')(delayed(block)(A, y, size, p, price) for i in range(sim))    
    
total = sum(results)

ad = total/sim/cost