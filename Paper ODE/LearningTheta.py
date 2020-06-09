"""
Created on Sat Jun  6 11:34:31 2020

@author: Erich
"""

import math
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize

'''ODE's for predicting results
'''


def SuEIR(t, x, beta, sigma, gamma, mu):
    '''Compartments
    '''
    S = x[0]
    E = x[1] 
    I = x[2]
    R = x[3]
    N = S + E + I + R
   # print(f'S:{S}; E:{E}; I:{I}; R:{R}; N:{N}')
   # print(f'beta:{beta}; sigma:{sigma}; gamma:{gamma}; mu:{mu}')

    '''ODE's
    '''
    dS = -(beta * (I + E) * S)/N
    dE = (beta * (I + E) * S)/N - sigma * E
    dI = mu * sigma * E - gamma * I
    dR = gamma * I

    return [dS, dE, dI, dR]


def LMSE(parameters, time_span, init_values, ground_truth, wanted_times):
   # print(f'Parms:{parameters}; t_span:{time_span}; init_vals:{init_values}; GT:{ground_truth}, t_eval:{wanted_times}')
    Predicitons = solve_ivp(SuEIR, time_span, init_values, args=(beta, sigma, gamma, mu), t_eval=wanted_times)
    I_pred = Predicitons.y[2]
    R_pred = Predicitons.y[3]
    I_true = ground_truth[0]
    R_true = ground_truth[1]

    try:
        assert len(I_pred) == len(I_true) == len(R_pred) == len(R_true)
    except AssertionError:
        raise ValueError(f'Check the lenghts of your arrays:\nI_pred: {len(I_pred)}\nI_true: {len(I_true)}\nR_pred: {len(R_pred)}\nR_true: {len(R_true)}')


    # smoothing parameter
    p = 1  # To prevent math error in Log-function

    results = []
    for t in range(len(I_pred)):
        result = math.sqrt(math.log10(I_pred[t] + p) - math.log10(I_true[t] + p)) + \
                 math.sqrt(math.log10(R_pred[t] + p) - math.log10(R_true[t] + p))
   #     print(f'result:{result}')
        results.append(result)
   # print(f'results:{results}')
    L = np.mean(results)
   # print(f'L:{L}')

    return L


'''Compartments; initial values
'''
S0 = 0  # Susceptible Individuals at t=0
E0 = 0  # Exposed Individuals at t=0; estimated via validation set
I0 = 0  # Infectious Individuals at t=0
R0 = 0  # Removed individuals at t=0
init_values = [S0, E0, I0, R0]  # Values from Data at start-time

'''time grid
'''
start = 0
end = 0
wanted_times = list(range(start, end))  # for t_eval
time_span = (start, end)

'''Reproted Values
'''
I_true = []
R_true = []
ground_truth = [I_true, R_true]
'''Parameters to tune
'''
beta = 0
sigma = 0
gamma = 0
mu = 0

results = minimize(LMSE, [beta, sigma, gamma, mu], args=(time_span,
                                                         init_values,
                                                         ground_truth,
                                                         wanted_times),
                   method='BFGS')
