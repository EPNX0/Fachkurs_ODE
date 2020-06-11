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
import random
random.seed(1)
np.random.seed(1)


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
    #print(f'S:{S}; E:{E}; I:{I}; R:{R}; N:{N}')
    #print(f'beta:{beta}; sigma:{sigma}; gamma:{gamma}; mu:{mu}')

    '''ODE's
    '''
    dS = -(beta * (I + E) * S)/N
    dE = (beta * (I + E) * S)/N - sigma * E
    dI = mu * sigma * E - gamma * I
    dR = gamma * I

    return [dS, dE, dI, dR]


def LMSE(parameters, time_span, init_values, ground_truth, wanted_times):
    '''
    Input:
        parameters = List of parameters
        time_span = time_span for solve_ivp
        init_values = y0 for solve_ivp
        ground_truth = True values to compare against
        wanted_times = t_eval for solve_ivp
        
    Return:
        Logarithmic-type mean square error between predicted and true values
    '''
    #print(f'Parms:{parameters}; t_span:{time_span}; init_vals:{init_values}; GT:{ground_truth}, t_eval:{wanted_times}')
    Predicitons = solve_ivp(SuEIR, time_span, init_values, args=(parameters), t_eval=wanted_times)
    I_pred = Predicitons.y[2]
    R_pred = Predicitons.y[3]
    I_true = ground_truth[0]
    R_true = ground_truth[1]
    #print(f'I_Pred: {I_pred}\nR_Pred:{R_pred}')

    try:
        assert len(I_pred) == len(I_true) == len(R_pred) == len(R_true)
    except AssertionError:
        raise ValueError(f'Check the lenghts of your arrays:\nI_pred: {len(I_pred)}\nI_true: {len(I_true)}\nR_pred: {len(R_pred)}\nR_true: {len(R_true)}')


    # smoothing parameter
    p = 1  # To prevent math error in Log-function

    results = []
    for t in range(len(I_pred)):
        result = (math.log10(I_pred[t] + p) - math.log10(I_true[t] + p))**2 + \
                 (math.log10(R_pred[t] + p) - math.log10(R_true[t] + p))**2
    #    print(f'result:{result}')
        results.append(result)
    #print(f'results:{results}')
    L = np.mean(results)
    #print(f'L:{L}')

    return L


def Argmin(parameters, time_span, initial_values, ground_truth, wanted_times, boundaries):
    '''
    Input:
        parameters: List of parameters for ODE-model
        time_span: time_span for solve_ivp() from scipy
        initial_values: y0 for solve_ivp() from scipy
        ground_truth: List of datapoints from real data to compare with results from ODE-model
        wanted_times: t_eval for solve_ivp() from scipy
        method: str; one of the methods available for minimize() from scipy
        boundaries: boundaries for the parameters for minimize() from scipy

    Output:
        Returns the result from minimize() from scipy
    '''
    res = minimize(LMSE, [param for param in parameters], args=(time_span,
                                                                initial_values,
                                                                ground_truth,
                                                                wanted_times),
                   bounds=boundaries, options={'disp': True})
    return res


