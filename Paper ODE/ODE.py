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
    N = S + E + I + R  # Some population < Total population

    '''ODE's
    '''
    dS = -(beta * (I + E) * S)/N
    dE = (beta * (I + E) * S)/N - sigma * E
    dI = mu * sigma * E - gamma * I
    dR = gamma * I

    return [dS, dE, dI, dR]

# TODO:
# URL for optimizer: https://docs.scipy.org/doc/scipy/reference/tutorial/optimize.html
# URL for MSE: https://www.kaggle.com/c/ashrae-energy-prediction/discussion/113064


def MSE(x):
    # smoothing parameter
    p = 1  # To ensure that log10 won't become a math error

    I_t_est = x[0]
    I_t = x[1]
    R_t_est = x[2]
    R_t = x[3]

    results = []

    for t in range(len(x[0])):
        result = math.sqrt(math.log10(I_t_est[t] + p) - math.log10(I_t[t] + p)) + \
            math.sqrt(math.log10(R_t_est[t] + p) - math.log10(R_t[t] + p))
    results.append(result)

    L = np.mean(results)

    return L


'''Compartments; Initial Values
'''
S0 = 0  # Susceptible Individuals at t=0
E0 = 0  # Exposed Individuals at t=0
I0 = 0  # Infectious Individuals at t=0
R0 = 0  # Removed individuals at t=0

Initial_values = [S0, E0, I0, R0]  # Values from Data at start-time

'''Time grid
'''
start = 0
end = 0
wanted_times = [None]  # t_eval in solve_ivp
time_span = (start, end)

'''Machine Learning to learn parameters
'''
# Reproted No.
I_t = np.array([])  # probably import via pandas through data
R_t = np.array([])  # probably import via pandas through data

'''
beta: Contact rate between S and (E+I)
sigma: ratio of cases in E that are either confirmed as I or dead/recovered without confirmation
gamma: Transition rate between I and R
mu: discovery rate of infected cases
'''
# Estimated No.
ODE_results = solve_ivp(SuEIR, time_span, Initial_values, args=(beta, sigma, gamma, mu))
I_t_est = ODE_results.y[-1]
R_t_est = ODE_results.y[-2]

X0 = [I_t_est, I_t, R_t_est, R_t]  # List for optimizer

L = minimize(MSE, X0, method='BFGS', options={'disp': True})  # Optimizer

