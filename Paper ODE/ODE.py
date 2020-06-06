"""
Created on Sat Jun  6 11:34:31 2020

@author: Erich
"""

import math
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from scipy.integrate import solve_ivp
from sklearn.metrics import mean_squared_log_error

'''Compartments; Initial Values
'''
S0 = 0  # Susceptible Individuals
E0 = 0  # Exposed Individuals
I0 = 0  # Infectious Individuals
R0 = 0  # Removed individuals

Initial_values = [S0, E0, I0, R0, N]

'''Time grid
'''
start = 
end = 
wanted_times = [None]
time_span = (start, end)

'''ODE's for predicting results
'''
def SuEIR(t, x):
    '''Parameters
    '''
    beta = 0  # Contact rate between S and (E+I)
    sigma = 0  # ratio of cases in E that are either confirmed as I or dead/recovered without confirmation
    gamma = 0  # Transition rate between I and R
    mu = 0  # discovery rate of infected cases
    
    '''Compartments
    '''
    S = x[0]
    E = x[1]
    I = x[2]
    R = x[3]
    N = 0  # total population; (S + E + I + R)
    
    '''ODE's
    '''
    dS = -(beta * (I + E) * S)/N
    dE = (beta * (I + E) * S)/N - sigma * E
    dI = mu * sigma * E - gamma * I
    dR = gamma * I
    
    return [dS, dE, dI, dR]
    
'''Machine Learning to learn parameters
'''
# Reproted No.
I_t = []
R_t = []

# smoothing parameter
p = 0

# Estimated No.
ODE_results = solve_ivp(SuEIR, time_span, Initial_values)
I_t_est = [2]
R_t_est = [3]

# MSE:
T = len(I_t)
results = []
for t in range(T):
    result = math.sqrt(math.log10(I_t_est[t] + p) - math.log10(I_t[t] + p)) + \
        math.sqrt(math.log10(R_t_est[t] + p) - math.log10(R_t[t] + p))
    results.append(result)
    
L = np.mean(results)
    
