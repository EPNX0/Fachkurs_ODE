"""
Created on Wed Jun 10 10:44:32 2020

@author: Erich
"""

import sys
import os
sys.path.append('C:/Users/Erich/Desktop/Theoretische Biophysik; Systembiologie/FK/Fachkurs_ODE/Paper ODE')
from LearningTheta import SuEIR, LMSE, Argmin
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import random
random.seed(1)
np.random.seed(1)
try:
    dic
except NameError:
    dic = {' ':['beta', 'sigma', 'gamma', 'mu', 'E0', 'L']}

'''time grid

start = 0
end = 0
wanted_times = list(range(start, end))  # for t_eval
time_span = (start, end)
Reproted Values; needed for parameter tuning

I_true = []
R_true = []
ground_truth = [I_true, R_true]

Compartments; initial values

I0 = 0  # Infectious Individuals at t=0; who are infected and can transmit it (confimred - removed)
R0 = 0  # Removed individuals at t=0; who died+recovered
E0 = 0  # Exposed Individuals at t=0; infected, not tested
N0 = 0  # Some Population < Total population because of stay-at-home order
S0 = N0 - E0 - I0 - R0  # Susceptible Individuals at t=0; Those who can be infected
init_values = [S0, E0, I0, R0]  # Values from Data at start-time

Parameters to tune; random initialization

random_pars = np.array([0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.])
beta = random.choice(random_pars)  # Contact rate between S and (E + I)
sigma = random.choice(random_pars)  # Frac of E that either are I or R without confirmation
gamma = random.choice(random_pars)  # Transition rate between I and R
mu = random.choice(random_pars)  # Discovery rate of I
bounds = ((0, 1), (0, 1), (0, 1), (0, 1))  # boundaries for parameters
parameters = {'beta': beta, 'sigma': sigma, 'gamma': gamma, 'mu':mu}
'''


'''Concerning initial values:
I0 and R0 should be obtainable from Data;
E0 and N0 need to be estimated via validation set and grid search;
S0 would be therefore N0 - E0 - I0 - R0;
Data from 22.3.20 - 3.5.20 is training set; from 4.5.20 - 10.5.20 validation set; Total of 50 days
      index: 71   -  113                           114  -   120
'''
'''Test for the US
'''
PATH = 'C:/Users/Erich/Desktop/Theoretische Biophysik; Systembiologie/FK/Fachkurs_ODE/Paper ODE'
Infectious = list(pd.read_csv(os.path.join(PATH, 'US_infectious.csv')).iloc[:,-1])
Removed = list(pd.read_csv(os.path.join(PATH, 'US_removed.csv')).iloc[:, -1])
Population = pd.read_csv(os.path.join(PATH, 'Total_Populations.csv'))['US'][0]

# time span
start = 0
end = len(Removed)
wanted_times = list(range(start, 43))  # for t_eval, result for each day
time_span = (start, end)

# Reported Values for validation
I_true = Infectious[:-7]
R_true = Removed[:-7]
ground_truth = [I_true, R_true]

# Initial Values
I0 = Infectious[0]  # Infectious Individuals at t=0; who are infected and can transmit it (confirmed)
R0 = Removed[0]  # Removed individuals at t=0; who died+recovered
N0 = Population  # Some Population < Total population because of stay-at-home order
E0 = int((0.5 * N0) + 0.5)  # Exposed Individuals at t=0; infected but not tested
S0 = N0 - E0 - I0 - R0  # Susceptible Individuals at t=0; Those who can be infected
init_values = [S0, E0, I0, R0]  # Values from Data at start-time

# Parameters ti tune
random_pars = np.array([0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.])
beta = random.choice(random_pars)  # Contact rate between S and (E + I)
sigma = random.choice(random_pars)  # Frac of E that either are I or R without confirmation
gamma = random.choice(random_pars)  # Transition rate between I and R
mu = random.choice(random_pars)  # Discovery rate of I
bounds = ((0, 1), (0, 1), (0, 1), (0, 1))  # boundaries for parameters
parameters = {'beta': beta, 'sigma': sigma, 'gamma': gamma, 'mu':mu}

res = Argmin([param for param in parameters.values()], time_span, init_values, 
             ground_truth, wanted_times, bounds)
print(res)

if res.success==True:
    # Evaluation
    # Initial Values
    I0 = Infectious[0]  # Infectious Individuals at t=0; who are infected and can transmit it (confirmed)
    R0 = Removed[0]  # Removed individuals at t=0; who died+recovered
    E0 = int((0.5 * N0) + 0.5)  # Exposed Individuals at t=0; infected but not tested
    N0 = Population  # Some Population < Total population because of stay-at-home order
    S0 = N0 - E0 - I0 - R0  # Susceptible Individuals at t=0; Those who can be infected
    init_values = [S0, E0, I0, R0]  # Values from Data at start-time
    # ground truth
    I_true = Infectious[-7:]
    R_true = Removed[-7:]
    ground_truth = [I_true, R_true]
    # time span
    start = 0
    end = len(Removed)
    wanted_times = list(range(43, end))  # for t_eval, result for each day
    time_span = (start, end)
    # parameters taken from res if successful
    L = LMSE(res.x.tolist(), time_span, init_values, ground_truth, wanted_times)
    dic.update({'US0': [res.x[0], res.x[1], res.x[2], res.x[3], E0, L]})








DF = pd.DataFrame(dic)
DF.to_csv('./ResultsGridSearchTest.csv',
          index=False)

