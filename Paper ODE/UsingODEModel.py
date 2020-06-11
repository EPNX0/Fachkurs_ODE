"""
Created on Thu Jun 11 11:58:41 2020

@author: Erich
"""
import sys
import os
sys.path.append('C:/Users/Erich/Desktop/Theoretische Biophysik; Systembiologie/FK/Fachkurs_ODE/Paper ODE')
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from LearningTheta import SuEIR
from scipy.integrate import solve_ivp

Infectious = list(pd.read_csv('./TBPH/Infectious_TBPH.csv').iloc[:, -1])[72:401]
Removed = list(pd.read_csv('./TBPH/Removed_TBPH.csv').iloc[:, -1])[72:401]
Population = pd.read_csv('./Total_Populations.csv')['Germany']
Parameters = pd.read_csv('./TBPH/TBPHGridSearchOliver2.csv')
Keys = Parameters.keys()

'''Time span
'''
start = 0
end = len(Removed)+100
t_span = (start, end)
t_eval = list(range(start, end))

for i in range(1, len(Keys)):
    '''Initial Values
    '''
    I0 = Infectious[0]
    R0 = Removed[0]
    N0 = 10290
    E0 = 1004  # int((int(Parameters[Keys[i]][4][0]) * I0) + 0.5)
    S0 = N0 - I0 - R0 - E0
    Init_values = [S0, E0, I0, R0]
    '''Parameters
    '''
    params = Parameters[Keys[i]]
    beta = float(params[0])
    sigma = float(params[1])
    gamma = float(params[2])
    mu = float(params[3])
    parameters = [beta, sigma, gamma, mu]
    
    res = solve_ivp(SuEIR, t_span, Init_values, args=(parameters), t_eval=t_eval)
    
    X_true = np.arange(len(Infectious))
    X_pred = res.t
    I_true = Infectious
    R_true = Removed
    I_pred = res.y[-2]
    R_pred = res.y[-1]
    plt.plot(X_true, I_true, label='Infectious_true', color='blue')
    plt.plot(X_true, R_true, label='Removed_true', color='violet')
    plt.plot(X_pred, I_pred, label='Infectious_pred', linestyle='--', color='orange')
    plt.plot(X_pred, R_pred, label='Removed_pred', linestyle='--', color='red')
    plt.legend()
    plt.xlabel('Days')
    plt.ylabel('# Of Individuals')
    plt.savefig(f'./TBPH/GridOliver2/Grid{Keys[i]}.png', dpi=600, bbox_inches='tight')
    plt.close()

