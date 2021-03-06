"""
Created on Tue Jun  9 12:57:47 2020

@author: Erich
"""
import os
import math
from sklearn.metrics import mean_squared_error
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
np.random.seed(1)


def exp(a, b, t_span):
    try:
        assert a > 0 and b > 0
    except AssertionError:
        raise ValueError('Parameters a and b have to be greater than 0')

    r_of_t = []
    for t in range(t_span[0], t_span[1]):
        r = a * math.exp(-b * t)
        r_of_t.append(r)
    ratios = np.array(r_of_t)

    return ratios


def MSE(parameters, Fatalities, time_span, ground_truth):
    X = list(range(time_span[0], time_span[1]))
    try:
        assert len(X) == len(Fatalities)
    except AssertionError:
        raise ValueError('Either your time_span is too small or array of Fatalities too long')

    a = parameters[0]
    b = parameters[1]
    y_true = ground_truth
    ratios = exp(a, b, time_span)
    y_pred = Fatalities * np.ones(ratios.shape)/ratios
    print(f'True:{y_true}; Pred:{y_pred}')

    mse = mean_squared_error(y_true, y_pred)

    return mse
# data are from 1/22/20 - 6/9/20; need data from 3/22/20 - 5/10/20
PATH = 'C:/Users/Erich/Desktop/Theoretische Biophysik; Systembiologie/FK/Fachkurs_ODE/Paper ODE'
deaths = pd.read_csv(os.path.join(PATH, 'Germany_deaths.csv'))
removed = pd.read_csv(os.path.join(PATH,'Germany_removed.csv'))

Removed = np.array(removed.iloc[:, -1])  # Ground Truth to minimize MSE
Fatalities = np.array(deaths.iloc[:, -1])  # To get Removed from Exp; Get from Data
a = 10  # Parameter for Exp
b = 0.09  # Parameter for Exp
start = 0
end = len(Fatalities)
time_span = (start, end)
# boundaries for a and b: So that neither falls below 0... not pretty but helpful
bounds = ((1e-99, 1e99), (1e-99, 1e99))

'''Tuning the parameters a and b
'''

res = minimize(MSE, [a, b], args=(Fatalities, time_span, Removed), bounds=bounds,
               options={'disp': True})
print(res)


parameters = dict(_=['a', 'b'], China=[3.56064863e+03, 2.36845288e-01], France=[0.36734085, 0.00265846],
                  Italy=[0.53838823, 0.01752495], US=[0.59783823, 0.01699403], Germany=[0.05099213, 0.00053813])
'''After successful tuning; plotting the result

ratios = exp(parameters['Germany'][0], parameters['Germany'][1], time_span)
X = list(range(start, end))
y_pred = Fatalities * np.ones(ratios.shape)/ratios
y_true = Removed

plt.plot(X, y_true, label='GroundTruth', color='blue')
plt.plot(X, y_pred, label='Predicted', color='red', linestyle='--')
plt.legend()
plt.ylabel('# Removed Cases')
plt.xlabel('Days')
plt.savefig('./Germany_Exp.png', dpi=600)
'''

'''We need number of active cases (I) and removed cases (R); most data only
include confirmed cases (I+R); We need to get I and R separately; Model can only
predict number of removed cases but more interested in fatality cases (deaths);
--> estimate number of removed cases (R) and subtract R from confirmed cases
to get active cases (I);
exp-fct is ratio between daily increased fatalities & removed cases (#Deaths/R)
'''                                                                   
