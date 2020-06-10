"""
Created on Tue Jun  9 12:57:47 2020

@author: Erich
"""
import math
from sklearn.metrics import mean_squared_error
import numpy as np
from scipy.optimize import minimize


def exp(a, b, t_span, Fatalities):
    try:
        assert a > 0 and b > 0
    except AssertionError:
        raise ValueError('Parameters a and b have to be greater than 0')
    try:
        assert len(list(range(time_span[0],time_span[1])))==len(Fatalities)
    except AssertionError:
        raise ValueError('Either your time_span is too small or array of Fatalities too long')

    r_of_t = []
    for t in range(t_span[0], t_span[1]):
        r = a * math.exp(-b * t)
        r_of_t.append(r)
    ratios = np.array(r_of_t)
    print(f'Ratios:{ratios}')
    Removed = Fatalities * np.ones(ratios.shape)/ratios
    print(f'Removed:{Removed}')

    return Removed


def MSE(parameters, Fatalities, time_span, ground_truth):
    a = parameters[0]
    b = parameters[1]
    y_true = ground_truth
    y_pred = exp(a, b, time_span, Fatalities)
    print(f'True:{y_true}; Pred:{y_pred}')

    mse = mean_squared_error(y_true, y_pred)

    return mse


Removed = np.array([i**1.5 for i in range(100,200,10)])  # Ground Truth to minimize MSE
Fatalities = np.array([i**1.5 for i in range(50,150,10)])  # To get Removed from Exp; Get from Data
a = 5.70850227e-01  # Parameter for Exp
b = 1e-99  # Parameter for Exp
start = 1
end = 11
time_span = (start, end)
# boundaries for a and b: So that neither falls below 0... not pretty but helpful
bounds = ((1e-99, 1e99), (1e-99, 1e99))

res = minimize(MSE, [a, b], args=(Fatalities, time_span, Removed), bounds=bounds,
               options={'disp': True})
print(res)

'''We need number of active cases (I) and removed cases (R); most data only
include confirmed cases (I+R); We need to get I and R separately; Model can only
predict number of removed cases but more interested in fatality cases (deaths);
--> estimate number of removed cases (R) and subtract R from confirmed cases
to get active cases (I);
exp-fct is ratio between daily increased fatalities & removed cases (#Deaths/R)
'''



