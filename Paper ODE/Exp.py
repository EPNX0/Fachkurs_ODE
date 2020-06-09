"""
Created on Tue Jun  9 12:57:47 2020

@author: Erich
"""
import math
from sklearn.metrics import mean_squared_error
import numpy as np
from scipy.optimize import minimize


def exp(a, b, t_span):
    try:
        assert a > 0 and b > 0
    except AssertionError:
        raise ValueError('Parameters a and b have to be greater than 0')

    r_of_t = []
    for t in range(t_span[0], t_span[1]):
        r = a * math.exp(-b * t)
        r_of_t.append(r)

    return np.array(r_of_t)


def MSE(parameters, ground_truth, time_span):
    a = parameters[0]
    b = parameters[1]
    y_true = ground_truth
    y_pred = exp(a, b, time_span)

    mse = mean_squared_error(y_true, y_pred)

    return mse


Fatalities = [1,2,3,4,5,6,7,8,9]  # Ground truth values
a = 1  # Parameter for Exp
b = 1  # Parameter for Exp
start = 1
end = 10
time_span = (start, end)
# boundaries for a and b: So that neither falls below 0... not pretty but helpful
bounds = ((1e-99, 1e99), (1e-99, 1e99))

res = minimize(MSE, [a, b], args=(Fatalities, time_span), bounds=bounds,
               options={'disp': True})
# print(res)
