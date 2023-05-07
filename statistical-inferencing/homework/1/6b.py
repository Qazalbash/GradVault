import numpy as np
from scipy.stats import weibull_min

orignal_beta = 2.0
orignal_alpha = 3.0

n = 10000

x = weibull_min.rvs(orignal_alpha, loc=0, scale=orignal_beta, size=n)

log_x = np.log(x)
mean_log_x = np.mean(log_x)


def f(b):
    x_power_beta = x**b
    return b * (np.dot(x_power_beta, log_x) / np.sum(x_power_beta) -
                mean_log_x) - 1


def f_prime(b):
    return (f(b + delta) - f(b)) / delta


beta = 3
beta_old = 0

delta = 0.001
error_margin = 0.0001
max_step = 1000

while abs(beta - beta_old) > error_margin and max_step > 0:
    beta_old = beta
    beta = beta - f(beta) / f_prime(beta)
    max_step -= 1

print("beta =", beta)
print("error in the value of beta =",
      round(abs(beta - orignal_alpha) / orignal_alpha * 100, 3), "%")

alpha = (np.mean(x**beta))**(1 / beta)

print("alpha =", alpha)
print("error in the value of alpha =",
      round(abs(alpha - orignal_beta) / orignal_beta * 100, 3), "%")
