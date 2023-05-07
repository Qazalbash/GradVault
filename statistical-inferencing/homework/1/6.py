from collections import Counter

import numpy as np
from scipy.stats import weibull_min

# Set the parameters of the Weibull distribution
alpha = 3.0
beta = 2.0
n = 10000  # number of samples

ry = np.round(weibull_min.rvs(beta, loc=0, scale=alpha, size=n), 0)

# random numbers
# print(ry)

grouped_ry = Counter(ry)

# random numbers counted to generate the histogram
# print(grouped_ry)

for i, j in grouped_ry.items():
    print((int(i), j / n))
