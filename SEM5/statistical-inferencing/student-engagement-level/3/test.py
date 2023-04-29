from matplotlib import pyplot as plt
from math import factorial, exp

n = range(14)
F = [14, 30, 36, 68, 43, 43, 30, 14, 10, 6, 4, 1, 1, 0]

l = 3.893

P = lambda n: (300 * l**n * exp(-l)) / (factorial(n))

F_ = [P(i) for i in n]

plt.tight_layout()
plt.bar(x=n, height=F, label='Observation', color='b', width=0.5)
plt.plot(n, F_, 'b-', label='Estimation', color='r')
plt.xticks(n)
plt.xlabel('n')
plt.ylabel('Frequency')
plt.legend()
plt.show()
