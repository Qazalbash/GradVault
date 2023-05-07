import matplotlib.pyplot as plt
import numpy as np

mean = 0
std = 10
sample_size = 25

# Generate 1000 random numbers
samples = np.random.normal(mean, std**2, sample_size)

# Compute the mean and variance

sample_mean = np.mean(samples)
sample_var = np.var(samples)

print(f"1/sqrt(2*pi*{sample_var})*exp(-1/2*(x-{sample_mean})^2/{sample_var})")

# Plot the histogram

sample_size = (
    lambda x: 1
    / np.sqrt(2 * np.pi * sample_var)
    * np.exp(-1 / 2 * (x - sample_mean) ** 2 / sample_var)
)
x = np.linspace(-300, 300, 1000)
plt.plot(x, sample_size(x))
plt.show()
