import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# Define the data
x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
y = np.array([2.7, 3.9, 5.5, 5.8, 6.5, 6.3, 7.7, 8.5, 8.7])
sig = np.array([0.3, 0.5, 0.7, 0.6, 0.4, 0.3, 0.7, 0.8, 0.5])


# Linear polynomial function
def linear(x, *p):
    return p[0] + p[1] * x


# Square polynomial function
def squared(x, *p):
    return p[0] + p[1] * x + p[2] * x**2


# Cubic polynomial function
def cubic(x, *p):
    return p[0] + p[1] * x + p[2] * x**2 + p[3] * x**3

# Make a scatter plot of the data points with its errors
plt.errorbar(x, y, yerr=sig, label = "Measured Data", fmt='o', color = 'black')
xline = np.linspace(x.min(), x.max(), 100)


# Perform a fit for the 1st order polynomial
p0 = [1, 1]
p_lin, cov_lin = curve_fit(linear, x, y, p0, sig)
print(cov_lin)
plt.plot(xline, linear(xline, *p_lin), label = "Linear Fit", color = 'red')

# Perform a fit for the 2nd order polynomial
p0 = [1, 1, 1]
p_sq, cov_sq = curve_fit(squared, x, y, p0, sig)
print(cov_sq)
plt.plot(xline, squared(xline, *p_sq), label = "Square Fit", color = 'green')

# Perform a fit for the 3rd order polynomial
p0 = [1, 1, 1, 1]
p_cub, cov_cub = curve_fit(cubic, x, y, p0, sig)
print(cov_cub)
plt.plot(xline, cubic(xline, *p_cub), label = "Cubic Fit", color = 'blue')

plt.title("Fitting of 1$^{st}$, 2$^{nd}$ and 3$^{rd}$ Order Polynomials to the Given Data")
plt.xlabel("$x$")
plt.ylabel("$y$")
plt.legend(loc = "upper left")
plt.savefig("question1a.jpg")
