import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Define the data
i = np.array([10, 20, 30, 40, 50, 60, 70, 80])
r = np.array([8, 15.5, 22.5, 29, 35, 40.5, 45.5, 50])
dr = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])

# Define the model for formula_1
def formula_1(theta_i, *p):
    return p[0] * theta_i

# Define the model for formula_2
def formula_2(theta_i, *p):
    return p[0] * theta_i - p[1] * theta_i**2

# Define the model for Snell's Law
# Function converts the values into degree to keep it relevant and
# comparable to the previous 2 functions
def snell(theta_i, *p):
    return np.arcsin(np.sin(np.pi * theta_i / 180) / p[0]) * 180 / np.pi

# For each model perform the fitting and calculate the reduced Chi^2 value
p_1, cov_1 = curve_fit(formula_1, i, r, p0=[1], sigma=dr)
chi2_1 = np.sum(((r - formula_1(i, *p_1)) / dr)**2) / (r.size - p_1.size)
print(p_1)
print(chi2_1)

p_2, cov_2 = curve_fit(formula_2, i, r, p0=[1, 1], sigma=dr)
chi2_2 = np.sum(((r - formula_2(i, *p_2)) / dr)**2) / (r.size - p_2.size)

print(p_2)
print(chi2_2)

p_snell, cov_snell = curve_fit(snell, i, r, p0=[1.0], sigma=dr)
chi2_snell = np.sum(((r - snell(i, *p_snell)) / dr)**2) / (r.size - p_snell.size)

print(p_snell)
print(chi2_snell)

plt.errorbar(i, r, yerr=dr, label = "Measured Data", fmt='o', color = 'black')

x = np.linspace(min(i), max(i), 500)
plt.plot(x, formula_1(x, *p_1), label = "Hypothesis 1", color = 'red')
plt.plot(x, formula_2(x, *p_2), label = "Hypothesis 2", color = 'green')
plt.plot(x, snell(x, *p_snell), label = "Hypothesis 3 (Snell's Law)", color = 'blue')

plt.title("Fitting Three Hypotheses to Refraction Data")
plt.xlabel("Angle of Incidence (degrees)")
plt.ylabel("Angle of Refraction (degrees)")
plt.legend(loc = "upper left")
plt.savefig("question3.jpg")
