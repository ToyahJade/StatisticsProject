import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Define the data
h = np.array([1000, 828, 800, 600, 300])
d = np.array([1500, 1340, 1328, 1172, 800])
err_d = np.array([15, 15, 15, 15, 15])

# Define model for hypothesis 1
def hyp_1(x, *p):
    return p[0] * x


# Define model for hypothesis 2
def hyp_2(x, *p):
    return p[0] * x + p[1] * x ** 2


# Define model for hypothesis 3
def hyp_3(x, *p):
    return p[0] * (x ** p[1])


# Perform fitting for all three hypothesis
p_1, cov_1 = curve_fit(hyp_1, h, d, p0=[1], sigma=err_d)
p_2, cov_2 = curve_fit(hyp_2, h, d, p0=[1, 1], sigma=err_d)
p_3, cov_3 = curve_fit(hyp_3, h, d, p0=[1, 1], sigma=err_d)


# Plot the data
plt.errorbar(h, d, yerr=err_d, label = "Measured Data", fmt='o', color = 'black')

x = np.linspace(min(h), max(h), 500)
plt.plot(x, hyp_1(x, *p_1), label = "Hypothesis 1", color = 'red')
plt.plot(x, hyp_2(x, *p_2), label = "Hypothesis 2", color = 'green')
plt.plot(x, hyp_3(x, *p_3), label = "Hypothesis 3", color = 'blue')

plt.title("Fitting Three Hypotheses to Galileo's Data")
plt.xlabel("$h$ (punti)")
plt.ylabel("$d$ (punti)")
plt.legend(loc = "upper left")
plt.xlim(250, 1050)
plt.savefig("question2.jpg")

chi2_1 = np.sum(((d - hyp_1(h, *p_1)) / err_d)**2) / (d.size - p_1.size)
chi2_2 = np.sum(((d - hyp_2(h, *p_2)) / err_d)**2) / (d.size - p_2.size)
chi2_3 = np.sum(((d - hyp_3(h, *p_3)) / err_d)**2) / (d.size - p_3.size)

# Return the fit statistics for each model
print(chi2_1, p_1, cov_1)
print(chi2_2, p_2, cov_2)
print(chi2_3, p_3, cov_3)
