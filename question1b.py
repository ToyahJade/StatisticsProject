import numpy as np
import matplotlib.pyplot as plt
import question1a


# Define a function for calculating the model uncertainty based on the
# covariance function returned by the fitting routibe as well as the the
# degree of the polynomial, M
def sigma(x, cov, M):
    a = []
    for i in range(M+1):
        a.append(x ** i)

    a = np.array(a)
    print(a)
    return np.sqrt(np.sum(np.outer(a, a) * cov))

###############################
#### New figure for linear ####
###############################

plt.figure()

# Use previously found best fits for the polynomials to draw the
# confidence regions for the models
xline = np.linspace(0, 20, 100)
line = question1a.linear(xline, *question1a.p_lin)
sigma_lin = [sigma(x, question1a.cov_lin, 1) for x in xline]

plt.errorbar(question1a.x, question1a.y, yerr=question1a.sig, label = "Measured Data", fmt='o', color = 'black')
plt.plot(xline, line, label = "Linear Fit", color = 'red')
plt.fill_between(xline, line + sigma_lin, line - sigma_lin, facecolor = 'mistyrose')

plt.title("Fitting of 1$^{st}$, 2$^{nd}$ and 3$^{rd}$ Order Polynomials to the Given Data \nand Extrapolating the 1$^{st}$ Order Fit to $x = 20$.")
plt.xlabel("$x$")
plt.ylabel("$y$")
plt.legend(loc = "upper left")
plt.savefig("question1b_1.jpg")

###############################
#### New figure for square ####
###############################

plt.figure()

square = question1a.squared(xline, *question1a.p_sq)
sigma_sq = [sigma(x, question1a.cov_sq, 2) for x in xline]

plt.errorbar(question1a.x, question1a.y, yerr=question1a.sig,label = "Measured Data", fmt='o', color = 'black')
plt.plot(xline, square, label = "Square Fit", color = 'green')
plt.fill_between(xline, square + sigma_sq, square - sigma_sq, facecolor = 'honeydew')
plt.title("Fitting of 1$^{st}$, 2$^{nd}$ and 3$^{rd}$ Order Polynomials to the Given Data \nand Extrapolating the 2$^{nd}$ Order Fit to $x = 20$.")
plt.xlabel("$x$")
plt.ylabel("$y$")
plt.legend(loc = "upper left")
plt.savefig("question1b_2.jpg")

##############################
#### New figure for cubic ####
##############################

plt.figure()

cube = question1a.cubic(xline, *question1a.p_cub)
sigma_cub = [sigma(x, question1a.cov_cub, 3) for x in xline]

plt.errorbar(question1a.x, question1a.y, yerr=question1a.sig, label = "Measured Data",fmt='o', color = 'black')
plt.plot(xline, cube, label = "Cubic Fit", color = 'blue')
plt.fill_between(xline, cube + sigma_cub, cube - sigma_cub, facecolor = 'lightcyan')

plt.title("Fitting of 1$^{st}$, 2$^{nd}$ and 3$^{rd}$ Order Polynomials to the Given Data \nand Extrapolating the 3$^{rd}$ Order Fit to $x = 20$.")
plt.xlabel("$x$")
plt.ylabel("$y$")
plt.legend(loc = "upper left")
plt.savefig("question1b_3.jpg")
