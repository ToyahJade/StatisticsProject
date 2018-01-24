import numpy as np
import question1a

# Function for finding the difference betweem values taken from the best-fit
# model
def delta(a, b, p):
	sum = 0
	for i in range(p.size):
		sum += p[i] * (a**i - b**i)

	return sum

# Function for finding the uncertainty between two values in the best-fit model
# to the data
def error_delta(a, b, dp):
	sum = 0

	for i in range(p.size):
		sum += (dp[i] * (a**i - b**i))**2

	return np.sqrt(sum)


# Calculate the difference between 3 sets of models for the Cubic polynomial
# model
diag = np.sqrt(np.diag(question1a.cov_cub))
p = question1a.p_cub

print('\na = 5, b = 6')
print(delta(5, 6, p))
print(error_delta(5, 6, diag))

print('\na = 5, b = 10')
print(delta(5, 10, p))
print(error_delta(5, 10, diag))

print('\na = 5, b = 20')
print(delta(5, 20, p))
print(error_delta(5, 20, diag))
