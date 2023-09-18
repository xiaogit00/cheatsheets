import numpy as np
import pandas as pd
import numpy.typing as npt
npArray = npt.ArrayLike

# Calculating Mean/Var/SD/Covariance by hand  

# Creating probability and returns arrays
p: npArray = np.array([.1, .2, .5, .2]) # where p = 4 states of the economy (depression, recession, normal, boom)  
r1: npArray  = np.array([-.3, -.1, .2, .5]) # % returns of company 1 in 4 states
r2: npArray  = np.array([0, .05, .2, -.05]) # % returns of company 2 in 4 states

# Expected Returns
expected_r1: float = sum(p*r1) # Output: 0.15
expected_r2: float = sum(p*r2) # Output: 0.1

# Deviations from expected returns
deviation1: npArray = r1-expected_r1 
deviation2: npArray = r2-expected_r2 

# Squared deviations from mean 
squared_d1: npArray = deviation1**2
squared_d2: npArray = deviation2**2

# Variance
var1: float = sum(p * squared_d1) 
var2: float = sum(p * squared_d2)

# Standard Deviation
sd1: float = var1**.5
sd2: float = var2**.5

# Covariance between a & b
cov_1_2: float = sum(p*deviation1*deviation2)
#Intuition: if both rows of deviation of a and b are positive, they 'deviate in the same direction', hence their product is positive. This will add to a positive 'correlation' weight. If one is positive when the other is negative, their product will be negative, hence lowering the correlation. The summation at the end 'adds' up their 'correlation' weights. 

# Correlation
rho_1_2: float = cov_1_2 / (sd1 * sd2)
# Intuition: it's basically covariance with the 'standardizing effect' applied when you divide by the product of the 2 SDs. 

##### Portfolio Returns ######
w1 = 0.5 # weight of portfolio 1
w2 = 0.5 # weight of portfolio 2

# Expected portfolio returns
expected_portfolio_r: float = expected_r1 * w1 + expected_r2 * w2 
print("Expected portfolio returns:",expected_portfolio_r)

# Variance of portfolio
var_p: float= var1*(w1**2) + 2*(w1**2)*(w2**2)*cov_1_2 + var2*(w2**2)
# notes: the variance of a portfolio depends on the variances of the individual security and the covariance between the 2 securities. 
# A positive covariance between the 2 securities increases the variance of the entire portfolio, and vice versa. It makes sense. If both securities are correlated, then your risk increases. 
# A negative covariance decreases the variance of the portfolio - you achieve a hedge. 
# print(var_p)

# Standard deviation of portfolio
sd_p: float = var_p**.5
print("Portfolio Standard Deviation",sd_p)

# print("The sd of portfolio is:", sd_p)
# print("The weighted average of standard deviations of both stocks is:", sd1*w1 + sd2*w2)
# print("The SD of portfolio is less than the weighted average of SDs of individual securities, due to the effects of a diversification.")
# Note: if 2 items are perfectly correlated, at rho = 1, then the weighted average of SDs is the same as SD of portfolio! So anything less than perfect correlation, there'll be diversification effects and the portfolio SD will be less. 

# Variance of portfolio composed of 1 riskless and 1 risky asset
var_rf = w1**2 * var1 #w1 = weight of risky asset + var of risky asset
# Risk is dramatically reduced. 

# Beta
beta = cov_1_m/var_m # where cov_1_m is the covariance of the individual stock 1 and the return on the market, and var_m is the return on the market 

beta_p = w1*beta1 + w2*beta2 # Beta of a portfolio is the weighted average of 2 securities. 

# Capm
expected_r = r_f + beta*(expected_r_m - r_f) 
# expected_r - expected return of a security; expected_r_m -> expected return of the market; r_f risk free rate




