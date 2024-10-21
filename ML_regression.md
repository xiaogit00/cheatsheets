# Creating Regressions with statsmodels library 

`import statsmodels.api as sm`

`import statsmodels.formula.api as smf`

`est = sm.OLS( y: dataframe , sm.add_constant(X: dataframe) ).fit() ` - constructs regression of y = b0 + b1*X , where sm.add_constant literally just creates a constant column c, which is b0 column 

`est = smf.ols( formula='Sales ~ TV', data=advertising ).fit()` - same thing, pasty format

`est.summary()` - provides OLS regression results 

```
OLS Regression Results
Dep. Variable:	Sales	R-squared:	0.612
Model:	OLS	Adj. R-squared:	0.610
Method:	Least Squares	F-statistic:	312.1
Date:	Mon, 21 Oct 2024	Prob (F-statistic):	1.47e-42
Time:	19:46:35	Log-Likelihood:	-519.05
No. Observations:	200	AIC:	1042.
Df Residuals:	198	BIC:	1049.
Df Model:	1		
Covariance Type:	nonrobust		
coef	std err	t	P>|t|	[0.025	0.975]
const	7.0326	0.458	15.360	0.000	6.130	7.935
TV	0.0475	0.003	17.668	0.000	0.042	0.053
Omnibus:	0.531	Durbin-Watson:	1.935
Prob(Omnibus):	0.767	Jarque-Bera (JB):	0.669
Skew:	-0.089	Prob(JB):	0.716
Kurtosis:	2.779	Cond. No.	338.

```

`est.params` -> returns b0 and b1

`est.params[0] + est.params[1]*25` -> finding a particular prediction where x = 25  

`X_test = pd.DataFrame( {'TV': [25]} )` - creating input for prediction in line below

`est.predict( X_test )` - gives prediction of x=25

`est.ssr` - residual sum of squares (y-y_hat)

`est.rse` - residual standard error -> $\sqrt{RSS/(n-1)}$

`est.bse` - standard error of coefficients b0 and b1  

`est.conf_int()` - gets 95% confidence interval of coefficients 

`est.pvalues` - pvalues of coefficients 

`est.rsquared` - r^2 value - how much variation explained by predictors 

`poly_est = smf.ols( 'np.log(Sales) ~ TV + I(TV**2)', data=advertising ).fit()` - polynomial regression, of log(y)




### Plotting

`sns.regplot( x=advertising.TV, y=advertising.Sales, ci=None, scatter_kws={'color':'r', 's':9}  )` - simple linear regression chart in seaborn  

`plt.xlim( xmin=0 )` 

`sns.residplot(x=advertising.TV, y=advertising.Sales, lowess=True)` - residual plot of x and y 




