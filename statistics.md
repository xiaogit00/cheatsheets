## Stats with Scipy
`import scipy.stats as stats`

### T-tests
`motors.to_numpy()` - converting to a numpy array
`stats.ttest_1samp(motors, 26)` - runs two-sided t test and returns t-stat and p-value
> TtestResult(statistic=27.059409957536303, pvalue=4.3597985712286377e-159, df=26434)  
> null hypo: no statistical difference between sample mean and population mean  
> if p-value < 0.05 (significance value), can't reject null hypothesis 

