### Three main methods of predicting prices for ML:
1. Linear Regression
2. ML-Based
3. Deep Learning based


### Linear Regression
- `np.polyfit(x, y, deg=1)` - this creates a regression using y and x values! Basically finding the line of best fit. For linear regression, highest degree is 1. 
    - Outputs: `array([0.94612934, 0.22855261])` -> this is the coefficients of line (x, b)
- `np.polyval(np.polyfit(x, y, deg=1), x)` - creates the y values using coef and x 
- `plt.plot(x, np.polyval(reg, x), 'r', lw=2.5, label='linear regression')` - creates the regression line 

### Basic idea of price prediction
- You construct an augmented matrix that implements Ax=b, where A is a matrix of the last few days price, and b is the current day price - you're using the last N days price as independent variables to predict today's price. 
    - The assumption is that there's some magic 'x' values where if you 'solve' for the system of linear equations, it'll give you today's price, vector b. 'Solve' because technically the function that 'solves' it actually minimizes the squared error (Ax-b)^2 if there's no discrete solution. 

```python
In [64]: data.head(10)
Out[64]: 
             price   lag_1   lag_2   lag_3   lag_4   lag_5
Date                                                      
2010-01-11  1.4513  1.4412  1.4318  1.4412  1.4368  1.4411
2010-01-12  1.4494  1.4513  1.4412  1.4318  1.4412  1.4368
2010-01-13  1.4510  1.4494  1.4513  1.4412  1.4318  1.4412
2010-01-14  1.4502  1.4510  1.4494  1.4513  1.4412  1.4318
2010-01-15  1.4382  1.4502  1.4510  1.4494  1.4513  1.4412
2010-01-19  1.4298  1.4382  1.4502  1.4510  1.4494  1.4513
2010-01-20  1.4101  1.4298  1.4382  1.4502  1.4510  1.4494
2010-01-21  1.4090  1.4101  1.4298  1.4382  1.4502  1.4510
2010-01-22  1.4137  1.4090  1.4101  1.4298  1.4382  1.4502
2010-01-25  1.4150  1.4137  1.4090  1.4101  1.4298  1.4382
```
Note last row: price today is 1.4150. lag_1 to lag_5 gives 5 past 5 days price, basically. 

You're going to call: 

`np.linalg.lstsq(data[cols], data['price'],rcond=None)[0]` - where your A is `data[cols]` and your b is `data['price']`. This gives you the vector for x: `array([ 0.98635864, 0.02292172, -0.04769849, 0.05037365,-0.01208135])`

The following are the tricks you use to get to that lag table:

```python
lags = 5
cols = []
for lag in range(1, lags + 1):
    col = f'lag_{lag}'
    data[col] = data['price'].shift(lag)
    cols.append(col)
data.dropna(inplace=True)
```

We noticed in this case that the x vector is heavily biased towards the previous day's price. This supports the "random walk hypothesis", since today’s price almost completely explains the predicted price level for tomorrow. The four other values hardly have any weight assigned.

`data['prediction'] = np.dot(data[cols], reg)`

However, the results seem to be pretty shit. I will be documenting this further in a blogpost- because I am not sure if I will actually be referencing this in the future. 