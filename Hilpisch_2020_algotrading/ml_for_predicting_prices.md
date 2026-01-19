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

`data['prediction'] = np.dot(data[cols], reg)` - note, reg is coeffs. This implements Ax to get b. 

`data[['price', 'prediction']].loc['2019-10-1':].plot(figsize=(10, 6))` - plotting deviations between price and prediction

However, the results seem to be pretty shit. I will be documenting this further in a blogpost- because I am not sure if I will actually be referencing this in the future. 

But still, some tricks:
**Using returns to model**
- `data['return'] = np.log(data['price'] / data['price'].shift(1))` - getting a returns column, standard. Need to do `data.dropna(inplace=True)` immediately afterwards or rest won't work
- Creating lag columns based on returns:
```python
cols = []
for lag in range(1, lags + 1):
    col = f'lag_{lag}'
    data[col] = data['return'].shift(lag)
    cols.append(col)
data.dropna(inplace=True)
```

```
               lag_1     lag_2     lag_3     lag_4     lag_5
Date                                                        
2010-01-12       NaN       NaN       NaN       NaN       NaN
2010-01-13 -0.001310       NaN       NaN       NaN       NaN
2010-01-14  0.001103 -0.001310       NaN       NaN       NaN
2010-01-15 -0.000551  0.001103 -0.001310       NaN       NaN
2010-01-19 -0.008309 -0.000551  0.001103 -0.001310       NaN
```
- `reg = np.linalg.lstsq(data[cols], data['return'],rcond=None)[0]` - solving for coeffs on return data instead

- `data['prediction'] = np.dot(data[cols], reg)`

```
In [92]: data
Out[92]: 
             price     lag_1     lag_2     lag_3     lag_4     lag_5    prediction    return
Date                                                                                        
2010-01-20  1.4101 -0.005858 -0.008309 -0.000551  0.001103 -0.001310  6.055298e-05 -0.013874
2010-01-21  1.4090 -0.013874 -0.005858 -0.008309 -0.000551  0.001103  4.534101e-04 -0.000780
2010-01-22  1.4137 -0.000780 -0.013874 -0.005858 -0.008309 -0.000551 -2.102026e-06  0.003330
2010-01-25  1.4150  0.003330 -0.000780 -0.013874 -0.005858 -0.008309  4.223334e-04  0.000919
2010-01-26  1.4073  0.000919  0.003330 -0.000780 -0.013874 -0.005858 -9.825399e-05 -0.005457
```

So you see here, this is essentially a lagging returns - like, last 1 day return, last 2 days. You use these to predict current day return. 

**Using Signs**
`np.sign(data['return'] * data['prediction']).value_counts()` - gets the proportion of predictions that are correct

`reg = np.linalg.lstsq(data[cols], np.sign(data['return']),rcond=None)[0]` - solving for Ax=b augmented matrix where b is the sign of returns instead - essentially a positive or negative value. This means: get me the coeffs that correctly predict whether the return is pos or neg. 

`data['prediction'] = np.sign(np.dot(data[cols], reg))` - getting the actual predictions using the optimized params, reg

This gives more hits apparently:

```
In [98]: hits = np.sign(data['return'] *
    ...: data['prediction']).value_counts()

In [99]: hits
Out[99]: 
 1.0    1301
-1.0    1191
 0.0      13
Name: count, dtype: int64
```

Now, you can calculate strategy returns in the same way:
`data['strategy'] = data['prediction'] * data['return']` - what this means is that you're entering trade for the day, using the prediction value - and getting the daily returns. The data object:

```
In [100]: data.head()
Out[100]: 
             price     lag_1     lag_2     lag_3     lag_4     lag_5  prediction    return
Date                                                                                      
2010-01-20  1.4101 -0.005858 -0.008309 -0.000551  0.001103 -0.001310         1.0 -0.013874
2010-01-21  1.4090 -0.013874 -0.005858 -0.008309 -0.000551  0.001103         1.0 -0.000780
2010-01-22  1.4137 -0.000780 -0.013874 -0.005858 -0.008309 -0.000551         1.0  0.003330
2010-01-25  1.4150  0.003330 -0.000780 -0.013874 -0.005858 -0.008309         1.0  0.000919
2010-01-26  1.4073  0.000919  0.003330 -0.000780 -0.013874 -0.005858         1.0 -0.005457
```

