# Data Preprocessing

Scikit-learn is a ML library. It contains libraries commonly used for data preprocessing. Preprocessing is useful for changing raw feature vectors into representations more suitable for classification or regression

#### Importing
`from sklearn import preprocessing`

#### Standardization  
`scaler = preprocessing.StandardScaler()`   
`num_scaled = scaler.fit_transform(data.iloc[:, num_features])`  

```
TRANSFORMS a numerical DF: 

    a    b    c    d    e
0	130	322	109	2.4	3
1	115	564	160	1.6	0
2	124	261	141	0.3	0
3	128	263	105	0.2	1
4	120	269	121	0.2	1
...	...	...	...	...	...
265	172	199	162	0.5	0
266	120	263	173	0.0	0
267	140	294	153	1.3	0
268	140	192	148	0.4	0
269	160	286	108	1.5	3

INTO:
[[-0.07540984,  1.40221232, -1.75920811,  1.18101235,  2.47268219],
[-0.91675934,  6.0930045 ,  0.44640927,  0.48115318, -0.71153494],
[-0.41194964,  0.21982255, -0.37529132, -0.65611797, -0.71153494],
...,
[ 0.48548982,  0.85947603,  0.14367747,  0.21870599, -0.71153494],
[ 0.48548982, -1.11763472, -0.07255953, -0.56863558, -0.71153494],
[ 1.60728915,  0.70440852, -1.80245551,  0.39367078,  2.47268219]]

Note it'll be an array with R arrays, each with C items. Note the negative number in the first array item means 130 is below the mean, slightly. 
```

## Theory

### Transformers  
- a class used in sklearn that enable data transformation
    - clean
    - reduce (dimensionality reduction)
    - expand (kernel approximation)
    - generate (feature extraction)


### Main methods of transformers
1. **fit**
    - learns model parameters from a training set (e.g mean & SD for normalization)
    - This is like performing OLS regression to find $\hat{h}$!
2. **transform**
    - applies transformation model to unseen data
3. **fit_transform**
    - models and transforms at the same time 

Generally, transformers can be used for scaling (standardization and normalization) and encoding. 

### Estimators  
Class used to manage estimation and decoding of model. Can be used to discretization. 

### Standardization and normalization 
Standardization 
- rescaling features so they have properties of standard normal distribution with mean of 0 and SD of 1.  
- Feature scaling through standardization (or z-score normalization) can be an important preprocessing step for ML algos.  
- If a feature has a v large relative variance, it might end up dominating the estimator, and not learn well from other features. 


Normalization 
- scaling individual samples to have unit norm, independent of distribution of the samples.  

Standardization is feature-wise operation; Normalization is sample-wise operation. 

