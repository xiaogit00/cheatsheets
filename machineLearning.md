# Data Preprocessing

Scikit-learn is a ML library. It contains libraries commonly used for data preprocessing. Preprocessing is useful for changing raw feature vectors into representations more suitable for classification or regression

### Table of Content  
[Standardization](#standardization)

[Normalization](#normalization)

[One Hot Encoding](#one-hot-encoding)

[Discretization](#discretization)

[Theory](#theory)

### Importing
`from sklearn import preprocessing`

### Standardization  
`scaler = preprocessing.StandardScaler()`   

`num_scaled = scaler.fit_transform(data.iloc[:, num_features])` - transforms only numerical data

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
`num_scaled.mean(axis=0)` - check that mean is 0

`num_scaled.var(axis=0)` - check that var is 1 

### Normalization 
`normalizer = preprocessing.Normalizer()`  

`num_normalized = normalizer.fit_transform(data.iloc[:, num_features])`  

```
TRANSFORMS THE ABOVE INTO: 

[[0.357, 0.885, 0.299, 0.007, 0.008],
[0.192, 0.944, 0.268, 0.003, 0.   ],
[0.386, 0.812, 0.439, 0.001, 0.   ],
...,
[0.389, 0.817, 0.425, 0.004, 0.   ],
[0.5  , 0.686, 0.529, 0.001, 0.   ],
[0.464, 0.829, 0.313, 0.004, 0.009]]

*Note: Normalizer works on row data. formula: X' = X / |X|

For the first number of first row, it's derived with:  
130 / sqrt(130**2 + 322**2 + 109**2 + 2.4**2 + 3**2)
```

### Encoding Ordinal Data
`ord_enc = preprocessing.OrdinalEncoder( categories='auto' )`

### One Hot Encoding  
`oh_enc = preprocessing.OneHotEncoder( categories='auto', handle_unknown='ignore')` - if you specify handle unknown = ignore, if encoder finds unknown categories during transformation, the resulting column for this feature will be all zeros. 

`oh_enc.fit_transform(data.iloc[:, cat_features])` -> returns sparse matrix   

`oh_enc.fit_transform(data.iloc[:, cat_features]).toarray()` -> to view sparse matrix

```
Transforms: 

rest_ECG	chest	thal
0	2	4	3
1	2	3	7
2	0	2	7
3	0	4	7
4	2	2	3
...	...	...	...
265	0	3	7
266	0	2	7
267	2	2	3
268	0	4	6
269	2	4	3

INTO:
[[0., 0., 1., ..., 1., 0., 0.],
[0., 0., 1., ..., 0., 0., 1.],
[1., 0., 0., ..., 0., 0., 1.],
...,
[0., 0., 1., ..., 1., 0., 0.],
[1., 0., 0., ..., 0., 1., 0.],
[0., 0., 1., ..., 1., 0., 0.]]
```

`oh_enc.categories_` - returns the values of categories in qn  

### Discretization  
`discretizer = preprocessing.KBinsDiscretizer()`

At this step, you need to convert your target process column into an nx1 array :
`age_arr = np.array(data['age']).reshape(-1, 1)`  

```
[[80], 
[15],
[48]]
```  

`discretizer.fit(age_arr)` - must fit first, if not will output error when transforming ; returns KBinsDiscretizer object

`k = discretizer.transform(age_arr)`  -> returns sparse matrix

`k.toarray()`  

```
[[0., 0., 0., 0., 1.],
[0., 0., 0., 0., 1.],
[0., 0., 1., 0., 0.],
...,
[0., 0., 1., 0., 0.],
[0., 0., 1., 0., 0.],
[0., 0., 0., 0., 1.]]
```

### Polynomial Feature Construction
`poly_tfr = preprocessing.PolynomialFeatures()`

`poly_feats = poly_tfr.fit_transform(data.iloc[:, num_features])`

`poly_tfr.get_feature_names_out()`

```
It converts the original features (5): 
['rest_BP', 'cholesterol', 'ax_HR', 'oldpeak', 'vessels']

into (21):
['1', 'rest_BP', 'cholesterol', 'ax_HR', 'oldpeak', 'vessels',
'rest_BP^2', 'rest_BP cholesterol', 'rest_BP ax_HR',
'rest_BP oldpeak', 'rest_BP vessels', 'cholesterol^2',
'cholesterol ax_HR', 'cholesterol oldpeak', 'cholesterol vessels',
'ax_HR^2', 'ax_HR oldpeak', 'ax_HR vessels', 'oldpeak^2',
'oldpeak vessels', 'vessels^2']

5 (original) + 5C2 (10) + 5 (squared) + 1 = 21  
```

### Pipeline and ColumnTransformer
`from sklearn import pipeline`

`from sklearn import compose`

`numeric_transformer = pipeline.Pipeline(steps=[('scalar', scaler), ('poly', poly_tfr)])` - chains the StandardScaler and PolynomialFeatures transformers earlier 

`numeric_transformer.fit_transform(data)` 

However, the above creates polynomial variables comprising nominal and real variables, doesn't make much sense. 

**Column transformers** - performs different transformations for different columns
```
preprocessor = compose.ColumnTransformer(transformers=[
    ('num', numeric_transformer, num_features), #name, transformer, columns
    ('disc', discretizer, disc_features),
    ('cat', oh_enc, cat_features),
    ('ord', ord_enc, ordinal_features)
], remainder="passthrough")

processedData = preprocessor.fit_transform(data)
```

`data_processed = pd.DataFrame(processedData)` -- make back into DF 

### Cleaning missing values and imputing 

`from sklearn.impute import SimpleImputer`

`imp_mean = SimpleImputer( missing_values=np.nan, strategy='mean' )`

`imp_mean.fit(data_drop).statistics_` - to see what the values to impute for each column. 

```
[ 54.45769231,   0.67777778,   3.17407407, 131.34444444,
249.65925926,   0.14814815,   1.02222222, 149.67777778,
0.32962963,   1.05      ,   1.58518519,   0.67037037,
4.6962963 ]
- in this case, you're only interested in the first one, as that's the one with missing values
```

`data_filled = pd.DataFrame( imp_mean.transform(data_drop) )` - replaces those missing values with the mean!  

`data_filled.columns, data_filled.index = data_drop.columns, data_drop.index` - recreating the column names again


---

# Theory

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

### Standardization 
- rescaling features so they have properties of standard normal distribution with mean of 0 and SD of 1.  
- Feature scaling through standardization (or z-score normalization) can be an important preprocessing step for ML algos.  
- If a feature has a v large relative variance, it might end up dominating the estimator, and not learn well from other features. 


### Normalization 
- scaling individual samples to have unit norm, independent of distribution of the samples. 
- we can specify which norm to use (by default, it's l2 (Euclidean) norm) 
- Remember vectors; normalization simply reduces it to unit size. 

Standardization is feature-wise operation; Normalization is sample-wise operation. 

### One Hot Encoding
Assuming you have the following categorical variables:

```
rest_ECG	chest	thal
0	2	4	3
1	2	3	7
2	0	2	7
3	0	4	7
4	2	2	3
...	...	...	...
265	0	3	7
266	0	2	7
267	2	2	3
268	0	4	6
269	2	4	3
```
rest_ECG takes the values: 0, 1, 2  
chest takes the value 1, 2, 3, 4  
thal takes the values: 3, 6, 7  

What you're doing when running one hot encoding, is assigning a unique 'bitmap' to each unique combination of the samples.  

For instance, the first sample, with values 2, 4, 3, will be assigned:
[0, 0, 1, 0, 0, 0, 1, 1, 0, 0]

The length of this array will correspond to the total categories in question. (10 in this case)

Each bit corresponds to the presence or absence of an element. 

**At its core, it basically 'compresses' data by transforming 3 columns into 1 column** -> before, you had 3 columns for categorical data; now, you only need to work with 1, the 'data_cat' column, and each value specifies a unique permutation of category. 

This is pretty powerful - it allows you to quickly see the similarities between certain samples just by counting the length of sth.  

### Discretization
Putting continuous values into bins - very useful when combined with one hot encoding - making our model more expressive. 

For instance, age bins.  

The theory of discretization + one hot encoding is this: you have age column with values ranging 28 to 89. You want to put them into 5 bins. If you just discretize, the values will be 1, 2, 3, 4, 5 for the column. If you combine OHE, it'll be [0, 0, 0, 1, 0] for one of them.  

### Polynomial Feature Construction
Sometimes it's helpful to add complexity to model by considering non-linear features of input. `PolynomialFeatures` class allows us to generate higher order terms and interaction terms (representing joint effects of multiple features) to consider this non-linearity. 
