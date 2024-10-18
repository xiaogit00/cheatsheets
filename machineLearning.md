# Data Preprocessing

Scikit-learn is a ML library. It contains libraries commonly used for data preprocessing. Preprocessing is useful for changing raw feature vectors into representations more suitable for classification or regression

`from sklearn import preprocessing`


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

