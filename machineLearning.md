# Data Preprocessing

Scikit-learn is a ML library. It contains libraries commonly used for data preprocessing. 

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


### sklearn preprocessing  
Useful for changing raw feature vectors into representation more suitable for classification or regression.  

`from sklearn import preprocessing`

### On the idea of standardization and normalization 
Standardization - rescaling features so they have properties of standard normal distribution with mean of 0 and SD of 1. 