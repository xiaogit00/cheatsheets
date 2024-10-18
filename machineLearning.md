# Data Preprocessing

Scikit-learn is a ML library. It contains libraries commonly used for data preprocessing. 

### Transformers  
- a class used in sklearn that enable data transformation
    - clean
    - reduce (dimensionality reduction)
    - expand (kernel approximation)
    - generate (feature extraction)

- It has the following methods: 
    - **fit**
        - learns model parameters from a training set (e.g mean & SD for normalization)
        - This is like performing OLS regression to find $\hat{h}$!
    - **transform**
        - applies transformation model to unseen data
    - **fit_transform**
        - models and transforms at the same time 
- It can be used for scaling (standardization and normalization) and encoding. 
