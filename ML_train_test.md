# Machine Learning

### Train-test split
`rng = np.random.RandomState(0)`

`from sklearn.model_selection import train_test_split`

`dfTrain, dfTest = train_test_split(data, test_size=0.2, random_state=rng)` - this simply splits the DF into 2 DFs, randomly, with ratio of 8:2. 

### Stratified shuffle split
`from sklearn.model_selection import StratifiedShuffleSplit` - splits data in a way that preserves distribution of one of the classes in the data 

`sss = StratifiedShuffleSplit(n_splits=3, test_size=0.2, random_state=0)` - n_splits = number of sets of splits (like if you want 2 sets of train/test data, specify 2) 

```
y = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])

for train_index, test_index in sss.split(X, y):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

Output:
TRAIN: [3 8 4 9 2 6 1 5] TEST: [0 7]
TRAIN: [4 9 3 0 8 1 6 7] TEST: [2 5]
TRAIN: [8 2 5 6 1 3 7 9] TEST: [4 0]

Basically, if the ratio of your class of Y is 4:6, then in your training data, 4/10 will be from class 0, and 6/10 will be class 1; same for test data
```

```
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]
```

### Creating new features
`x_train = sms_train[['length', 'word_count', 'processed_word_count']].to_numpy()` - using some new columns as features

`y_train = [1 if l=="spam" else 0 for l in sms_train['label']]` - converting y to 1 and 0


### Count Vectorizer
`from sklearn.feature_extraction.text import CountVectorizer`

`count_vect = CountVectorizer()`

`X_train_counts = count_vect.fit_transform(sms_train.message)` -> transforms the message column into a sparse matrix of unique words, where each unique word is a bitmap in N columns (N=no. unique words). 
- e.g. if a row is "I love salmon salmon", and there's 10 unique words, this will be converted to sth like [0, 1, 0, 0, 0, 0, 1, 2, 0, 0] 
- rows with the same count of words will have the same vector

`count_vect.get_feature_names_out()` - returns the feature names; will be no. of columns in X_train_count

`cts = X_train_counts.toarray()` - the toarray() function converts the sparse matrix to array to be viewed

`cts[900]` - individual row of words 

`np.nonzero(cts[900])` - gets the indices with valus 

`count_vect.transform(['alamak']).toarray().sum()` - calls to words not found in the training set will be ignored in future calls to transform (i.e. when ran on testset)

### Adding a complete training data
`trainData = np.hstack( (cts, x_train) )`

```
[[  0,   0,   0, ...,  61,  15,   6],
[  0,   0,   0, ...,  72,  13,   9],
[  0,   0,   0, ...,  67,  12,   7],
...,
[  0,   0,   0, ..., 122,  19,  15],
[  0,   0,   0, ...,  56,  15,   9],
[  0,   0,   0, ...,  53,  10,   7]]

Creates each row as a count vector, along with the new features. 
```

### Generating count vector for test data 
`X_test_counts = count_vect.transform( sms_test.message )` - must not use fit_transform as it'll pollute the model  

`testData = np.hstack( (X_test_counts.toarray(), x_test) )`

### Classification 
`from sklearn.naive_bayes import MultinomialNB, ComplementNB`

`clf = MultinomialBN()`

`clf.fit( x_train, y_train)` - this is the meat. You're passing in just the 3 generated features, and with predictor, and finding the line of best fit, $\hat{h}$. the `clf` object will contain your regression results 

`from sklearn.metrics import precision_score, confusion_matrix, accuracy_score`

`y_pred = clf.predict(x_test)` - pass in the test samples, and use it to predict Y. 

`confusion_matrix(y_test, y_pred)` - this gives a confusion matrix of the results of your prediction
```
[[955   0]
 [160   0]]
```

`accuracy_score(y_test, y_pred)` - 0.8565022

`tn, fp, fn, tp = c.ravel()` -> where c is the confusion matrix. assigns: 955 0 160 0 as true negative, false positive, false negative, true positive

`sensitivity = tp/(tp+fn)`

`specificity = tn/(tn+fp)`



# Theory
Train-test split: train your model with 80% of the data (regression), use 20% of the data to test if model has any predictive powers. 

Note: you should train-test split done before preprocessing. This is to prevent data leakage, where info from test data is used to make choices when building the model, resulting in overly optimistic performance estimates. 