---
title: "Categorical Feature Processing Using Aggregated Mean and Std"
date: 2019-11-19 09:00:00 -0400
categories: machine-learning
---

## Introduction
Preprocessing cartegorical features is no easy task. The most common techniques would probably be one hot encoding. However, one-hot-encoding is not an efficient preprocessing method when the number of features is large. In this article, We will learn how to handle many categorical features effectively even when the number of features is large. This technique was used by the winner of Kaggle's "IEEE-CIS Fraud Detection" competition and can be found at https://www.kaggle.com/cdeotte/xgb-fraud-with-magic-0-9600.

## How the Magic Works
The magic is two things. First we need a UID variable to identify clients (credit cards). Second, we need to create aggregated group features. Then we remove UID. Suppose we had 10 transactions `A, B, C, D, E, F, G, H, I, J` as below.  
  
![image](http://playagricola.com/Kaggle/table.jpg)  
  
If we only use FeatureX, we can classify 70% of the transactions correctly. Below, yellow circles are `isFraud=1` and blue circles are `isFraud=0` transactions. After the tree model below splits data into left child and right child, we predict `isFraud=1` for left child and `isFraud=0` for right child. Thus 7 out of 10 predictions are correct.
  
![image](http://playagricola.com/Kaggle/tran.jpg)  
  
Now suppose that we have a UID which defines groups and we make an aggregated feature by taking the average of FeatureX within each group. We can now classify 100% of the transactions correctly. Note that we never use the feature UID in our decision tree.  
  
![image](http://playagricola.com/Kaggle/cred.jpg)

## Data Explanation

![Selection_004](https://user-images.githubusercontent.com/57972646/69213947-f28d9e00-0ba8-11ea-8347-61bfd27f4f3c.png)

In the original dataframe, there are 392 features including both numeric features and categrical features. 

## Feature Information

### Transaction Table
* TransactionDT: timedelta from a given reference datetime (not an actual timestamp)
* TransactionAMT: transaction payment amount in USD
* ProductCD: product code, the product for each transaction
* card1 - card6: payment card information, such as card type, card category, issue bank, country, etc.
* addr: address
* dist: distance
* P_ and (R__) emaildomain: purchaser and recipient email domain
* C1-C14: counting, such as how many addresses are found to be associated with the payment card, etc. The actual meaning is masked.
* D1-D15: timedelta, such as days between previous transaction, etc.
* M1-M9: match, such as names on card and address, etc.
* Vxxx: Vesta engineered rich features, including ranking, counting, and other entity relations.

Categorical Features:
* ProductCD
* card1 - card6
* addr1, addr2
* Pemaildomain Remaildomain
* M1 - M9

### Identity Table

Variables in this table are identity information – network connection information (IP, ISP, Proxy, etc) and digital signature (UA/browser/os/version, etc) associated with transactions. 
They're collected by Vesta’s fraud protection system and digital security partners.
(The field names are masked and pairwise dictionary will not be provided for privacy protection and contract agreement)

Categorical Features:
* DeviceType <br/>
* DeviceInfo <br/>
* id12 - id38 

### Read data

```python
import numpy as np, pandas as pd, os, gc
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

# LOAD TRAIN
X_train = pd.read_csv('../input/ieee-fraud-detection/train_transaction.csv',index_col='TransactionID', nrows=10000)
train_id = pd.read_csv('../input/ieee-fraud-detection/train_identity.csv',index_col='TransactionID', nrows=10000)
X_train = X_train.merge(train_id, how='left', left_index=True, right_index=True)
# LOAD TEST
X_test = pd.read_csv('../input/ieee-fraud-detection/test_transaction.csv',index_col='TransactionID', nrows=10000)
test_id = pd.read_csv('../input/ieee-fraud-detection/test_identity.csv',index_col='TransactionID', nrows=10000)
X_test = X_test.merge(test_id, how='left', left_index=True, right_index=True)
# TARGET
y_train = X_train['isFraud'].copy()
del train_id, test_id, X_train['isFraud']; 
x = gc.collect()
```

```python
# PRINT STATUS
>>>print('Train shape',X_train.shape,'test shape',X_test.shape)
Train shape (10000, 432) test shape (10000, 432)
```

We use pandas's factorize function to convert categorical variables into numeric variables.

### Pandas.factorize
pandas.factorize(values, sort=False, order=None, na_sentinel=-1, size_hint=None) <br/>
Encode the object as an enumerated type or categorical variable.

```python
>>> labels, uniques = pd.factorize(['b', 'b', 'a', 'c', 'b'])
>>> labels
array([0, 0, 1, 2, 0])
>>> uniques
array(['b', 'a', 'c'], dtype=object)
```

The next things to do are:

1. (factorize) Convert categorical variables using pandas' factorize function.
2. (memory reduce) If the max value is 32000 or higher, the data type is converted to 'int32' type, otherwise it is converted to 'int32'.
3. (positive) Change all numeric values to zero or above. 
4. (NAN to -1) Convert all NAN values to -1.

```python
# LABEL ENCODE AND MEMORY REDUCE
for i,f in enumerate(X_train.columns):
    # FACTORIZE CATEGORICAL VARIABLES
    if (np.str(X_train[f].dtype)=='category')|(X_train[f].dtype=='object'): 
        df_comb = pd.concat([X_train[f],X_test[f]],axis=0)
        df_comb,_ = df_comb.factorize(sort=True)        # (factorize)
        if df_comb.max()>32000: print(f,'needs int32')  # (memory reduce) 
        X_train[f] = df_comb[:len(X_train)].astype('int16')
        X_test[f] = df_comb[len(X_train):].astype('int16')
        
    # SHIFT ALL NUMERICS POSITIVE. SET NAN to -1
    elif f not in ['TransactionAmt','TransactionDT']:
        mn = np.min((X_train[f].min(),X_test[f].min()))
        X_train[f] -= np.float32(mn)                   # (positive)
        X_test[f] -= np.float32(mn)                    # (NAN to -1)
        X_train[f].fillna(-1,inplace=True)
        X_test[f].fillna(-1,inplace=True)
```

Since this is time series data, we use the first 75% as the train set and the latter 25% as the validation set.
```python
# CHRIS - TRAIN 75% PREDICT 25%
idxT = X_train.index[:3*len(X_train)//4]
idxV = X_train.index[3*len(X_train)//4:]
```

We will now test the performance of the original version of the XGBoost model.


```python
import xgboost as xgb
print("XGBoost version:", xgb.__version__)

clf = xgb.XGBClassifier( 
    n_estimators=2000,
    max_depth=12, 
    learning_rate=0.02, 
    subsample=0.8,
    colsample_bytree=0.4, 
    missing=-1, 
    eval_metric='auc',
    # USE CPU
    nthread=4,
    tree_method='hist' 
    # USE GPU
    #tree_method='gpu_hist' 
)
h = clf.fit(X_train.loc[idxT], y_train[idxT], 
    eval_set=[(X_train.loc[idxV],y_train[idxV])],
    verbose=50, early_stopping_rounds=100)

feature_imp = pd.DataFrame(sorted(zip(clf.feature_importances_,cols)), columns=['Value','Feature'])
plt.figure(figsize=(20, 10))
sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False).iloc[:50])
plt.title('XGB95 Most Important Features')
plt.tight_layout()
plt.show()
del clf, h; x=gc.collect()
```


You’ll find this post in your `_posts` directory. Go ahead and edit it and re-build the site to see your changes. You can rebuild the site in many different ways, but the most common way is to run `jekyll serve`, which launches a web server and auto-regenerates your site when a file is updated.

To add new posts, simply add a file in the `_posts` directory that follows the convention `YYYY-MM-DD-name-of-post.ext` and includes the necessary front matter. Take a look at the source for this post to get an idea about how it works.

Jekyll also offers powerful support for code snippets:

```python
def print_hi(name):
  print("hello", name)
print_hi('Tom')
```

Check out the [Jekyll docs][jekyll-docs] for more info on how to get the most out of Jekyll. File all bugs/feature requests at [Jekyll’s GitHub repo][jekyll-gh]. If you have questions, you can ask them on [Jekyll Talk][jekyll-talk].

[jekyll-docs]: https://jekyllrb.com/docs/home
[jekyll-gh]:   https://github.com/jekyll/jekyll
[jekyll-talk]: https://talk.jekyllrb.com/
