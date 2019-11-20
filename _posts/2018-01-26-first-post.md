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

## Read Data
```python
import numpy as np, pandas as pd
X_train = pd.read_csv('../input/ieee-fraud-detection/train_transaction.csv',index_col='TransactionID', nrows=20)
X_train
```
![Selection_002](https://user-images.githubusercontent.com/57972646/69213746-5c597800-0ba8-11ea-974f-54e53957a872.png)


In the original dataframe, there are 392 features including both numeric features and categrical features. 

## Feature Information


## Import Libraries
```python
import numpy as np, pandas as pd, os, gc
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
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
