---
title: "Categorical Feature Processing Using Aggregated Mean and Std"
date: 2019-11-19 09:00:00 -0400
categories: machine-learning
---

## Introduction
Preprocessing cartegorical features is no easy task. The most common techniques would probably be one hot encoding. However, one-hot-encoding is not an efficient preprocessing method when the number of features is large. In this article, We will learn how to handle many categorical features effectively even when the number of features is large. This technique was used by the winner of Kaggle's "IEEE-CIS Fraud Detection" competition and can be found at https://www.kaggle.com/cdeotte/xgb-fraud-with-magic-0-9600.


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
