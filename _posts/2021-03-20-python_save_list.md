---
title: '[Python] list를 txt 파일로 쓰기'
date: 2021-03-20 09:00:00 -0400
author: Dohee Jung
layout: posts
categories: Python
tags: [Python, Pickle]
---

이 글에서는 `pickle`를 이용하여 `list` 오브젝트를 `txt` 파일로 쓰고, 읽어오는 방법에 대해 설명합니다. 


```python
file_path = './test.txt'
mylist = ['123', '456', 'abc', 'def']

# txt 파일 쓰기
with open(file_path, 'wb') as lf:
    pickle.dump(mylist, lf)

# txt 파일 읽기
with open(file_path, 'rb') as lf:
    readList = pickle.load(lf)
    print(readList)
    
```

```
['123', '456', 'abc', 'def']
```