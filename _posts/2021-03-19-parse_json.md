---
title: 'pandas를 이용하여 json 데이터 파싱하기'
date: 2021-03-19 09:00:00 -0400
author: Dohee Jung
categories: pandas
tags: [pandas, json, parsing, preprocessing]
---


이 글에서는 `pandas`를 이용하여 `json` 데이터를 분석하기 좋은 형태로 변환하는 방법에 대해 설명합니다. 

데이터의 출처는 [UCSD Amazon Product Dataset](http://jmcauley.ucsd.edu/data/amazon/links.html)이고, Amazon의 Data Scientist인 Eugene Yan의 [글](https://eugeneyan.com/writing/recommender-systems-baseline-pytorch/)을 참고하였습니다.

## 데이터 소개

본 글에서 사용할 상품 데이터의 형태는 다음과 같습니다.


```
{ 
"asin": "0000031852",
"title": "Girls Ballet Tutu Zebra Hot Pink",
"price": 3.17,
"imUrl": "http://ecx.images-amazon.com/images/I/51fAmVkTbyL._SY300_.jpg",
"related”:
    { "also_bought":[
		  	"B00JHONN1S",
		  	"B002BZX8Z6",
		  	"B00D2K1M3O", 
		  	...
		  	"B007R2RM8W"
                    ],
      "also_viewed":[ 
		  	"B002BZX8Z6",
		  	"B00JHONN1S",
		  	"B008F0SU0Y",
		  	...
		  	"B00BFXLZ8M"
                     ],
      "bought_together":[ 
		  	"B002BZX8Z6"
                     ]
    },
"salesRank":
    { 
      "Toys & Games":211836
    },
"brand": "Coxlures",
"categories":[ 
	    [ "Sports & Outdoors",
	      "Other Sports",
	      "Dance"
	    ]
    ]
}
```

<div align="center">
  <i>Amazon product dataset</i>
</div>

위 데이터는 다음 [링크](http://jmcauley.ucsd.edu/data/amazon/links.html)에서 다운받을 수 있습니다. 

이 페이지에서 다운로드 할 수 있는 여러 데이터 중에서, *meta_Electronics.json.gz*를 사용하였습니다.

---
## 파싱(Parsing)이란?

`json` 데이터를 어떻게 분석에 사용할 수 있을까요?

위와 같은 소스파일을 사용자가 해석하기 좋은 형태로 변환하는 작업을 **파싱(parsing)**이라고 합니다.

**파싱(Parsing)**은 소스파일을 의미있는 단어의 단위로 잘라서 해석하는 작업을 말합니다.

파싱의 예는 다음과 같습니다. 

- `printf("hello")`라는 구문을 `printf`와 `(, ", hello, ", )`로 단어와 기호들을 하나씩 나누기

- 브라우저에서 HTML을 DOM 트리로 변환
  
- 어떤 data를 원하는 form으로 만들어내는 작업

이처럼 data를 이해하기 쉬운 형태로 변환하는 작업을 **파싱**이라고 할 수 있습니다.

---

## pandas를 이용하여 json 데이터 파싱하기

용량이 큰 `json` 파일을 한번에 읽어서 `dataframe`으로 변환하는 것은 불가능합니다.

따라서 파이썬의 `generator`를 사용하여 한줄씩 읽고 변환하는 것이 좋습니다. 

```python
def parse(path: str):
    g = gzip.open(path, "rb")
    for l in g:
        yield eval(l)
```

### `eval`

위 코드의 `eval` 함수는 문자열을 실행하는 함수 입니다.

`json`의 `value`값은 문자열로 저장되어 있기 때문에, 이를 `dictionary`로 변환하기 위해서 `eval` 함수를 사용합니다.

```python
eval(b"{'asin': '0132793040', 'imUrl': 'http://ecx.images-amazon.com/images/I/31JIPhp%2BGIL.jpg', 'categories': [['Electronics', 'Computers & Accessories', 'Cables & Accessories', 'Monitor Accessories']], 'title': 'Kelby Training DVD: Mastering Blend Modes in Adobe Photoshop CS5 By Corey Barker'}\n")
```

```
{'asin': '0132793040', 'imUrl': 'http://ecx.images-amazon.com/images/I/31JIPhp%2BGIL.jpg', 'categories': [['Electronics', 'Computers & Accessories', 'Cables & Accessories', 'Monitor Accessories']], 'title': 'Kelby Training DVD: Mastering Blend Modes in Adobe Photoshop CS5 By Corey Barker'}
```

<div align="center">
  <i>eval 함수의 활용 예시</i>
</div>

위에서 정의한 `parse` 함수를 사용하여, `json` 데이터를 `dataframe`으로 변경하는 코드는 다음과 같습니다.

```python
def parse_json_to_df(path: str) -> pd.DataFrame:
    i = 0
    df_dict = {}
    for d in parse(path):
        df_dict[i] = d
        i += 1
        if i % 100000 == 0:
            logger.info("Rows processed: {:,}".format(i))

    df = pd.DataFrame.from_dict(df_dict, orient="index")
    df["related"] = df["related"].astype(str)
    df["categories"] = df["categories"].astype(str)
    df["salesRank"] = df["salesRank"].astype(str)
    return df
```

### `from_dict`
`from_dict` 함수는 `dictionary`를 `dataframe`으로 변환시켜주는 함수입니다. 예를 들어,
```python
df_dict = {0: {"a": 1, "b": 2}, 1: {"a": 3, "b": 4}}
pd.DataFrame.from_dict(df_dict)
```

||0|1|
|------|---|---|
|a|1|3|
|b|2|4|


<br/>

와 같이 사용할 수 있습니다. 이때 `orient`의 기본값은 `columns`이며, `dictionary`의 키를 열의 레이블로 설정합니다.

만일 `orient='index'`로 설정하면 `dictionary`의 키를 행의 레이블로 설정할 수 있습니다.

```python
df_dict = {0: {"a": 1, "b": 2}, 1: {"a": 3, "b": 4}}
pd.DataFrame.from_dict(df_dict, , orient="index")
```

||a|b|
|------|---|---|
|0|1|2|
|1|3|4|

저희도 `orident="index"`로 설정하여 코드를 실행하겠습니다.

---

<br/>

## 전체코드

```python
import gzip
import pandas as pd
import gzip
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(message)s")

# create console handler and set level to info
ch = logging.StreamHandler()
ch.setFormatter(formatter)
ch.setLevel(logging.INFO)

# add ch to logger
logger.addHandler(ch)

def parse(path: str):
    g = gzip.open(path, "rb")
    for l in g:
        yield eval(l)

def parse_json_to_df(path: str) -> pd.DataFrame:
    i = 0
    df_dict = {}
    for d in parse(path):
        df_dict[i] = d
        i += 1
        if i % 100000 == 0:
            logger.info("Rows processed: {:,}".format(i))

    df = pd.DataFrame.from_dict(df_dict, orient="index")
    df["related"] = df["related"].astype(str)
    df["categories"] = df["categories"].astype(str)
    df["salesRank"] = df["salesRank"].astype(str)
    return df

parse_json_to_df("../../save/meta_Electronics.json.gz")
```

## 실행결과

```
2021-03-19 12:06:36,158 - Rows processed: 100,000
2021-03-19 12:06:44,236 - Rows processed: 200,000
2021-03-19 12:06:53,033 - Rows processed: 300,000
2021-03-19 12:07:02,418 - Rows processed: 400,000
```


![](https://user-images.githubusercontent.com/57972646/111726593-97f37780-88ac-11eb-991f-85974c30362d.png)


## Reference

[1] [[Building a Strong Baseline Recommender in PyTorch, on a Laptop](https://eugeneyan.com/writing/recommender-systems-baseline-pytorch/)

[2] [파싱(Parsing)과 컴파일(Compile)의 차이점](https://rednose86.tistory.com/10)
