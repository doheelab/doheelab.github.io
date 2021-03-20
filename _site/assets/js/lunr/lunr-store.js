var store = [{
        "title": "Categorical variable encoding using aggregated mean and std",
        "excerpt":"Introduction Preprocessing cartegorical features is no easy task. The most basic technique would probably be one-hot-encoding method. However, one-hot-encoding is not efficient whenever the number of features is large. In this article, We will learn how to handle many categorical features effectively even though the number of features is large....","categories": ["machine-learning"],
        "tags": ["Pandas","Gradient Boosting","Ensemble"],
        "url": "/machine-learning/first-post/",
        "teaser": null
      },{
        "title": "Feature Pyramid Networks for Object Detection",
        "excerpt":"source: Feature Pyramid Networks for Object Detection (paper link) List of contents Introduction Feature Pyramids structure Network details Applications Experiments Conclusion Introduction      In this paper, authors introduce the multi-scale, pyramidal hierarchy of deep convolutional networks to construct feature pyramids with marginal extra cost. Using FPN in a basic Faster...","categories": ["machine-learning"],
        "tags": ["Object Detection","Review"],
        "url": "/machine-learning/review/",
        "teaser": null
      },{
        "title": "뉴럴 네트워크로 시계열 데이터 예측하기(time series prediction)",
        "excerpt":"블로그 글 “How To Use Neural Networks to Forecast Multiple Steps of a Time Series“을 참고하여 작성하였습니다. 이 글에서 다룰 내용은 시계열 데이터(상품 구매량 데이터)를 활용하여 현재로부터 1, 2, 3주 미래의 구매량을 예측하는 것입니다. 데이터 다운로드 https://archive.ics.uci.edu/ml/datasets/Sales_Transactions_Dataset_Weekly 800개가 넘는 상품에 대하여 52주 동안의 주별 구매량 데이터를 제공합니다. 데이터 불러오기 import...","categories": ["machine-learning"],
        "tags": ["Time Series","Neural Network","Machine Learning","Deep Learning","Keras","LSTM"],
        "url": "/machine-learning/TimeSeries/",
        "teaser": null
      },{
        "title": "케라스(Keras)로 Stacked LSTM 구현하기",
        "excerpt":"소개 이 글에서 다룰 내용은 Keras를 활용하여 Stacked LSTM 구하고 time series prediction task에 적용해보는 것입니다. Stacked LSTM을 사용하는 이유 Stacked LSTM은 hidden layer에 여러개의 LSTM 층을 쌓아서, 여러개의 memory cell을 이용할 수 있게 합니다. 일반적으로 neural network 에서 모델의 성능을 향상시키기 위해, hidden lyaer의 노드의 갯수를 과도하게 증가시키는 것보다는...","categories": ["machine-learning"],
        "tags": ["Time Series","Neural Network","Machine Learning","Deep Learning","Keras","LSTM","Stacked LSTM"],
        "url": "/machine-learning/TimeSeries2/",
        "teaser": null
      },{
        "title": "git reset과 git revert 쉽게 이해하기",
        "excerpt":"이 글은 git add, git commit, git push 등 git의 기본 개념에 대한 이해를 전제로 합니다. git reset, revert를 사용하는 이유 git reset과 git revert는 commit 또는 push했던 내용을 이전 상태로 되돌리는 경우에 사용하는 명령어입니다. 로컬의 commit 내용을 변경하고자 할 때는 reset을 주로 사용하지만, 원격 저장소에 push한 결과를 되돌리고 싶을...","categories": ["github"],
        "tags": ["Github","Git Reset","Git Revert"],
        "url": "/github/gitReset/",
        "teaser": null
      },{
        "title": "git stash 쉽게 이해하기",
        "excerpt":"git stash를 사용하는 이유 한 브랜치에서 작업을 하다가 다른 브랜치로 checkout 해야하는 경우가 있습니다. 이때 git checkout [브랜치]를 사용하면 다음과 같은 오류를 만나게 됩니다. rror: Your local changes to the following files would be overwritten by checkout: _posts/2021-03-17-gitStash.md Please, commit your changes or stash them before you can switch branches....","categories": ["github"],
        "tags": ["Github","Git Stash"],
        "url": "/github/gitStash/",
        "teaser": null
      },{
        "title": "비전공자로 개발자 커리어를 시작하기",
        "excerpt":"최근 개발자에 대한 수요가 많아지고, 대우가 좋아지면서 대학교에서 CS를 전공하지 않은 비전공자 개발자들이 많아지고 있습니다. 저 또한 CS가 아닌 다른 과목을 전공했지만 현재 AI 스타트업에서 개발자로 일하고 있습니다. 비전공자로 개발자 커리어를 이어나갈 때 여러 어려움을 겪게 되는데요. 대표적으로, CS 기반 지식의 깊이가 얇고 업무 효율성이 떨어진다. 운영체제, 데이터베이스(인덱싱, 파티셔닝, 파일시스템),...","categories": ["motivation"],
        "tags": ["Motivation","Career","Software Engineering"],
        "url": "/motivation/Seungineer1/",
        "teaser": null
      },{
        "title": "pandas를 이용하여 json 데이터 파싱하기",
        "excerpt":"이 글에서는 pandas를 이용하여 json 데이터를 분석하기 좋은 형태로 변환하는 방법에 대해 설명합니다. 데이터의 출처는 UCSD Amazon Product Dataset이고, Amazon의 Data Scientist인 Eugene Yan의 글을 참고하였습니다. 데이터 소개 본 글에서 사용할 상품 데이터의 형태는 다음과 같습니다. { \"asin\": \"0000031852\", \"title\": \"Girls Ballet Tutu Zebra Hot Pink\", \"price\": 3.17, \"imUrl\": \"http://ecx.images-amazon.com/images/I/51fAmVkTbyL._SY300_.jpg\",...","categories": ["pandas"],
        "tags": ["Pandas","Json","Parsing","Preprocessing"],
        "url": "/pandas/parse_json/",
        "teaser": null
      }]
