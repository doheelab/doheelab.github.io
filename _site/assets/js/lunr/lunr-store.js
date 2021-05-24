var store = [{
        "title": "Categorical variable encoding using aggregated mean and std",
        "excerpt":"Introduction Preprocessing cartegorical features is no easy task. The most basic technique would probably be one-hot-encoding method. However, one-hot-encoding is not efficient whenever the number of features is large. In this article, We will learn how to handle many categorical features effectively even though the number of features is large....","categories": ["machine-learning"],
        "tags": ["pandas","gradient-boosting","ensemble"],
        "url": "/machine-learning/first-post/",
        "teaser": null
      },{
        "title": "Feature Pyramid Networks for Object Detection",
        "excerpt":"source: Feature Pyramid Networks for Object Detection (paper link) List of contents Introduction Feature Pyramids structure Network details Applications Experiments Conclusion Introduction      In this paper, authors introduce the multi-scale, pyramidal hierarchy of deep convolutional networks to construct feature pyramids with marginal extra cost. Using FPN in a basic Faster...","categories": ["machine-learning"],
        "tags": ["object-detection","review"],
        "url": "/machine-learning/review/",
        "teaser": null
      },{
        "title": "뉴럴 네트워크로 시계열 데이터 예측하기(time series prediction)",
        "excerpt":"블로그 글 “How To Use Neural Networks to Forecast Multiple Steps of a Time Series“을 참고하여 작성하였습니다. 이 글에서 다룰 내용은 시계열 데이터(상품 구매량 데이터)를 활용하여 현재로부터 1, 2, 3주 미래의 구매량을 예측하는 것입니다. 데이터 다운로드 https://archive.ics.uci.edu/ml/datasets/Sales_Transactions_Dataset_Weekly 800개가 넘는 상품에 대하여 52주 동안의 주별 구매량 데이터를 제공합니다. 데이터 불러오기 import...","categories": ["machine-learning"],
        "tags": ["time-series-prediction","neural-network","machine-learning","deep-learning","keras","lstm"],
        "url": "/machine-learning/TimeSeries/",
        "teaser": null
      },{
        "title": "케라스(Keras)로 Stacked LSTM 구현하기",
        "excerpt":"소개 이 글에서는 Keras를 활용하여 Stacked LSTM 구현을 구현하고 time series prediction에 적용하겠습니다. Stacked LSTM을 사용하는 이유 보통 neural network 에서 모델의 성능을 향상시키기 위해 hidden lyaer의 노드 갯수를 과도하게 증가시키는 것보다, 층을 깊게 쌓는 것이 더욱 효울적인 것으로 알려져 있습니다. Stacked LSTM은 LSTM이 더 복잡한 task를 해결할 수 있도록,...","categories": ["machine-learning"],
        "tags": ["time-series-prediction","neural-network","machine-learning","deep-learning","keras","lstm","stacked-lstm"],
        "url": "/machine-learning/TimeSeries2/",
        "teaser": null
      },{
        "title": "git reset과 git revert 쉽게 이해하기",
        "excerpt":"이 글은 git add, git commit, git push 등 git의 기본 개념에 대한 이해를 전제로 합니다. git reset, revert를 사용하는 이유 git reset과 git revert는 commit 또는 push했던 내용을 이전 상태로 되돌리는 경우에 사용하는 명령어입니다. 로컬의 commit 내용을 변경하고자 할 때는 reset을 주로 사용하지만, 원격 저장소에 push한 결과를 되돌리고 싶을...","categories": ["github"],
        "tags": ["github","git-reset","git-revert"],
        "url": "/github/gitReset/",
        "teaser": null
      },{
        "title": "git stash 쉽게 이해하기",
        "excerpt":"git stash를 사용하는 이유 한 브랜치에서 작업을 하다가 다른 브랜치로 checkout 해야하는 경우가 있습니다. 이때 git checkout [브랜치]를 사용하면 다음과 같은 오류를 만나게 됩니다. rror: Your local changes to the following files would be overwritten by checkout: _posts/2021-03-17-gitStash.md Please, commit your changes or stash them before you can switch branches....","categories": ["github"],
        "tags": ["github","git-stash"],
        "url": "/github/gitStash/",
        "teaser": null
      },{
        "title": "비전공자로 개발자 커리어를 시작하기",
        "excerpt":"최근 개발자에 대한 수요가 많아지고, 대우가 좋아지면서 대학교에서 CS를 전공하지 않은 비전공자 개발자들이 많아지고 있습니다. 저 또한 CS가 아닌 다른 과목을 전공했지만 현재 AI 스타트업에서 개발자로 일하고 있습니다. 비전공자로 개발자 커리어를 이어나갈 때 여러 어려움을 겪게 되는데요. 대표적으로, CS 기반 지식의 깊이가 얇고 업무 효율성이 떨어진다. 운영체제, 데이터베이스(인덱싱, 파티셔닝, 파일시스템),...","categories": ["motivation"],
        "tags": ["motivation","career","software-engineering"],
        "url": "/motivation/Seungineer1/",
        "teaser": null
      },{
        "title": "pandas를 이용하여 json 데이터 파싱하기",
        "excerpt":"이 글에서는 pandas를 이용하여 json 데이터를 분석하기 좋은 형태로 변환하는 방법에 대해 설명합니다. 데이터의 출처는 UCSD Amazon Product Dataset이고, Amazon의 Data Scientist인 Eugene Yan의 글을 참고하였습니다. 데이터 소개 본 글에서 사용할 상품 데이터의 형태는 다음과 같습니다. { \"asin\": \"0000031852\", \"title\": \"Girls Ballet Tutu Zebra Hot Pink\", \"price\": 3.17, \"imUrl\": \"http://ecx.images-amazon.com/images/I/51fAmVkTbyL._SY300_.jpg\",...","categories": ["pandas"],
        "tags": ["pandas","json","parsing","preprocessing"],
        "url": "/pandas/parse_json/",
        "teaser": null
      },{
        "title": "[JavaScript] 이진 트리(Binary Tree)와 트리 순회(Tree Traversal)",
        "excerpt":"이번 글에서는 이진 트리(Binary Tree)와 트리 순회(Tree Traversal)에 대해서 알아보고, JavaScript를 이용해서 구현해보겠습니다. 그래프(Graph) 노드(node)들과 노드들 사이를 연결하는 간선(edge)으로 구성되어 있습니다. 그래프는 root node가 하나 있고, 각 노드에는 child node가 연결되어 있습니다. 트리(Tree) 트리는 그래프의 일종으로, cycle이 없고, 서로 다른 두 노드를 잇는 길이 하나 뿐인 그래프를 트리라고 합니다. 노드가...","categories": ["algorithm"],
        "tags": ["algorithm","data-structure","javascript","tree","binary-tree"],
        "url": "/algorithm/binary_tree/",
        "teaser": null
      },{
        "title": "[Pytorch] Neural Collaborative Filtering - MLP 실험",
        "excerpt":"이번 글에서는 Pytorch를 이용하여, Neural Collaborative Filtering논문의 MLP(Multi Layer Perceptron) 파트의 실험을 구현해보겠습니다. 참고한 코드는 hexiangnan의 PyTorch 구현 코드입니다. 실습을 위한 코드는 링크에서 확인하실 수 있습니다. 학습 데이터 저희가 사용할 테이터는 MovieLens 1 Million (ml-1m)입니다. 데이터에 대한 자세한 설명은 링크에서 확인하실 수 있습니다. 이 데이터는 6000명의 유저가 4000개의 영화에 대해서...","categories": ["recommender-system"],
        "tags": ["recommender-system","machine-learning","pytorch","tensorboard","collaborative-filtering","mlp","neural-network"],
        "url": "/recommender-system/ncf_mlp/",
        "teaser": null
      },{
        "title": "Pytorch를 이용한 협업 필터링(Matrix Factorization) 구현",
        "excerpt":"이번 글에서는 Pytorch와 MovieLens 데이터셋을 이용하여, 협업필터링을 구현하겠습니다. 협업 필터링의 여러 기법 중에서 Matrix Factorization을 사용하겠습니다. 마지막으로 Neural Collaborative Filtering 논문에서 제안한 Generalized Matrix Factorization 모델에 대해서 알아보고, 기존 알고리즘과의 성능 비교 실험을 해보겠습니다. 1. Matrix Factorization 소개 유저 벡터($p_u$)와 아이템 벡터($p_i$)가 주어졌을 때, 유저와 아이템의 상호작용(interaction)을 다음과 같이 내적으로...","categories": ["recommender-system"],
        "tags": ["recommender-system","machine-learning","pytorch","tensorboard","collaborative-filtering","mlp","neural-network","matrix-factorization"],
        "url": "/recommender-system/ncf_mf/",
        "teaser": null
      },{
        "title": "JWT(JSON Web Token)를 활용한 권한 부여(authentication)",
        "excerpt":"JWT는 특정 유저에게 권한 부여(authorization)를 하기 위한 방법 중 하나입니다. 서버는 HTTP 요청을 받았을 때, 요청한 유저가 이미 인증(ID와 패스워드가 맞는 지 확인, authentication)을 한 유저인지 (JWT를 통해) 확인 후 알맞은 권한을 부여합니다. 세션(session)을 통한 권한 부여 권한 부여(authorization)을 위한 가장 일반적은 방법은 세션(session)을 통한 방법입니다. 서버는 인증 과정에서 유저의...","categories": ["web-development"],
        "tags": ["jwt","json-web-token","authentication","authorization","web-development","server","client","security"],
        "url": "/web-development/jwt/",
        "teaser": null
      },{
        "title": "[React] 유용한 Custom hook 만들기",
        "excerpt":"Custom hook은 React에서 제공하는 hook을 활용하여 사용자가 원하는 기능을 수행하도록 만든 함수를 의미합니다. 이를 통해 hook을 포함한 반복적인 작업을 다른 component에서 쉽게 사용할 수 있게 해줍니다. Custom hook의 이름은 반드시 use로 시작해야 하는데, 그 이유는 다음과 같습니다. 한눈에 보아도 Hook 규칙이 적용되는지를 파악할 수 있습니다. use로 시작하면 Hook 규칙의 위반...","categories": ["web-development"],
        "tags": ["react","react-hook","web-development","custom-hook","client"],
        "url": "/web-development/custom_hook/",
        "teaser": null
      },{
        "title": "[Clean Code] 실무에서 바로 쓰는 Frontend Clean Code",
        "excerpt":"이 글은 개발자 컨퍼런스 SLASH의 “실무에서 바로 쓰는 Frontend Clean Code” 동영상을 정리한 글입니다. 실무에서 클린 코드의 의의 실무에서 클린 코드가 중요한 이유는, 클린 코드는 유지보수 시간의 단축 (코드 리뷰, 디버깅)에 유리하기 때문입니다. 안일한 코드 추가의 함정 기존 코드에 기능(연결전문가)을 추가할 때, 조심하지 않으면 다음과 그림과 같이 하나의 기능을 하는...","categories": ["clean-code"],
        "tags": ["react","web-development","clean-code","javascript"],
        "url": "/clean-code/slash21_clean_code/",
        "teaser": null
      }]
