---
title: '[Clean Code] 실무에서 바로 쓰는 Frontend Clean Code'
date: 2021-05-24 00:00:00 -0400
categories: clean-code
tags: [react, web-development, clean-code, javascript, front-end]
---

이 글은 개발자 컨퍼런스 SLASH의 ["실무에서 바로 쓰는 Frontend Clean Code"](https://www.youtube.com/watch?v=edWbHp_k_9Y) 동영상을 정리한 글입니다. 



## 실무에서 클린 코드의 의의

실무에서 클린 코드가 중요한 이유는, 클린 코드는 **유지보수 시간의 단축 (코드 리뷰, 디버깅)**에 유리하기 때문입니다.



## 안일한 코드 추가의 함정

기존 코드에 기능(연결전문가)을 추가할 때, 조심하지 않으면 다음과 그림과 같이 하나의 기능을 하는 코드를 분산 배치하기 쉽습니다. 

<div style="text-align:center"><img src="https://user-images.githubusercontent.com/71360682/119293520-86a06000-bc8d-11eb-86e5-6fea1428be64.png" /></div>

<div align="center">
  <i>연결 중인 전문가가 있을 때 팝업을 띄우는 로직 주가 </i>
</div>

<br/>
<br/>
 
위 코드의 문제점은 다음과 같습니다.

### **1. 하나의 목적인 코드가 흩어져 있습니다. (초록색 부분)**


연결전문가 관련 기능이 흩어져 있기 때문에, 기능을 이해하기 위해서 스크롤을 이동해야 합니다. 


해결 방법은, **같은 기능을 하는 부분을 하나의 컴포넌트로 분리**하는 것입니다.

<div style="text-align:center"><img src="https://user-images.githubusercontent.com/71360682/119295381-a89be180-bc91-11eb-8122-9cb71e194e6d.png" /></div>

<div align="center">
  <i> 연결 전문가 관련 기능을 PopupTriggerButton 컴포넌트로 분리 </i>
</div>

<br/>


### **2. 하나의 함수가 여러 가지 일을 하고 있습니다.**

세부 구현을 모두 읽어야 함수의 역할을 알 수 있게 됩니다. 

해결 방법은, **각 함수가 하나의 일만 하도록 쪼개는 것**입니다.



<div style="text-align:center"><img src="https://user-images.githubusercontent.com/71360682/119295397-b6e9fd80-bc91-11eb-92a8-2978b6bd850f.png" /></div>


<div align="center">
  <i> 약관동의 함수를 쪼개서 필요한 시점에 호출하도록 변경 </i>
</div>

<br/>

### **3. 함수의 세부 구현 단계가 제각각입니다.**

`handleQuestionSubmit`, `handleMyExpertQuestionSubmit`은 둘 다 이벤트 핸들링 관련 함수인데, 이 중 `handleQuestionSubmit`은 질문전송 이외에 여러가지 일을 동시에 하고 있기 때문에 읽기 어렵습니다.

해결 방법은, **각 함수의 세부 구현 단계를 통일**하는 것입니다.

<div style="text-align:center"><img src="https://user-images.githubusercontent.com/71360682/119295352-98840200-bc91-11eb-81b7-febe7c4c1f54.png" /></div>

<div align="center">
  <i> 기존 코드에서 세부 구현 단계를 통일 </i>
</div>

<br/>

## 로직을 빠르게 찾을 수 있는 코드

클린 코드는 짧은 코드가 아니라, 원하는 로직을 빠르게 찾을 수 있는 코드입니다. 이를 위해서 다음 세 가지를 유념해야 합니다.

- **응집도:** 하나의 목적을 가진 코드의 응집도를 높여서 뭉쳐두기
- **단일책임:** 하나의 함수는 하나의 일만 하도록 쪼개기
- **추상화:** 함수의 세부구현 단계를 통일

<br/>

### **1. 응집도**

당장 몰라도 되는 디테일은 뭉쳐서 짧은 코드로 만들면 좋지만, 코드 파악에 필수적인 핵심 정보는 숨기면 안됩니다. 남겨야할 핵심 데이터와 세부 구현 단계를 구분하여, 핵심 데이터는 남기고 세부 구현을 숨기는게 좋습니다. 

응집도 정도에 따라, 다음 두 가지 프로그래밍 패러다임이 있습니다.

- **선언적 프로그래밍:** 핵심 데이터만 전달받고 세부 구현은 뭉쳐 숨겨 두는 개발 스타일

- **명령형 프로그래밍:** 어떻게 해야 할지 하나하나 명령하기

<br/>


### **2. 단일책임**

하나의 일을 하는 뚜렷한 이름의 함수를 만들어야 합니다. 아래는 함수명을 잘못 지은 예시입니다.

<div style="text-align:center"><img src="https://user-images.githubusercontent.com/71360682/119297798-a4be8e00-bc96-11eb-97b9-00a69efe1c1a.png" /></div>

<div align="center">
  <i> 함수명이 기능을 모두 포함하지 않는 예시 (bad) </i>
</div>

<br/>

이를 해결하기 위해, 한 가지 일만 하는 명확한 이름의 함수가 되도록 리펙토링 하였습니다.

<div style="text-align:center"><img src="https://user-images.githubusercontent.com/71360682/119297958-f8c97280-bc96-11eb-8ec9-a1259e64c262.png" /></div>


<div align="center">
  <i> 리펙토링1. 한 가지 일만 하는 함수로 분리 </i>
</div>

<br/>

다음 예시는 버튼 클릭 함수의 기능을 분리하기 위해 새로운 컴포넌트를 만들어 감싸여 리펙토링 한 경우입니다.

<div style="text-align:center"><img src="https://user-images.githubusercontent.com/71360682/119298127-5c53a000-bc97-11eb-919d-c558a78d6bc4.png" /></div>


<div align="center">
  <i> 리펙토링2. 한 가지 일만 하는 기능성 컴포넌트 도입</i>
</div>

<br/>



## 참고자료 

[1] [토스ㅣSLASH 21 - 실무에서 바로 쓰는 Frontend Clean Code](https://www.youtube.com/watch?v=edWbHp_k_9Y)
