---
title: '[React] 유용한 Custom hook 만들기'
date: 2021-04-13 00:00:00 -0400
categories: web-development
tags: [react, react-hook, web-development, custom-hook, client]
---


**Custom hook**은 React에서 제공하는 hook을 활용하여 사용자가 원하는 기능을 수행하도록 만든 함수를 의미합니다. 이를 통해 hook을 포함한 반복적인 작업을 다른 component에서 쉽게 사용할 수 있게 해줍니다.

**Custom Hook**의 이름은 반드시 **use**로 시작해야 하는데, 그 이유는 다음과 같습니다.

 - 한눈에 보아도 **Hook 규칙**이 적용되는지를 파악할 수 있습니다.
 - `use`로 시작하면 **Hook 규칙의 위반 여부**를 react가 자동으로 체크합니다.



## Hook의 규칙

이름이 `use`로 시작하는 함수가 지켜야하는 **Hook 규칙**은 다음과 같습니다.

- **최상위**(at the Top Level)에서만 Hook을 호출해야 합니다

- 오직 **React 함수 내**에서 Hook을 호출해야 합니다

<br/>

## 실습 1. useState

먼저 `custom hook`을 사용해서 `useState`와 같은 기능을 하는 가장 기본적인 hook을 만들어 보겠습니다.

```javascript
import { useState } from "react";

// useState와 기능이 같은 custom hook
export default function useBasicHook(initialValue) {
  const [value, setValue] = useState(initialValue);
  return [value, setValue];
}
```

`useBasicHook`의 기능은 `useState`의 기능과 일치합니다.

<br/>


## 실습 2. useLocalStorage

다음으로 **Local Storage**에 state를 저장하고, 저장된 값이 있다면  key를 통해 불러오는 custom hook을 생성해보겠습니다.

```javascript
import { useState, useEffect } from "react";

// key에 해당하는 값이 있으면 그 값을, 아니면 initialValue를 리턴
function getSavedValue(key, initialValue) {
  console.log("rendering");
  const savedValue = JSON.parse(localStorage.getItem(key));
  if (savedValue) return savedValue;

  if (initialValue instanceof Function) return initialValue();
  return initialValue;
}

export default function useLocalStorage(key, initialValue) {
  const [value, setValue] = useState(() => {
    return getSavedValue(key, initialValue);
  });

  // (Tip) 아래와 같이 선언하면 rendering시 매번 useState 실행되므로 함수형으로 써야한다.
  // const [value, setValue] = useState(getSavedValue(key, initialValue));

  // local storage의 key에 value를 저장
  useEffect(() => {
    localStorage.setItem(key, JSON.stringify(value));
  }, [key, value]);

  return [value, setValue];
}
```

<mark> TIP 1.</mark> `useState`를 사용할 때, initialValue를 **함수형**으로 입력하면 부모 component(useLocalStorage)가 rerender 하더라도 `useState`는 단 한번만 실행이 됩니다. 

<br/>


## 참고자료 

[1] [자신만의 Hook 만들기](https://ko.reactjs.org/docs/hooks-custom.html)

[2] [Learn Custom Hooks In 10 Minutes](https://www.youtube.com/watch?v=6ThXsUwLWvc)
