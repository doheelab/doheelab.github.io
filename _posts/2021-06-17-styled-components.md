---
title: '[CSS] styled-components 활용하기'
date: 2021-06-17 00:00:00
categories: css
tags: [css, react]
---

**styled-components**는 CSS를 React component나 HTML 태그에 적용하기 간편하게 적용하기 위한 기술입니다. 기본적인 사용 예시는 다음과 같습니다.

```javascript
import styled from 'styled-components'

// Create a Title component that'll render an <h1> tag with some styles
const Title = styled.h1`
  font-size: 1.5em;
  text-align: center;
  color: palevioletred;
`;
```

## props에 따라 스타일 변경하기

styled-components에 함수를 전달하면 **props에 따라 스타일을 변경**할 수 있습니다. 아래의 버튼은 `primary state`(true or false)에 따라서, 글자색과 배경색이 변경되도록 설정되었습니다.


```javascript
import styled from 'styled-components'

const Button = styled.button`
  /* Adapt the colors based on primary prop */
  background: ${props => props.primary ? "palevioletred" : "white"};
  color: ${props => props.primary ? "white" : "palevioletred"};

  font-size: 1em;
  margin: 1em;
  padding: 0.25em 1em;
  border: 2px solid palevioletred;
  border-radius: 3px;
`;

render(
  <div>
    <Button>Normal</Button>
    <Button primary>Primary</Button>
  </div>
);
```

## 다른 컴포넌트의 스타일 상속하기

styled-components를 통해 기존의 리엑트 컴포넌트의 스타일을 그대로 **상속**하고, 일부만 변경하여 사용할 수 있습니다.

```javascript

// The Button from the last section without the interpolations
const Button = styled.button`
  color: palevioletred;
  font-size: 1em;
  margin: 1em;
  padding: 0.25em 1em;
  border: 2px solid palevioletred;
  border-radius: 3px;
`;

// A new component based on Button, but with some override styles
const TomatoButton = styled(Button)`
  color: tomato;
  border-color: tomato;
`;

render(
  <div>
    <Button>Normal Button</Button>
    <TomatoButton>Tomato Button</TomatoButton>
  </div>
);
```

## ThemeProvider

**ThemeProvider**는 styled-components를 context API를 통해 component tree에 삽입하는 기술입니다. 이를 사용하면 모든 하위 콤포넌트에서 `props.theme`을 통해 **ThemeProvider**에서 정의된 속성을 사용할 수 있습니다.

```javascript
import styled, { ThemeProvider } from 'styled-components'

const Box = styled.div`
  color: ${props => props.theme.color};
`

const theme = { 
  color: 'mediumseagreen' 
}

render(
  <ThemeProvider theme={theme}>
    <Box>I'm mediumseagreen!</Box>
  </ThemeProvider>
)
```


## 참고자료

[[1] Styled Components 공식문서](https://styled-components.com/docs/basics#getting-started)

[[2] Styled Components 공식문서: ThemeProvider](https://styled-components.com/docs/api#themeprovider)