---
title: '[JavaScript] 이진 트리(Binary Tree)와 트리 순회(Tree Traversal)'
date: 2021-03-25 09:00:00 -0400
categories: algorithm
tags: [algorithm, data-structure, javascript, tree, binary-tree]
---

이번 글에서는 **이진 트리(Binary Tree)**와 **트리 순회(Tree Traversal)**에 대해서 알아보고, `JavaScript`를 이용해서 구현해보겠습니다.

## 그래프(Graph)

- `노드(node)`들과 노드들 사이를 연결하는 `간선(edge)`으로 구성되어 있습니다.
- 그래프는 `root node`가 하나 있고, 각 노드에는 `child node`가 연결되어 있습니다.

## 트리(Tree)

- `트리`는 그래프의 일종으로, `cycle`이 없고, **서로 다른 두 노드를 잇는 길이 하나 뿐인 그래프**를 트리라고 합니다.
- 노드가 `N개`인 트리는 항상 `N-1개`의 간선을 가집니다.
- `child`의 갯수가 2개로 제한되면 **이진 트리(Binary Tree)**라고 합니다.

## 이진 트리의 종류

- **Full Binary Tree**: 각각의 노드가 `child`가 0개 혹은 2개
- **Complete Binary Tree**: 왼쪽 위에서부터 가득 차 있는 트리
- **Perfect Binary Tree**: 모든 내부 노드가 2개의 `children`을 가지고 있으며, `leaf node`의 `level`이 같은 트리

## 이진 트리 순회 알고리즘(Binary Tree Traversal)

**이진 트리 순회 알고리즘**은 트리에 저장된 모든 값을 중복이나 빠짐없이 살펴보고 싶을 때 사용합니다. 이진 트리의 순회 방법 중 **깊이 우선 순회 방법(Depth First Traversal)**으로는 `전위 순회(Pre-order traversal)`, `정위 순회(In-order traversal)`, `후위 순회(Post-order traversal)`가 있으며, **너비 우선 순회 방법(Breadth First Traversal)**으로는 `레벨 순회`가 있습니다.

<div style="text-align:center"><img src="https://user-images.githubusercontent.com/71360682/112273731-05464480-8cc1-11eb-9316-831b34246be2.png" /></div>

<div align="center">
  <i>Binary Tree 1 (from 코드없는프로그래밍)</i>
</div>
<br/>

- `Pre-order`: **N**LR
- `In-order`: L**N**R
- `Post-order`: LR**N**
- `Level-order`: **N**LR

<div style="text-align:center"><img src="https://user-images.githubusercontent.com/71360682/112273743-0a0af880-8cc1-11eb-9953-1bf855e4dd17.png" /></div>

<div align="center">
  <i>Binary Tree 2 (from 코드없는프로그래밍)</i>
</div>
<br/>

- `Pre-order`: 1 2 4 5 3 6 7
- `In-order`: 4 2 5 1 6 3 7
- `Post-order`: 4 5 2 6 7 3 1
- `Level-order`: 1 2 3 4 5 6 7

## 이진 트리 순회 알고리즘의 구현

## 재귀적(Recursive) 방법

이진 트리 순회 방법 중 `깊이 우선 순회 방법(BFS)`은 **재귀적(Recursive)** 혹은 **반복적(Iterative)** 방법으로 구현할 수 있습니다. 먼저 재귀적인 방법으로 구현해보겠습니다.

### 트리 정의하기

```javascript
class Tree {
  constructor(val) {
    this.val = val;
    this.leftNode = null;
    this.rightNode = null;
  }

  setVal(val) {
    this.val = val;
  }

  setLeft(node) {
    this.leftNode = node;
  }

  setRight(node) {
    this.rightNode = node;
  }
}
```

### 전위 순회(Pre-order)
```javascript
var recursivePreOrder = function (node) {
  if (!node) {
    return;
  }
  console.log(node.val);
  this.recursivePreOrder(node.leftNode);
  this.recursivePreOrder(node.rightNode);
};
```

### 정위 순회(In-order)
```javascript
var recursiveInOrder = function (node) {
  if (!node) {
    return;
  }
  this.recursiveInOrder(node.leftNode);
  console.log(node.val);
  this.recursiveInOrder(node.rightNode);
};
```

### 후위 순회(Post-order)
```javascript
var recursivePostOrder = function (node) {
  if (!node) {
    return;
  }
  this.recursivePostOrder(node.leftNode);
  this.recursivePostOrder(node.rightNode);
  console.log(node.val);
};
```

## 반복적(Iterative) 방법

반복적인 방법으로 구현할 때는 **스택(stack)**을 사용합니다. 먼저 그림을 살펴보고, 이를 코드로 구현하겠습니다.

### 전위 순회(Pre-order)

<div style="text-align:center"><img src="https://camo.githubusercontent.com/6ca60eb809d07ee410ae860bbdbc3c92032b7853142dea79d5970c1847433bf1/687474703a2f2f3130382e36312e3131392e31322f77702d636f6e74656e742f75706c6f6164732f323031342f31302f62696e6172792d747265652d312d7072652d6f726465722d736d616c6c2e676966"/></div>


<div align="center">
  <i>Pre-order traversal from http://ejklike.github.io/</i>
</div>
<br/>

```javascript
var iterativePreOrder = function (node) {
  if (node == null) {
    return;
  }
  let stack = [];
  stack.push(node);
  while (stack.length > 0) {
    let pop_node = stack.pop();
    console.log(pop_node.val);
    if (pop_node.right) stack.push(pop_node.right);
    if (pop_node.left) stack.push(pop_node.left);
  }
};
```

<div align="center">
  <i>Pre-order traversal</i>
</div>
<br/>


<div style="text-align:center"><img src="https://camo.githubusercontent.com/7041073b508c5c7768c8bacebfe9cb2d0ee994070912a58bbbc9835cdae85ed0/687474703a2f2f3130382e36312e3131392e31322f77702d636f6e74656e742f75706c6f6164732f323031342f31302f62696e6172792d747265652d312d6f726465722d736d616c6c2e676966"/></div>

<div align="center">
  <i>In-order traversal from http://ejklike.github.io/</i>
</div>
<br/>

```javascript
var iterativeInOrder = function (node) {
  if (node == null) {
    return;
  }
  let crnt_node = node;
  let stack = [];
  while (true) {
    if (crnt_node != null) {
      stack.push(crnt_node);
      crnt_node = crnt_node.left;
    } else if (stack.length > 0) {
      crnt_node = stack.pop();
      console.log(crnt_node.val);
      crnt_node = crnt_node.right;
    } else {
      break;
    }
  }
};
```

<div align="center">
  <i>In-order traversal</i>
</div>
<br/>

<div style="text-align:center"><img src="https://camo.githubusercontent.com/6ca60eb809d07ee410ae860bbdbc3c92032b7853142dea79d5970c1847433bf1/687474703a2f2f3130382e36312e3131392e31322f77702d636f6e74656e742f75706c6f6164732f323031342f31302f62696e6172792d747265652d312d7072652d6f726465722d736d616c6c2e676966"/></div>


<div align="center">
  <i>Post-order traversal from http://ejklike.github.io/</i>
</div>

```javascript
var iterativePostOrder = function (node) {
  if (node == null) {
    return;
  }
  let crnt_node = node;
  let stack = [];
  let last_visit_node = null;
  while (true) {
    if (crnt_node != null) {
      stack.push(crnt_node);
      crnt_node = crnt_node.left;
    } else if (stack.length > 0) {
      peek_node = stack[stack.length - 1];
      if (peek_node.right != null && last_visit_node != peek_node.right) {
        crnt_node = peek_node.right;
      } else {
        console.log(peek_node.val);
        last_visit_node = stack.pop();
      }
    } else {
      break;
    }
  }
};
```

<div align="center">
  <i>Post-order traversal</i>
</div>
<br/>


## 너비 우선 순회 방법(BFS)

이진 트리의 `너비 우선 순회`에는 **레벨 순회**가 있습니다. **큐(queue)** 자료구조를 사용하면 간단히 구현할 수 있습니다.

```javascript
var levelOrderTraversal = function (node) {
  if (node == null) {
    return;
  }
  let queue = [];
  queue.push(node);
  while (queue.length > 0) {
    let pop_node = queue.shift();
    console.log(pop_node.val);
    if (pop_node.left) queue.push(pop_node.left);
    if (pop_node.right) queue.push(pop_node.right);
  }
};

levelOrderTraversal(root)
```

## 문제풀이 1. Path Sum II [(LeetCode)](https://leetcode.com/problems/path-sum-ii/description/)

Given the `root` of a binary tree and an integer `targetSum`, return all **root-to-leaf** paths where each path's sum equals `targetSum`.

(해석) 루트 노드와 정수 `targetSum`이 주여질 때, 루트 노드에서 `leaf`까지의 `path`가 지나는 노드의 합이 `targetSum`이 되도록 하는 모든 `path`를 찾아라.

### Solution


```javascript
var pathSum = function (root, targetSum) {
  if (root == null) {
    return [];
  }
  let result = [];

  var repeat = function (node, path, residual) {
    if (!node) return;
    path.push(node.val);
    residual -= node.val;
    if (residual == 0 && !node.left && !node.right) result.push(Array.from(path));
    repeat(node.left, path, residual);
    repeat(node.right, path, residual);
    path.pop();
  };
  repeat(root, [], targetSum);
  return result;
};
```



## 관련문항 (LeetCode)

[Path Sum II](https://leetcode.com/problems/path-sum-ii/description/)

[Binary Tree Inorder Traversal](https://leetcode.com/problems/binary-tree-inorder-traversal/)

## 참고자료

[1] [Inorder Preorder Postorder Traversal of Binary Tree](https://laptrinhx.com/inorder-preorder-postorder-traversal-of-binary-tree-3322436720/)

[2] [[자료구조] Javascript로 Tree와 Tree 순회 구현하기](https://gogomalibu.tistory.com/55)

[3] [코딩테스트, 기초, 트리, Tree 소개](https://www.youtube.com/watch?v=bOZhvOc5xlQ&list=PLDV-cCQnUlIaTA41swrZwgH4mX7iPxLH4&index=1)

[4] [파이썬을 사용한 이진 트리와 순회 알고리즘 구현 (2)](http://ejklike.github.io/2018/01/09/traversing-a-binary-tree-2.html)

[5] [113. Path Sum II](https://baffinlee.com/leetcode-javascript/problem/path-sum-ii.html)
