---
layout: post
title: Tutorial [2] AUTOGRAD
comments: true
categories : [Pytorch, Tutorials]
use_math: true
---


딥러닝의 이론적인 내용은 최대한 배제하고 Pytorch를 활용하여 코딩할 수 있도록 필수로 알아야하는 내용에 대해서만 초점을 두었습니다. 

Pytorch 1.4 버전을 기준으로 공식 홈페이지에 나와있는 튜토리얼을 참고하였으며, 데이터 로드, CNN모델 설계 학습, 추론까지의 일련의 과정을 경험해보는것이 Tutorial의 목표입니다.

<hr>

## AUTOGRAD: AUTOMATIC DIFFERNETIATION

autograd를 이용하여 각 Tensor에 대하여 자동으로 미분을 수행한다. data, grad, grad_fn 등의 정보를 가지고 있으며 requires_grad라는 함수를 통해 작업을 추적할 수 있다.

__(torch.autograd.Variable은 0.4.0 버전 이후로는 더이상 사용하지 않습니다.)__

<br>

### autograd 추적하기 : 

Tensor에 추적기를 붙여놓는것과 같은 기능이다. requires_grad=True로 해놓지 않는다면 역전파를 수행할 수 없으니 반드시 True로 설정해야 한다.

```python
x = torch.ones(2, 2, requires_grad=True)
print(x)
```
Out:
```python
tensor([[1., 1.],
        [1., 1.]], requires_grad=True) # tensor에 requires_grad가 추가됨
```
<br>

### grad_fn : add

추적기가 붙어있는 Tensor에 대하여 어떠한 연산이 이루어져있는지 확인 할 수 있다.

```python
y = x + 2
print(y)
print(y.grad_fn) 
```
Out:
```python
tensor([[3., 3.],
        [3., 3.]], grad_fn=<AddBackward0>)
<AddBackward0 object at 0x7fac3a8d0080> # 어떠한 연산을 했는지 확인 가능
```

Tensor간 덧셈 연산이 있는 경우 grad_fn이 Add로 되어있음을 확인 할 수 있다.
<br>

### grad_fn : product, mean

```python
z = y * y * 3
out = z.mean()
print(z)
print(out)
```
Out:
```python
tensor([[27., 27.],
        [27., 27.]], grad_fn=<MulBackward0>)
tensor(27., grad_fn=<MeanBackward0>)
```

<br>

#### .requires_grad_ 옵션을 통해 requires_grad를 inplace로 교체 : 
```python
a = torch.randn(3, 3)
a = ((a*3) / (a-1))
print(a.requires_grad)
a.requires_grad_(True)
print(a.requires_grad)
b = (a*a).sum()
print(b.grad_fn)
```
Out:
```python
False # 첫번째 a는 연산만 되어있는 상태. 추적 불가능
True # 두번째 a는 requires_grad를 추가하였음. 추적 가능
<SumBackward0 object at 0x7fac3a8d0c18>
```
<br>

#### Gradients : $\frac{\partial out}{\partial x}

추적기가 붙어있는 모든 Tensor 노드들에 대해서 역전파를 수행한다.

```python
out.backward()
print(x.grad)
```
Out:
```python
tensor([[4.5000, 4.5000],
        [4.5000, 4.5000]])
```

<br>

### no_grad : 
추적을 중지하고 싶을 때 사용합니다. with구문과 함께 사용하며, 일반적으로 Infrerence를 할 때 사용한다.
```python
print(x.requires_grad)
print((x ** 2).requires_grad)
with torch.no_grad():
        print((x ** 2).requires_grad)
```
Out:
```python
True
True
False
```

일반적으로 Inference를 할때는 역전파를 할 이유가 저~~언혀 없기 때문에 requires_grad를 False로 해주어야 한다.

이것을 응용하면 Transfer Learning을 할때 특정 Layer의 Parameter만 변경하고 싶거나, FC Layer의 마지막 채널의 수를 변경할 때 유용하게 사용할 수 있다(추후 다루도록 하겠음).
<hr>