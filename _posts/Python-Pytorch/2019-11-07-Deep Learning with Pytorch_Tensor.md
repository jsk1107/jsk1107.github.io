---
layout: post
title: Tutorial[1] - Tensor & Demension Handling & CUDA
comments: true
categories : [Pytorch, Tutorials]
use_math: true
---

딥러닝의 이론적인 내용은 최대한 배제하고 Pytorch를 활용하여 코딩할 수 있도록 필수로 알아야하는 내용에 대해서만 초점을 두었습니다. 

Pytorch 1.4 버전을 기준으로 공식 홈페이지에 나와있는 튜토리얼을 참고하였으며, 데이터 로드, CNN모델 설계, 학습, 추론까지의 일련의 과정을 경험해보는것이 Tutorial의 목표입니다.

<hr>

## Pytorch Install

[Pytorch 홈페이지](https://pytorch.org) → Get Started → 자신의 PC 사양에 맞는 환경을 설정하면 Command를 알려줍니다. 현재 저의 PC는 Ubuntu18.04LTS와 Python3를 사용하고 있습니다.(Not conda)

```Shell
pip install torch torchvision
```
<hr>

## Tensor
우선 Tensor가 무엇인지부터 알고 넘어가야한다. Tensor는 물리와 수학에서 시작된것으로 어떠한 변환을 하더라도 변하지 않는것이다. 간단하게는 벡터 계산을 좀더 쉽게 보기위해 행렬로 표기한것이며 Python에서는 ndarray와 유사하다. 코딩을 할때는 그냥 행렬의 집합이구나 정도로 생각하시면 된다.

<br>

### Tensor 선언 : 

```python
x = torch.Tensor(3) # 3개의 스칼라값을 가진 Tensor
print(x)
print(x.dtype)
```
Out : 
```
tensor([1.0561e-38, 1.0286e-38, 1.6956e-43])
torch.float32
```

변수 x의 타입이 Tensor라는것을 눈여겨 보자. dtype은 기본으로 float32이다.
<br>

### 3x3 행렬 생성하기 :
```python
x = torch.Tensor(3, 3) # 3x3 행렬의 Tensor
print(x)
```
Out:
```
tensor([[1.0653e-38, 1.0194e-38, 4.6838e-39],
        [8.4489e-39, 9.6429e-39, 8.4490e-39],
        [9.6429e-39, 9.2755e-39, 1.0286e-38]])
```
<br>

#### 랜덤하게 초기화된 3x3 행렬 생성하기 :

```python
x = torch.rand(3, 3)
print(x)
```
Out:
```
tensor([[0.3508, 0.7147, 0.2823],
        [0.9523, 0.1514, 0.8020],
        [0.7782, 0.4528, 0.8484]])
```

Numpy에서 사용하는 np.random.rand() 함수와 똑같다.
<br>

### Tensor에서 Vaule Return하기 :

```python
x = torch.randn(1) # 1차원 Tensor만 가능
print(x.item())
```
Out:
```
0.6625566482543945
```

<hr>

## Transform Tensor

Tensor의 데이터타입은 Tensor이다. 하지만 전처리를 하는 과정에서는 Tensor를 Numpy와 같은 자료로 변환해야할 경우(혹은 반대의 경우)도 있으며, 차원을 바꾸어야 하는 경우도 있다.

<br>

### 데이터타입 변환 : 
Tensor -> Numpy
```python
a = torch.rand(3, 3) # a는 tensor
b = a.numpy() # b는 numpy
print(b)
```
Out:
```
[[0.36377007 0.3272568  0.12410921]
 [0.19085902 0.4008761  0.4487794 ]
 [0.27390736 0.01607198 0.60198116]]
```

Numpy -> Tensor
```python
a = np.array([[3, 1], [2, 4]])
b = torch.as_tensor(a)
c = torch.Tensor(a)
d = torch.from_numpy(a)
print(b, b.dtype)
print(c, c.dtype)
print(d, d.dtype)
```
Out:
```
tensor([[3, 1],
        [2, 4]]) torch.int64
tensor([[3., 1.],
        [2., 4.]]) torch.float32
tensor([[3, 1],
        [2, 4]]) torch.int64
```
<br>

Tensor로의 변환은 3가지 방법이 있다. as_tensor, from_numpy 메서드는 기존 numpy의 dtype을 따라가며 Tensor객체를 사용한 변환은 float32로 자동 형변환을 해준다.

### 차원변환 : 

```python
a = torch.rand(3, 3) # 3x3 행혈 생성
a = a.view(1, 1, 3, 3) # 4차원 변환
print(a.shape)
```
Out:
```
torch.Size([1, 1, 3, 3])
```

a의 전체 size가 9이므로 변형될 size도 9가 되어야 한다. 그렇지 않으면 Error 발생.
<br>

### Tensor 합치기 : 

```python
a = torch.randn(4, 3)
b = torch.randn(5, 3)
c = torch.cat((a, b), dim=0) # dim: 0이면 0번째 차원으로 합치기. 1이면 첫번째 차원으로 합치기
print(c.shape)
```
Out:
```
torch.Size([9, 3])
```

위에서 공부한것을 토대로 a와 b의 size는 각각 `torch.shape([4, 3])`, `torch.shape([5, 3])` 이라는 것을 알 수 있다. dim이 0이므로 a에서 0번째 차원인 4, b에서 0번째 차원인 5가 합쳐지는 것이다. 

4차원인 경우를 한번 살펴보자.

```python
a = torch.randn(2, 3, 5, 1)
b = torch.randn(3, 3, 5, 1)
c = torch.cat((a, b), dim=0)
print(c.shape)
```

Out:
```
torch.Size([5, 3, 5, 1])
```

이해가 되었는가?? 하지만 여기서 dim을 1로 바꾸면 Error가 발생한다.

Out:
```python
RuntimeError: invalid argument 0: Sizes of tensors must match except in dimension 1. Got 2 and 3 in dimension 0 at /pytorch/aten/src/TH/generic/THTensor.cpp:612
```

dim이 1이면 첫번째 차원을 제외한 나머지 차원의 size가 매칭되야만 한다는 에러메세지를 뿜너낸다. 즉, 묶을 차원을 제외한 나머지 행렬의 모양이 똑같아야만 Error가 발생하지 않는다.

### 선택적 차원 늘리고 줄이기 :
```python
a = torch.rand(3, 3) # 3x3 행렬 생성
b = a.unsqueeze(dim=2) # 2번째 index에 차원을 늘리기
c = a.unsqueeze(dim=0) # 0번째 index에 차원을 늘리기
print(b.shape)
print(c.shape)
```
Out:
```
torch.Size([3, 3, 1])
torch.Size([1, 3, 3])
```

위에서 배웠던 `view를` 이용해도 무관하다. 하지만 왜인지모르게 `unsqueeze` 함수를 더 많이 사용한다. Numpy에서도 `expand_dim` 이라는 함수로 동일한 기능을 할 수 있다.

```python
a = torch.rand(1, 3, 3) # 3x3 행렬 생성
b = a.squeeze(dim=0) # 0번째 index에 차원을 축소하기
print(b.shape)
```
Out:
```
torch.Size([3, 3])
```

`unsqueeze와` 달리 `squeeze는` 차원을 축소한다. 여기서 중요한 뽀인트는 행렬의 차원이 **1** 이라고 되어있는 차원만 축소가 가능하다. 즉, 2이상의 형태를 가진 차원에서는 축소할 수 없다.

### 차원의 순서 바꾸기 : 
```python
a = torch.rand(3, 3, 1, 5)
b = a.transpose(0, 3)
c = a.permute(3, 1, 2, 0)
print(b.shape)
print(c.shape)
```
Out:
```
torch.Size([5, 3, 1, 3])
torch.Size([5, 3, 1, 3])
```

`transpose` 메서드는 바꿀 차원의 index만 적어준다. 위의 예시에는 0번째 차원과 3번째 차원의 순서를 바꾼다는 뜻.

`permute`는 바꿀 차원의 전체 index를 순서에 맞게 적어주면 된다. 위의 예시는 (3, 3, 1, 5) 형상을 지닌 행렬을 (5, 3, 1, 3)로 변경한것이다. 즉 0번째 차원과 3번째 차원의 순서를 바꾼것이다.
<hr>

## CUDA

CUDA 프로세서를 이용해서 GPU연산을 하는 방법이다. 각 Tensor에 **.cuda**를 직접 호출하거나, **.to** 메서드를 호출하여 device를 설정하여 사용할 수 있다.

<br>

### cuda 메서드 호출 : 
```python
x = torch.rand(3, 3)
y = torch.rand(3, 3)

if torch.cuda.is_available(): # cuda 사용가능하면 True를 리턴
    s = x.cuda() + y.cuda() # 각 Tensor에서 cuda 메서드를 호출
```
<br>

### to 메서드 호출 : 
```python
x = torch.rand(3, 3)
y = torch.rand(3, 3)
if torch.cuda.is_available():
    device = torch.device("cuda")
    x = x.to(device)                       
    y = y.to(device)
    z = x + y
print(z)
print(z.to("cpu", torch.double))
```
Out:
```
tensor([-0.0658], device='cuda:0') # CUDA 적용된 Tensor
tensor([-0.0658], dtype=torch.float64) # CPU 적용된 Tensor
```

아마 대부분의 유저들이 GPU를 사용할 것이기 때문에 무조건 cuda를 붙이는것은 권한다.

하지만 Inference를 할때, 혹은 엣지컴퓨팅 등의 디바이스에서 실행할 경우도 있기 때문에 잘 작성된 코드를 보면 CPU, GPU 모든 환경에서 실행 가능하도록 코딩이 된것을 심심찮게 볼 수 있을 것이다.
<hr>

