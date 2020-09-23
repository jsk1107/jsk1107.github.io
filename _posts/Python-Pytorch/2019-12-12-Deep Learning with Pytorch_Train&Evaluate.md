---
layout: post
title: Tutorial [6] Loss Function & Optimizer
comments: true
categories : [Pytorch, Tutorials]
use_math: true
---


딥러닝의 이론적인 내용은 최대한 배제하고 Pytorch를 활용하여 코딩할 수 있도록 필수로 알아야하는 내용에 대해서만 초점을 두었습니다. 

Pytorch 1.4 버전을 기준으로 공식 홈페이지에 나와있는 튜토리얼을 참고하였으며, 데이터 로드, CNN모델 설계 학습, 추론까지의 일련의 과정을 경험해보는것이 Tutorial의 목표입니다.

<hr>

## Loss Function & Optimizer

모델 설계가 끝난 후 바로 해야할것은 손실함수와 최적화함수를 정하는 것이다. 의외로 많은 초보자들이 이 부분을 코딩하지 않고 그냥 넘어간다(나도 그랬다). 그러니 아예 암기를 해버리도록 하자. 모델 -> Loss -> Optimizer는 반.드.시. 순서대로 설계되어야 한다.

<hr>

### Loss_function

손실함수는 존나 많다. 심지어 나도 석사 졸업논문으로 손실함수를 하나 만들었다. 하지만 겁먹지마라. 함수의 기본적인 뼈대는 비슷비슷하기 때문이다.

이론적인 내용은 생략하고 지금부터 그 기본적인 뼈대가 되는 함수 2가지의 활용법을 알아보자.

- `torch.nn.CrossEntropyLoss()` : CrossEntropy(이하 CE)는 거의 모든 Loss function의 기본이 된다. 코딩을 할때도 가장 많이 사용한다. 받을 수 있는 인자는 세가지가 있다.
    1. `weight` : Class가 불균형일때 사용한다. 입력되는 값은 의견이 제각각이지만 공통적인 의견은 정규화(Normalize)를 하라는것이다. 예를들어, 0: 300, 1: 2000, 2: 700 이라면, weight = Tensor([(1/300)/3000, 2000, 3000, 700/3000]). [discuss.pytorch](https://discuss.pytorch.org/t/what-is-the-weight-values-mean-in-torch-nn-crossentropyloss/11455/10)에서는 특정한 공식을 사용해서 좋은 결과를 얻었다고 하니 참고해보자.

    2. `ignore_index` :  특정 Class에 대하여 Loss 계산을 제외하겠다는 인자이다. Detection이나 Segmentation에서 사용자가 특정 Class를 제외하고 싶을때 사용하면 편하다. 원래대로라면 Model의 마지막 Layer인 FC Layer부분의 채널을 변경하고 원-핫 인코딩을 다시 해주어야한다. 하지만 해당 인자를 사용하면 이러한 번거로움을 줄일 수 있다.

    3. `reduction` : 인자로 줄 수 있는것은 'sum', 'mean', None' 3가지가 있다. 사실 이거는 나도 잘 모르겠다. 이 인자는 쓰지 말고 넘어가도록 하자.

    - 또 하나의 특징으로는, CE의 수식을 보면 p(x) 에 대한 부분이 명시 되어있다. 즉 Softmax를 거쳐나온 확률을 쓰는것인데, 이는 CE 클래스 내부에 탑재되어있으며, 자동으로 계산된다.

- `torch.nn.Softmax()` : `dim`이라는 인자가 있기는 하지만 딱히 고려하지 않아도 된다. 왜냐하면 Softmax 연산할때 output으로 나온 모든 Tensor를 더하기 때문에 dim은 무조건 1차원이 되기 때문이다.

몇년전만해도 L1 Loss가 활발히 쓰였으나, 이젠 더이상 사용하지 않는 퇴물이 되었으니 안봐도 된다.

### Optimizer

최적화 또한 Loss Function처럼 존나 많다. 여기서도 가장 대표적으로 사용되는 4가지 Optimizer를 알아보도록 한다.

Optimizer는 `torch.optim` 클래스에 들어있다. 당연한 말이지만 Model의 가중치를 최적화 하는것이기 때문에 사용자가 생성한 신경망의 가중치를 뽑아서 (`model.parameters()`) 인자로 넣어주어야 한다.

- `torch.optim.SGD()` : CE처럼 가장 많이 사용되는 최적화 방법이다. 
    1. `params` : 신경망에서 뽑아낸 가중치. `model.parameters()`을 인자로 받는다.
    2. `lr` : 초기 학습률을 인자로 받는다.
    3. `momentum` : 학습 방향을 고려해주는 값이다. 보통 0.9를 사용.
    4. `weight_decay` : 가중치에 L2 패널티를 부여한 정규화를 한다. 보통 0.1을 사용.
- `torch.optim.Adam()` : SGD와 더불어 최근 가장 많이 사용하고 있는 최적화 방법이다.
    1. `params` : 신경망에서 뽑아낸 가중치. `model.parameters()`을 인자로 받는다.
    2. 
