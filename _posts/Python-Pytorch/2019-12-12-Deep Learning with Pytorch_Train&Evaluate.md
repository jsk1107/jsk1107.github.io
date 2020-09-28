---
layout: post
title: Tutorial[6] - Loss Function & Optimizer & Scheduler
comments: true
categories : [Pytorch, Tutorials]
use_math: true
---

딥러닝의 이론적인 내용은 최대한 배제하고 Pytorch를 활용하여 코딩할 수 있도록 필수로 알아야하는 내용에 대해서만 초점을 두었습니다. 

Pytorch 1.4 버전을 기준으로 공식 홈페이지에 나와있는 튜토리얼을 참고하였으며, 데이터 로드, CNN모델 설계, 학습, 추론까지의 일련의 과정을 경험해보는것이 Tutorial의 목표입니다.

<hr>

## Loss Function & Optimizer & Scheduler

모델 설계가 끝난 후 바로 해야할것은 손실함수와 최적화함수를 정하는 것이다. 의외로 많은 초보자들이 이 부분을 코딩하지 않고 그냥 넘어간다(나도 그랬다). 그러니 아예 암기를 해버리도록 하자. 모델 -> Loss -> Optimizer -> Scheduler(선택)는 반.드.시. 순서대로 설계되어야 한다.

<hr>

### Loss_function

손실함수는 존나 많다. 심지어 나도 석사 졸업논문으로 손실함수를 하나 만들었다. 하지만 겁먹지마라. 함수의 기본적인 뼈대는 비슷비슷하기 때문이다.

이론적인 내용은 생략하고 지금부터 그 기본적인 뼈대가 되는 함수 2가지의 활용법을 알아보자.

- `torch.nn.CrossEntropyLoss()` : CrossEntropy(이하 CE)는 거의 모든 Loss function의 기본이 된다. 코딩을 할때도 가장 많이 사용한다. 받을 수 있는 인자는 세가지가 있다.
    1. `weight` : Class가 불균형일때 사용한다. 입력되는 값은 의견이 제각각이지만 공통적인 의견은 정규화(Normalize)를 하라는것이다. 예를들어, 0: 300, 1: 2000, 2: 700 이라면, weight = Tensor([(1/300)/3000, 2000, 3000, 700/3000]). [discuss.pytorch](https://discuss.pytorch.org/t/what-is-the-weight-values-mean-in-torch-nn-crossentropyloss/11455/10)에서는 특정한 공식을 사용해서 좋은 결과를 얻었다고 하니 참고해보자.

    2. `ignore_index` :  특정 Class에 대하여 Loss 계산을 제외하겠다는 인자이다. Detection이나 Segmentation에서 사용자가 특정 Class를 제외하고 싶을때 사용하면 편하다. 원래대로라면 Model의 마지막 Layer인 FC Layer부분의 채널을 변경하고 원-핫 인코딩을 다시 해주어야한다. 하지만 해당 인자를 사용하면 이러한 번거로움을 줄일 수 있다.

    3. `reduction` : 인자로 줄 수 있는것은 'sum', 'mean', None' 3가지가 있다. 사실 이거는 나도 잘 모르겠다. 이 인자는 쓰지 말고 넘어가도록 하자.

    - 또 하나의 특징으로는, CE의 수식을 보면 p(x) 에 대한 부분이 명시 되어있다. 즉 Softmax를 거쳐나온 확률을 쓰는것인데, 이는 CE 클래스 내부에 탑재되어있으며, 자동으로 계산된다.

    ```python
    # 대부분의 경우 인자는 사용하지 않는다.
    criterion = torch.nn.CrossEntropyLoss()
    ```

- `torch.nn.Softmax()` : `dim`이라는 인자가 있기는 하지만 딱히 고려하지 않아도 된다. 왜냐하면 Softmax 연산할때 output으로 나온 모든 Tensor를 더하기 때문에 dim은 무조건 1차원이 되기 때문이다.
    ```python
    # 대부분의 경우 인자는 사용하지 않는다.
    softmax = torch.nn.Softmax()
    ```

몇년전만해도 L1 Loss가 활발히 쓰였으나, 이젠 더이상 사용하지 않는 퇴물이 되었으니 안봐도 된다.


### Optimizer

최적화 또한 Loss Function처럼 존나 많다. 여기서도 가장 대표적으로 사용되는 3가지 Optimizer를 알아보도록 한다.

Optimizer는 `torch.optim` 클래스에 들어있다. 당연한 말이지만 Model의 가중치를 최적화 하는것이기 때문에 사용자가 생성한 신경망의 가중치를 뽑아서 (`model.parameters()`) 인자로 넣어주어야 한다. 설명하지 않은 옵션에 대해서는 굳이 알지 않아도 된다.

- `torch.optim.SGD()` : CE처럼 가장 많이 사용되는 최적화 방법이다. 
    1. `params` : 신경망에서 뽑아낸 가중치. `model.parameters()`을 인자로 받는다.
    2. `lr` : 초기 학습률을 인자로 받는다.
    3. `momentum` : 학습 방향을 고려해주는 값이다. 보통 0.9를 사용.
    4. `weight_decay` : 가중치에 L2 패널티를 부여한 정규화를 한다. default는 0. 사용시에는 보통 0.1을 사용.

    ```python
    optimizer = torch.optim.SGD(params=model.parameters(),
                                lr=1e-04,
                                momentum=0.9
                                weight_decay=0.1)
    ```

- `torch.optim.Adam()` : SGD와 더불어 최근 가장 많이 사용하고 있는 최적화 방법이다.
    1. `params` : 신경망에서 뽑아낸 가중치. `model.parameters()`을 인자로 받는다.
    2. `lr` : 초기 학습률을 인자로 받는다.
    - 이것 이외에도 자질구래한 인자가 있지만, 사용한것은 단 한번도 못보았으니 그냥 넘어가도 된다.

    ```python
    optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-04)
    ```
- `torch.optim.RMSprop()` : 유명한 힌튼(Hinton) 교수님이 제안하신 방법.
    1. `params` : 신경망에서 뽑아낸 가중치. `model.parameters()`을 인자로 받는다.
    2. `lr` : 초기 학습률을 인자로 받는다.
    ```python
    optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-04)
    ```

### LR_Scheduler

Optimizer를 설정한 후 가장 중요한것이 LR_Scheduler이다. 굳이 이것을 설정하지 않아도 상관없지만, 보다 좋은 정확도를 뽑아내기 위해서는 반드시 설정해주어야 한다. Scheduler에는 4가지 방법이 가장 많이 사용된다. 설명하지 않은 옵션에 대해서는 굳이 알지 않아도 된다.

- `torch.optim.lr_scheduler.StepLR()` : 사용자가 설정한 값에 Epoch이 도달하게 되면 lr값을 조절한다.
    1. `optimizer` : Optimizer 함수를 인자로 받는다.
    2. `step_size` : 몇 Epoch에 lr값을 조절할지 정한다. 20 Epoch이 되었을때 lr을 감소 시키고 싶으면 step_size=20이 된다.
    3. `gamma` : lr이 감소될때 얼마만큼 조절할지 감소량을 정한다. gamma=0.1로 설정하면, 현재 lr에 0.1을 곱한값이 새로운 lr이 된다.
    
    ```python
    scheduler = torch.optim.lr_scheduler.StepLR()
    ```
- `torch.optim.lr_scheduler.MultiStepLR()` : StepLR 함수와 똑같은 기능을 하지만, 사용자가 여러개 설정값을 줄 수 있다.
    1. `optimizer` : Optimizer 함수를 인자로 받는다.
    2. `milestones` : 몇 Epoch에 lr값을 조절할지 정한다. 여러개의 인자를 받기 때문에 List 타입을 받는다. 20 Epoch, 50 Epoch에 lr을 감소시키고 싶으면, milestones=[20, 50]이 된다.
    3. `gamma` : lr이 감소될때 얼마만큼 조절할지 감소량을 정한다.

- `torch.optim.lr_scheduler.ExponentialLR()` : 매 Epoch마다 lr값을 조절한다.
    1. `optimizer` : Optimizer 함수를 인자로 받는다.
    2. `gamma` : lr이 감소될때 얼마만큼 조절할지 감소량을 정한다.

- `torch.optim.lr_scheduler.ReduceLROnPlateau()` : Loss가 더이상 감소되지 않을때마다 lr값을 조절한다.
    1. `optimizer` : Optimizer 함수를 인자로 받는다.
    2. `mode` : 'min', 'max' 에서 선택할 수 있다. min은 모니터링되고 있는 수의 감소가 멈출때 lr을 조절한다. 반대로 max는 모니터링되고 있는 수의 상승이 멈출때 lr을 조절한다. 여기서 모니터링되고 있는 수 라는것은 Loss값을 받아와서 내부에서 어떠한 수식을 통해 산출되는 값인것 같다. default로 설정되어있는 min을 쓰는것이 정신건강에 좋을듯 하다.
    3. `factor` : lr이 감소될때 얼마만큼 조절할지 감소량을 정한다. gamma 인자와 같은 기능을 한다.
    4. `patience` : 더이상 성능이 개선되지 않았을때로부터 사용자가 지정한 Epoch까지는 값을 변화시키지 않는다. 예를들어 patience=5라고 했을때, Loss가 5 Epoch이후에도 감소하지 않는다면 그때 lr값을 감소시킨다.

