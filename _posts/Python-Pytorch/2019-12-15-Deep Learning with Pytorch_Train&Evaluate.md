---
layout: post
title: Tutorial [5] Train&Evaluate
comments: true
categories : [Pytorch, Tutorials]
use_math: true
---

딥러닝의 이론적인 내용은 최대한 배제하고 Pytorch를 활용하여 코딩할 수 있도록 필수로 알아야하는 내용에 대해서만 초점을 두었습니다. Pytorch 1.3.1 버전을 기준으로 공식 홈페이지에 나와있는 튜토리얼을 참고하였으며, 가장 기초적인 CNN모델인 ResNet을 활용하여 classification을 직접 설계해보는 단계까지가 Tutorial의 목표입니다.

<hr>

## Train

직접 수집한 데이터를 불러오는 단계까지 마무리가 되었습니다. 이제는 본격적으로 학습을 시작해보겠습니다. 학습 네트워크는 pytorch에서 제공해주는 model을 사용하도록 하고 각각의 개별적인 모델에 대해서는 추후 따로 다루도록 하겠습니다.

<hr>

#### model, loss_function, optim 설계

고려해야하는 네트워크의 순서는 다음과 같습니다.

1. 학습 네트워크
2. 손실함수
3. 최적화 방법

위 세가지를 고려하여 순서대로 코딩해주면 아래와 같습니다.

```python
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = torchvision.models.resnet50()
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-03)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=20, gamma=0.1)
```

* Line 1 : 학습을 시작할때 CPU자원을 사용할 것인지, GPU자원을 사용할 것인지를 체크합니다.
* Line 2 : 모델을 생성합니다. `torchvision.models`에는 Classification, Detection, segmentation 등 여러 종류의 Offical Model이 있습니다. 
* Line 3 : 손실함수(Loss Function)를 정합니다. `torch.nn`에 다양한 손실함수가 있습니다. 대표적으로 CrossEntropyLoss와 Softmax를 사용합니다.
* Line 4 : Optimizer 방법을 정합니다. `torch.optim`에 다양한 최적화 방법이 있으며, 입력되는 옵션들은 상이합니다. Adam의 경우에는 model에서 학습되는 가중치(parameter)를 입력해주어야 합니다. 대표적으로 Adam과 SGD를 사용합니다.
* Line 5 : Optimizer의 학습률(Learning_rate)을 스탭별로 조정할 수 있습니다. 위의 예시는 20 에폭마다 0.1의 **비율**로 줄이겠다는 의미입니다. 즉, 초기 lr이 1e-04(0.001)이므로 20 에폭이후의 lr은 1e-04(0.0001)가 됩니다.

<hr>

#### Train

학습의 시작은 DataLoader에서 return받은 변수에 for문을 돌려줍니다. 한번의 스탭마다 batch_size만큼의 img, label 정보가 dictionary형태로 들어있습니다. 우선 img, target, model에 device을 붙이고 model의 input으로 img를 입력합니다. ResNet50 네트워크를 타고 input으로 입력된 이미지가 마지막층까지 전달된 후 예측한 label을 return 받습니다. 손실함수는 이 값과 실제 정답 label을 비교하여 loss를 구하게 됩니다. 다음으로 최적화를 하기위해 미분값을 0으로 만들겠다는 `zero_grad()`를 선언한 후 역전파를 시행합니다. 이와같은 과정을 에폭마다 반복하는 코드를 작성하면 아래와 같습니다.

```python
for EPOCH in range(1, EPOCHS):
    model.train()
    for i, (img, target) in enumerate(data_loader):
        img, target = img.to(device), target.to(device)
        model.to(device)
        out = model(img)
        loss = criterion(out, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(loss.item())
```

<hr>

#### evalutate

학습 이후에는 평가를 진행합니다. 아래 코드 중 `torch.no_grad()` 이 부분은 미분의 추적을 중지하겠다는 의미 입니다. 평가를 할때는 역전파와 같이 미분을 할 필요가 없기 때문에 가용 메모리를 확보하기 위해서 반드시 이 코드를 사용합니다.

DataLoader로 부터 이미지를 불러와서 model에 넣어줍니다. `topk`는 상위 몇개까지의 정확도를 확인할지 설정합니다. 현재는 5개의 Label을 확인합니다. `out.topk`는 입력으로 넣어준 img를 예측한 Label의 확률이 가장 높은 순서대로 5개를 가져옵니다. 그리고 `pred.eq(target[None])`을 통해서 몇개의 Label이 맞았는지를 체크합니다. `acc1`은 정확도가 가장 높다고 예측한 상위 1개의 Label중에 img의 정답 Label이 포함되어있을 확률을 뜻합니다. `acc5`는 정확도가 가장 높다고 예측한 상위 5개의 Label중에 img의 정답 Label이 포함되어있을 확률을 뜻합니다. 단순히, acc5는 당연히 값이 높게 나올 수 밖에 없습니다. 그냥 **acc1이 높으면 좋습니다.**

```python
model.eval()
with torch.no_grad():
    for i, (img, target) in enumerate(data_loader):
        img, target = img.to(device), target.to(device)
        out = model(img)
        topk = (1, 5)
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = out.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target[None])
        res = []
        for k in topk:
            correct_k = correct[:k].flatten().sum(dtype=torch.float32)
            res.append(correct_k * (100.0 / batch_size))
        acc1, acc5 = res
        print(acc1.item(), acc5.item())
```

<hr>

#### checkpoint

마지막으로 남은것이 checkpoint 입니다. 학습을 하면서 업데이트된 가중치를 저장하는 역할을 합니다. 함께 저장해야할 것은 optimizer, lr_scheduler, loss, EPOCHS를 저장합니다. 학습 도중에 중단되더라도, 그 시점에서부터 다시 학습이 진행될 수 있도록 하기 위함입니다. 저장할때의 확장자는 `tar`가 일반적입니다.

```python
    #checkpoint
    checkpoint = {'model': model.state_dict(),
                  'optimizer': optimizer.state_dict(),
                  'lr_scheduler': lr_scheduler.state_dict(),
                  'loss': loss,
                  'EPOCHS': EPOCHS}
    torch.save(checkpoint, './checkpoint.tar')
```
<hr>
