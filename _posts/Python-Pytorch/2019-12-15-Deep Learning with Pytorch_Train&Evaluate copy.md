---
layout: post
title: Tutorial[7] - Train & Validation
comments: true
categories : [Pytorch, Tutorials]
use_math: true
---

딥러닝의 이론적인 내용은 최대한 배제하고 Pytorch를 활용하여 코딩할 수 있도록 필수로 알아야하는 내용에 대해서만 초점을 두었습니다. 

Pytorch 1.4 버전을 기준으로 공식 홈페이지에 나와있는 튜토리얼을 참고하였으며, 데이터 로드, CNN모델 설계, 학습, 추론까지의 일련의 과정을 경험해보는것이 Tutorial의 목표입니다.

<hr>

## Train

이전에 학습했던 모든 단계가 선행되어야만 학습을 진행할 수 있다. 여기는 의외로 코딩하기가 쉬운데, 앞서 배웠던것들을 하나씩 나열하기만 하면 된다. 

<hr>

### 프로세스

```python
model.train() # 학습시 반드시 모드 변경을 해줄것.
for EPOCH in range(1, EPOCHS):
    epoch_loss = .0
    for i, (img, target) in enumerate(data_loader):
        if torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'
        img, target = img.to(device), target.to(device)

        optimizer.zero_grad()
        model.to(device)
        out = model(img)
        loss = criterion(out, target)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

        print(f'loss_iter : {loss.item()}')
    scheduler.step(np.mean(epoch_loss))
    print(f'loss_epoch : {epoch_loss / i}')
```

1. 학습은 DataLoader를 반복문에 넣어주는것으로 시작한다. 한번의 스탭마다 미리 설정했던 batch_size만큼 img, label 정보를 반환한다. 
2. 설계해두었던 Model에 img 정보를 인자로 넣어준다.
3. 설계해두었던 Loss에 model에서 구한 예측셋(Pred)과 정답셋을 인자로 넣어준다. 그리고 역전파를 시행한다.
4. 설계해두었던 Optimizer에서 step함수를 호출하여 가중치를 갱신한다.
5. batch단위의 Loss와 Epoch단위의 Loss를 구하기 위해 Loss함수에서 값을 호출하여 저장해둔다.
6. Epoch이 진행될때마다 설계해두었던 LR_Scheduler에서 step함수를 호출하고 인자로  Epoch단위의 loss를 넣어준다.

<hr>

## Validation

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
