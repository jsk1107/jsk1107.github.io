---
layout: post
title: Tutorial[7] - Train & Validation & Matric
comments: true
categories : [Pytorch, Tutorials]
use_math: true
---

딥러닝의 이론적인 내용은 최대한 배제하고 Pytorch를 활용하여 코딩할 수 있도록 필수로 알아야하는 내용에 대해서만 초점을 두었습니다. 

Pytorch 1.4 버전을 기준으로 공식 홈페이지에 나와있는 튜토리얼을 참고하였으며, 데이터 로드, CNN모델 설계, 학습, 추론까지의 일련의 과정을 경험해보는것이 Tutorial의 목표입니다.

<hr>

## Train

이전에 학습했던 모든 단계가 선행되어야만 학습을 진행할 수 있다. 여기는 의외로 코딩하기가 쉬운데, 앞서 배웠던것들을 하나씩 나열하기만 하면 된다. 

<br>

### 프로세스 : 

```python
# data_loader는 CustomDataSet에서 call한것을 사용.

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

<br>

2. 설계해두었던 Model에 img 정보를 인자로 넣어준다.

<br>

3. 설계해두었던 Loss에 model에서 구한 예측셋(Pred)과 정답셋을 인자로 넣어준다. 그리고 역전파를 시행한다.

<br>

4. 설계해두었던 Optimizer에서 step함수를 호출하여 가중치를 갱신한다.

<br>

5. batch단위의 Loss와 Epoch단위의 Loss를 구하기 위해 Loss함수에서 값을 호출하여 저장해둔다.

<br>

6. Epoch이 진행될때마다 설계해두었던 LR_Scheduler에서 step함수를 호출하고 인자로  Epoch단위의 loss를 넣어준다.

<hr>

## Validation

학습 이후에는 평가코드가 이어서 나와야 한다. 이곳에서는 평가지표(흔히 Matrix이라 부르는)를 위한 코드와 모델을 저장하는 코드가 별도로 필요하다.

<br>

### 프로세스 :

```python
model.eval() # 반드시 설정해주자

label_map = 'label_map.name' 
matric = MatricTracker(label_map)
best_acc = .0

for i, (img, target) in enumerate(data_loader):
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    img, target = img.to(device), target.to(device)
    model.to(device)

    with torch.no_grad():
        out = model(img)

    matric.update(target, out)

total_acc = matric.accuracy()
print(f'confusion_matric : {matric.result()}')
print(f'ToTal_ACC : {total_acc}')

if best_acc < total_acc:
    state = {'state_dict': model.state_dict(),
             'optimizer': optimizer.state_dict(),
             'lr_scheduler': lr_scheduler.state_dict(),
             'best_acc': total_acc,
             'epoch': epoch}
    torch.save(state, './best_model.pth.tar')
```

1. 학습단계에서와 마찬가지로 DataLoader에서 이미지를 불러와서 모델에 넣어준다. <br><br> 학습에서 진행된 모델이 얼마만큼의 성능을 내는지 검증하는 단계이기 때문에, 최적화나 역전파와 같은 작업은 하지 않아야 한다.  때문에 이미지에 달아두었던 추적기(Auto_Grad)는 모두 제거를 해주어야 한다. <br><br> 이 작업을 한방에 해주는 코드가 바로 `torch.no_grad()` 라는 코드를 with 구문과 함께 감싸주면 된다. with구문이 시작하는 시점에서 추적기를 해제한다.

<br>

2. 미리 만들어둔 MatricTracker 클래스를 호출하여 정확도를 Update해준다.

<br>

3. 현재 에폭에서의 정확도가 이전 모든 에폭에서의 정확도보다 높다면 모델을 Save해준다. <br><br> 모델을 저장할때는 Dict 자료형으로 저장을 해주어야 한다. model 클래스는 `state_dict()`라는 메서드를 가지고 있는데, model의 각 Layer에 가중치값만 모두 뽑아 저장하는것이다. <br><br> 이때 함께 저장해주어야 하는것은 optimizer이다. 마찬자기로 optimizer, lr_scheduler에도 `state_dict()` 메서드를 통해 Gradient값과 lr값을 저장해두도록 하자. <br><br> accuracy와 epoch 등과 같은 메타정보에 대해서는 필요한것들을 저장하도록 한다.

<hr>

## Matric

평가코드에 등장했던 MatricTracker를 더 자세히 알아본다. Classification에서는 Confusion Matric이라는 평가지표를 사용해서 모델의 성능을 종합적으로 검증한다.

어떠 종류의 Task냐에 따라 평가지표는 달라지며 MSE, MAPE, IOU 등등 다양한 방법이 존재한다.

<br>

### 프로세스 : 

```python
import pandas as pd
import torch
import numpy as np

def label_map(label_map_path):
    label_map = {}

    with open(label_map_path, 'r', encoding='utf-8') as t:
        for i, line in enumerate(t.readlines()):

            line = line.rstrip('\n')
            label_map[i] = line
    
    return label_map

class MetricTracker(object):
    def __init__(self, label_map: List[str], writer=None):
        self.label_map = label_map
        self.label_name = list(self.label_map.keys())
        self.switch_kv_label_map = {v: k for k, v in self.label_map.items()}
        self.writer = writer
        self.confusion_metric = pd.DataFrame(0, index=self.label_name, columns=self.label_name)
        self.reset()

    def reset(self):
        for col in self.confusion_metric.columns:
            self.confusion_metric[col].values[:] = 0

    def update(self, target, preds):
        pred = torch.argmax(preds, dim=1)
        target = target.cpu().data.numpy()
        pred = pred.cpu().data.numpy()
        for i in range(len(target)):
            self.confusion_metric.loc[self.label_map[target[i]],
                                  self.label_map[pred[i]]] += 1

    def result(self):
        return self.confusion_metric

    def accuracy(self):
        ACC_PER_CATEGORY = {}
        mAP, mAR, TOTAL_F1_SCORE, TOTAL_ACC = [], [], [], []

        for l in self.switch_kv_label_map:
            ok_cnt = self.confusion_metric.loc[l, l]
            c_values = self.confusion_metric.loc[:, l].values
            r_values = self.confusion_metric.loc[l, :].values
            diff_values = self.confusion_metric.loc[self.confusion_metric.columns != l,
                                                    self.confusion_metric.columns != l].values
            
            # 0으로 초기화 했던 자리가 업데이터가 안될경우 예외처리. 
            if ok_cnt == 0 or np.sum(c_values) == 0 or np.sum(r_values) == 0:
                continue

            AP = ok_cnt / np.sum(c_values)
            AR = ok_cnt / np.sum(r_values)
            F1_SCORE = 2 * (AP * AR) / (AP + AR)
            ACC = (ok_cnt + np.sum(diff_values)) / np.sum(np.array(self.confusion_metric))
            ACC_PER_CATEGORY[l] = {'AP': AP,
                                   'AR': AR,
                                   'F1_SCORE': F1_SCORE,
                                   'ACC': ACC}
            mAP.append(AP), mAR.append(AR), TOTAL_F1_SCORE.append(F1_SCORE), TOTAL_ACC.append(ACC)
        true = np.sum(np.diag(np.array(self.confusion_metric)))
        total_cnt = np.sum(np.array(self.confusion_metric))
        ACC = true / total_cnt
        return ACC_PER_CATEGORY, np.mean(mAP), np.mean(mAR), np.mean(TOTAL_F1_SCORE), ACC
```

- `init` : 여기에서 label_map을 Dictionary형태의 인자로 받는다. 때문에, 클래스 외부에서 label_map을 오픈하여 line by line으로 dictionary를 미리 만들어주어야 한다. key는 0, 1, 2, ... 순서대로 증가시키고 value로는 label_name을 넣어주면 된다. <br> 이렇게 생성된 Dict를 Tracker에 인자로 받아서 초기화 시켜주는것이다. 그리고 Pandas 패키지를 활용하여 Confusion Matric의 골격을 만들어주고 0으로 초기화 한다. 
    
- `reset` : Confusion Matric은 Epoch마다 다시 계산을 해주어야 하기 때문에 초기화하는 메서드.

- `update` : 위와 마찬가지로 Batch_Size마다 학습을 진행하기 때문에 결과를 갱신해주는 메서드.
    
- `result` : Epoch이 끝나면 전체 Confusion Matric이 어떻게 생겼는지 보고 싶을 수 있다. 전체 구조를 볼 수 있게 matric을 리턴해주는 메서드.
    
- `accuracy` : 각 class별로 Accuracy와 같은 지표를 뽑아볼 수 있는 메서드. 코드를 보면 Total_Acc이외에 AP, AR, F1_Score와 같은 지표도 함께 추출한다. 여러개의 class에서는 class별 Accuracy를 보는것이 아닌 AP, AR 지표를 보는것이 합리적이다.

이렇게 만들어놓은 MatricTracker 클래스는 Classification 문제에서 어디서든 활용 할 수 있는 API가 된다.