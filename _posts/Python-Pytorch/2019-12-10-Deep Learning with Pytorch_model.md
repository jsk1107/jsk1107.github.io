---
layout: post
title: Tutorial[5] - model
comments: true
categories : [Pytorch, Tutorials]
use_math: true
---

딥러닝의 이론적인 내용은 최대한 배제하고 Pytorch를 활용하여 코딩할 수 있도록 필수로 알아야하는 내용에 대해서만 초점을 두었습니다. 

Pytorch 1.4 버전을 기준으로 공식 홈페이지에 나와있는 튜토리얼을 참고하였으며, 데이터 로드, CNN모델 설계, 학습, 추론까지의 일련의 과정을 경험해보는것이 Tutorial의 목표입니다.

<hr>

## model

Pytorch에서 모델을 생성하는 방법은 Tensorflow보다 쉽다. 그 방법이 매우 직관적이기 때문이다. 모든 함수는 `torch.nn`이라는 곳에서 사용하며, 모델과 관련된 모든 함수가 이곳에 있다.

### Create Network : 

```python
class MyNetwork(nn.Module):
    def __init__(self, num_class=10):
        super(MyNetwork, self).__init__()
        """
            Layer 클래스 초기화
        """

    def forward(self, x):

        """
            Layer 디자인
        """
        return x
```

신경망을 설계할때는 반드시 `torch.nn.Module` 라는 클래스를 상속받아 사용한다. 이 Module이라는 객체에서 순전파, 역전파 등을 할 수 있는 함수들이 내장되어있기 때문에 이를 오버라이딩하여 사용한다.

- `__init__` : 모듈 초기화에 필요한것을 셋팅한다. 일반적으로 각각의 Layer, Block, FPN 등 신경망에 필요한 모든 클래스를 생성해주는 곳이다.

<br>

- `forward` : 초기화 했던 클래스를 불러와서 Layer를 디자인하는 곳이다. 이미지가 입력되었을때 디자인된 순서대로 순전파된다.

그럼 간단한 신경망을 설계해보도록 하자.

<br>

### Basic CNN Network : 

```python
class MyNetwork(nn.Module):
    def __init__(self, num_class=10):
        super(MyNetwork, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32,
                               kernel_size=7, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64,
                               kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128,
                               kernel_size=3, stride=1, padding=1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(in_features=128, out_features=1024)
        self.fc2 = nn.Linear(in_features=1024, out_features=num_class)


    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.conv3(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.fc2(x)

        return x
```

- `__init__` : 영역에서는 Conv Layer, FC Layer, Activation Layer, Pooling Layer, Batch_Norm Layer를 초기화 하였다. 얕은 깊이의 신경망을 만들것이기 때문에, Layer를 모두 생성해주었다(conv1, conv2, ...). <br><br> 이렇게 초기화 하였을때 문제점은 재사용성이 매우 떨어진다는 점이다. 따라서 깊은 신경망을 구축하고자 하면 Block이라는 메서드를 만드는것이 더 좋다.

<br>

- `forward` : 영역에서는 초기화한 Layer를 하나씩 쌓아가는 작업을 하였다. 이처럼 쌓은 순서대로 순전파를 진행하기 때문에(ex. bn, conv의 순서를 변경) 사용자가 직접 이곳을 커스터마이징 하여 신경망의 성능을 향상시킬 수 있다.

<br>

### 신경망 구조 확인하기 : 

위처럼 신경망 설계를 마무리 하였다면 아래처럼 불러와서 사용한다.

```Python
model = MyNetwork(num_class=2)
print(model)

"""
MyNetwork(
  (conv1): Conv2d(3, 32, kernel_size=(7, 7), stride=(2, 2), padding=(1, 1))
  (bn1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (conv2): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (relu): ReLU()
  (maxpool): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
  (conv3): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (avgpool): AdaptiveAvgPool2d(output_size=(1, 1))
  (fc1): Linear(in_features=128, out_features=1024, bias=True)
  (fc2): Linear(in_features=1024, out_features=2, bias=True)
)
"""
```

단순히 model 객체를 print하면 신경망의 요약정보가 나타난다. 눈여겨 볼점은 두가지이다.

<br>

첫번째로는 괄호안에 사용자가 붙여준 인스턴스 이름이 붙어있다는점. 이를 통해 해당 Layer가 어떤 기능을 하는것인지 파악할 수 있다. 그러니 변수명을 잘 해야한다!!

<br>

두번째로는 Layer에 사용한 클래스의 초기화값을 볼 수 있다는점이다. 이를 통해 각각의 Layer를 통과했을때 이미지의 형상이 어떻게 되는지 유추해볼 수 있다.

<br>

### torchsummary :

Layer통과 후 이미지의 형상이 어떻게 되는지 유추를 하는것은 번거로운 작업일 뿐만 아니라, Parameter의 갯수에 대한 정보도 알 수 없다. 무언가 부족한 느낌이 있다.

<br>

이를 보완해준 패키지가 있으니 `torchsummary`이다. github에서 따봉을 2000개 넘게 받은 국민 패키지이다. Pytorch에서 생성한 신경망의 정보를 Keras스타일로 보여주는것으로, 아주 편리하다.

```Shell
$ pip install torchsummary
```

```python
from torchsummary import summary
summary(model, input_size=(3, 112, 112), batch_size=4)
"""
================================================================
            Conv2d-1            [4, 32, 54, 54]           4,736
       BatchNorm2d-2            [4, 32, 54, 54]              64
              ReLU-3            [4, 32, 54, 54]               0
         MaxPool2d-4            [4, 32, 27, 27]               0
            Conv2d-5            [4, 64, 27, 27]          18,496
       BatchNorm2d-6            [4, 64, 27, 27]             128
              ReLU-7            [4, 64, 27, 27]               0
         MaxPool2d-8            [4, 64, 14, 14]               0
            Conv2d-9           [4, 128, 14, 14]          73,856
AdaptiveAvgPool2d-10             [4, 128, 1, 1]               0
           Linear-11                  [4, 1024]         132,096
           Linear-12                     [4, 2]           2,050
================================================================
Total params: 231,426
Trainable params: 231,426
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.57
Forward/backward pass size (MB): 14.71
Params size (MB): 0.88
Estimated Total Size (MB): 16.17
----------------------------------------------------------------
"""
```

summary함수에 인자로 신경망 모델과, Test할 이미지의 형상, batch_size를 입력해주면 위 처럼 표현해준다.

<br>

각각의 Layer를 통과했을때의 이미지의 형상이 어떻게 변하는지와 해당 Layer에서 사용된 Parameter의 수를 보여준다. 아래쪽에는 전체 Parameter의 갯수, 학습에 실제 사용된 Parameter의 수(DropOut을 걸었을때 꺼지는 Parameter는 제외된다) 등을 보여주고, 모델이 저장되었을 때, 어느정도의 용량이 될것인지 추정도 해준다. 아주 혜자 패키지.  
