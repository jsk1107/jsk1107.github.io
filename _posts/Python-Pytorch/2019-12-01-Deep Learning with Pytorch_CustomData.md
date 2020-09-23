---
layout: post
title: Tutorial [3] DataSet(ImageFolder)
comments: true
categories : [Pytorch, Tutorials]
use_math: true
---


딥러닝의 이론적인 내용은 최대한 배제하고 Pytorch를 활용하여 코딩할 수 있도록 필수로 알아야하는 내용에 대해서만 초점을 두었습니다. 

Pytorch 1.4 버전을 기준으로 공식 홈페이지에 나와있는 튜토리얼을 참고하였으며, 데이터 로드, CNN모델 설계 학습, 추론까지의 일련의 과정을 경험해보는것이 Tutorial의 목표입니다.

<hr>

## Dataset

수집한 데이터를 불러오는 작업을 해야한다. Classification에서는 torchvision에서 제공하는 ImageFolder를 사용하는것이 가장 효율적이다.

<hr>

### ImageFolder 사용법 : 

ImageFolder를 사용하기 위해선 가장먼저 수집된 데이터의 폴더구조를 아래와 같이 설계해야한다. **반.드.시.**

```Shell
root_dir
    | --- cat/
    |      |-- 0001.jpg
    |      |-- 0002.jpg
    |      |-- 0003.jpg
    |      |-- ...
    |
    | --- dog/
    |      |-- 0001.jpg
    |      |-- 0002.jpg
    |      |-- 0003.jpg
    |      |-- ...
    |
    | --- rabbit/
    |      |--...
```

최상위 경로 아래에 각각의 class name을 가지는 폴더를 구성하고 그 하위경로에 이미지가 저장되어있는 방식이다.

```Python
from torchvision.datasets import ImageFolder

root_dir = '/home/jsk/data/kaggle_dog_cat/train' # 이미지의 최상위 디렉토리
dataset - ImageFolder(root_dir=root_dir, transform=transforms, target_transform=None)
```

ImageFolder 객체는 root_dir, transform등을 인자를 받을 수 있으며, root_dir에는 최상위 경로를 적어주면 된다.

### transform 사용법 :

ImageFolder의 인자로 transform이라는것이 있다. 이것은 이미지를 변형하겠다라는 뜻이다. Shift, Rotate, 형변환 등을 할때 사용된다.

```python
from torchvision.transforms import transforms as T

transforms = T.Compose([T.RandomRotation(degrees=20),
                        T.RandomHorizontalFlip(p=0.5),
                        T.Resize((400, 500)), # (h, w) 순서
                        T.ToTensor()])
```

transform은 반드시 Compose객체에 List형식으로 초기화 해주어야한다.

- `RandomRotation(int)`: +20, -20 사이에서 랜덤으로 값을 정하여 회전한다.
- `RandomHorizontalFlip(int)`: 0.5의 확률로 좌우반전을 한다.
- `Resize(tuple(int, int))`: 이미지 크기를 강제로 조정한다. height, width 순서임.
- `ToTensor(None)`: 입력으로 들어오는 Numpy 또는 PIL 자료형을 Tensor 타입으로 변경한다.

이외에도 많은 함수들이 있다. ([여기서 확인](https://pytorch.org/docs/stable/torchvision/transforms.html)). 반드시 설정해주어야할 클래스는 `ToTensor()`, `Normalize()` 가 있다. 나머지는 기호에 맞게 사용하자(영단어의 뜻으로 어떻게 동작할지 유추할 수 있다).

### DataLoader 사용법 : 

ImageFolder 객체를 활용해 인스턴스를 생성한 후, 하나씩 데이터를 가지고오는 작업을 해야한다. 

DataLoader 클래스는 ImageFolder로부터 생성된 인스턴스를 인자로 받아 Load하는 기능을 가지고 있다. 

```Python
from torch.utils.data import DataLoader

dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
```

DataLoader클래스에서는 초기화된 dataset에서 얼마만큼의 데이터를 꺼내올지(batch), 섞을것인지(shuffle) 등을 정해야한다. 

- `batch_size(int)` : 반복문을 돌때, 한번에 몇개의 이미지를 꺼내올지 정한다.
- `shuffle(bool)` : 한번의 시행이미지를 랜덤하게 섞을지를 정한다.
- `num_worker(int)`: 데이터 I/O를 할 때 사용할 CPU 자원의 수를 지정한다. 멀티 프로세싱의 수를 정하는 개념으로 무조건 높이 설정한다고 좋은것은 아니며, GPU 처리속도와 비교를 통해 튜닝을 해주는것이 좋다.
- `pin_memory(bool)` : 이미지가 Load되기 전, CUDA 메모리에 고정된 메모리 영역을 할당하여 Tensor를 복사를 한다(True하면 빨라진다고 생각하자).
- `collate_fn(func)` : dataset으로부터 리턴된 객체에 대하여 customize한 함수를 적용할때 사용한다.

`sampler`라는 인자도 있는데, 이는 데이터 불균형 문제와 직결되어 있기 때문에 별도로 다루기로 하겠다.

<br>

테스트해보기 : 

```python
import matplotlib.pyplot as plt

for i, data in dataloader:
    print(data.shape) # (batch_size, c, h, w)
    tmp_data = data.numpy().transpose(0, 2, 3, 1) 
    img = tmp_data[0]
    plt.imshow(img)
    plt.show()
    break
```

[이미지 추가하기]


**주의할 점**

입력으로 받는 이미지 사이즈가 다를경우 batch_size로 묶을수가 없다. 그래서 아래와 같은 에러가 발생할 수 있다. 

```python
RuntimeError: invalid argument 0: Sizes of tensors must match except in dimension 0. Got 500 and 225 in dimension 2 at /pytorch/aten/src/TH/generic/THTensor.cpp:612
```

이것은 이전에도 학습했듯이, Tensor의 형상이 다를경우 `torch.cat()`을 적용할때 뿜어냈던 Error와 같다.

**즉, 내가 가진 이미지의 Height, Width가 모두 다른경우에는 transform에서 `Resize()`를 적용할여 모든 이미지를 일괄적인 size로 변환해주어야 한다.**

<hr>
