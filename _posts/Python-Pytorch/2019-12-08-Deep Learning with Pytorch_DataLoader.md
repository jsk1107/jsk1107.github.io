---
layout: post
title: Tutorial [4] DataSet(CumstomDataset)
comments: true
categories : [Pytorch, Tutorials]
use_math: true
---

딥러닝의 이론적인 내용은 최대한 배제하고 Pytorch를 활용하여 코딩할 수 있도록 필수로 알아야하는 내용에 대해서만 초점을 두었습니다. Pytorch 1.3.1 버전을 기준으로 공식 홈페이지에 나와있는 튜토리얼을 참고하였으며, 가장 기초적인 CNN모델인 ResNet을 활용하여 classification을 직접 설계해보는 단계까지가 Tutorial의 목표입니다.

<hr>

## CustomDataset

이미지를 수집할 때, 간혹 ImageFolder에서 사용할 수 있는 폴더구조를 갖지 못하는경우가 발생한다. 예를들어 아래와 같은 폴더 구조에서는 사용할 수 없다.

```Shell
root_dir/
    | --- cat_0001.jpg
    | --- cat_0002.jpg
    | --- dog_0001.jpg
    | --- rabbit_0001.jpg
    | --- rabbit_0002.jpg
    | --- horse_0001.jpg
    | --- horse_0002.jpg
    | --- cat_0003.jpg
    | ...
    | ...
```

데이터의 수가 많지 않다면 수작업으로 ImageFolder의 폴더구조를 만들어주면 된다. 하지만 데이터가 1만장, 10만장...이렇다면? 수작업으로 하기에는 불가능하다.

따라서 위와같은 데이터를 받을 수 있도록 사용자가 직접 DataSet 클래스를 만들어주어야 한다.

**주의할 점**

위의 폴더구조를 잘 살표보면, 이미지 이름에 cat, dog, rabbit, ... 과 같이 이미지를 설명해줄 수 있는 Label이 달려있다.

즉, 데이터 수집을 할 때, 마구잡이로 수집하면 절.대. 안된다. 최소한 해당 이미지가 어떠한 사물을 대표하는지 Label를 명시해주어야 한다.

(처음 딥러닝을 접했을 때, 이것때문에 멘탈파괴가 되었던 경험이 있음)

### CustomDataset의 구조:

torch에 Dataset클래스를 상속받아서 나만의 Dataset을 생성한다. 이때, 반드시 필요한 메서드는 3가지가 있다.

```python
from torch.utils.data import Dataset

class CustomDataset(Dataset):
	# 초기화
	def __init__(self, x, ...):
		# 데이터셋 경로설정. 모든 이미지 파일에 대한 경로를 잡아주어야함.

    # DataLoader 클래스에서 접근하며 Load되는 부분
    def __getitem__(self, idx, ...):
        ...
        return img, target

    # 데이터셋에서 한개의 이미지와 Label 정보를 리턴.
	def __len__(self):
        return len(...)
		# 데이터셋의 길이를 반환을 정의
```

- `__init__()`: CustomDataset을 호출할때 초기화되는 부분이다. 때문에 데이터, 메타정보 등 학습에 필요한 모든 정보를 여기에서 초기화 해야한다. 

- `__getitem__(idx)`: 이미지와 메타정보를 DataLoader 클래스에 return하는 영역이다. 이미지 경로로 부터 idx(인덱스)에 해당하는 이미지를 선택한다. 따라서 이부분에서는 이미지가 실제로 Load 되야하며 메타정보도 함께 처리되야한다.

- `__len__()`: 사용자가 학습에 사용할 이미지의 전체 갯수를 return해준다.

### Create CustomDataset : 
```python
import os
from PIL import Image

class CustomDataset(Dataset):
    def __init__(self, root_dir, transforms=None):
        self.root_dir = root_dir # 데이터 폴더의 최상위경로
        self.transforms = transforms

        # 모든 이미지의 경로 저장.
        self.img_path = []
        self.category_names = []
        for idx, f in enumerate(os.scandir(root_dir)):
            if not f.is_dir():
                ValueError("It isn't folder. Check your folder structure")    
            self.img_path.append(os.path.abspath(f.name))
            self.category_names.append(f.name.split('_')[0])

        self.img_path = sort(self.img_path) # 반.드.시.!!!!
        self.category_names = list(set(self.category_names)) # 중복 제거하고 List 형변환

        # category 사전 만들기
        for i, category in enumerate(self.category_names):
            self.category_dict[category] = i

    def __getitem__(self, idx):
         
         # 이미지 오픈
        img = Image.open(self.img_path[idx]).conver('RBG')

        # category 가져오기
        target_name = os.path.basename(self.img_path[idx]).split('_')[0]
        target = self.category.get(target_name)
        target = torch.as_tensor(target, dtype=torch.int64)

        if self.transform is not None:
            img = self.transforms(img)

        return img, target # return되는것은 항상 Tensor 타입이어야 한다.
    
    def __len__(self):
        return len(self.img_path)
```

1. `__init__`

우선 DataSet이 있는 최상위 폴더에 대한 경로 `root_dir`을 인자로 받는다. 우선 각각의 이미지에 대하여 절대경로를 받아서 초기화 해주어야 한다.

또한, ImageFolder와 달리 해당 이미지가 어떤 객체를 가리키는지에 대한 메타정보를 이미지의 이름에서 가지고 와야한다. 때문에 파싱하는 작업을 해야하고, 데이터를 수집할때 파싱을 할 수 있는 구분자가 있으면 더욱 편하게 작업할 수 있다.

파싱된 글자를 통해 `category`라는 딕셔너리에 하나씩 담아준다. Github에 올라온 코드를 보면 이와같은 파싱작업의 수고로움을 덜기위해 **label_map.name** 과 같은 파일을 자주 볼 수 있다. 미리 category에 대한 정보를 담아둔 파일이다(추후 모듈화에서 다루도록 하겠음).

2. `__getitem__`

DataLoader로 이미지와 메타정보를 넘겨주기 위한 작업을 하는 공간이다.

인자로써 idx를 가지는데, 이것은 전체 이미지의 갯수에서 랜덤하게 하나의 인덱스를 가져와준다. 즉, 한장의 이미지를 처리한다.

따라서, 초기화해둔 이미지를 불러오는 작업과 어떠한 카테고리인지 특정하는 코드를 작성한다. 또한 `self.transform`을 통해 데이터 변환을 해준 후, 이미지와 메타정보(여기서는 카테고리)를 return해주면 끝.

3. `__len__`

`self.img_path`에는 `root_dir`경로 하위에 있는 모든 img에 대한 경로가 들어있다. 여기에서 총 img가 몇개인지만 return한다.

여기까지가 CustomDataset을 생성하는 가장 기본적인 방법이다. 이것은 앞으로도 Detection, Segmentation, NLP등에서도 계~~~속 사용하는 방법이니 꼭 이해하고 넘어가자.

<hr>