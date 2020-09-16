---
layout: post
title: Tutorial [7] Inference
comments: true
categories : [Pytorch, Tutorials]
use_math: true
---

딥러닝의 이론적인 내용은 최대한 배제하고 Pytorch를 활용하여 코딩할 수 있도록 필수로 알아야하는 내용에 대해서만 초점을 두었습니다. Pytorch 1.3.1 버전을 기준으로 공식 홈페이지에 나와있는 튜토리얼을 참고하였으며, 가장 기초적인 CNN모델인 ResNet을 활용하여 classification을 직접 설계해보는 단계까지가 Tutorial의 목표입니다.

<hr>

## Inference

이제 마지막으로 남은것은 추론만이 남아있습니다. 우리는 알지 못하는 이미지를 새롭게 구하여 학습이 완료된 모델에 Input으로 넣어주면 추론이 완료됩니다.

<p align="center">
<img width="400" alt="cat" src="https://www.dropbox.com/s/zuyg6datwuni0eg/92969_25283_5321.jpg?raw=1">
</p>

<center>(이것은 고양이가 아니다...미지의 동물이다...)</center>

#### inference.py

```python
import os
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T

class InferenceDataset(Dataset):

    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.img_path = list()
        for f in os.scandir(root):
            if f.is_file():
                self.img_path.append(os.path.abspath(f))

    def __getitem__(self, idx):
        img = Image.open(self.img_path[idx]).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)
        return img

    def __len__(self):
        return len(self.img_path)


def load_data(root, transform):
    dataset = InferenceDataset(root, transform())

    return dataset

def transform():
    transform = list()
    transform.append(T.ToTensor())
    return T.Compose(transform)

def inference(args):

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    root = args.data_path

    inference_set = load_data(root, transform)
    inference_loader = DataLoader(inference_set, batch_size=1, num_workers=4)
    model = torchvision.models.resnet50()
    checkpoint = torch.load(os.path.join(args.checkpoint_path, 'checkpoint_2019_12_20.tar'), map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()
    criterion = torch.nn.Softmax()
    for i, img in enumerate(inference_loader):
        out = model(img)
        pred = criterion(out)
        # print(out.data.numpy().argmax())
        print(inference_set.img_path[i], pred.data.numpy(), pred.data.numpy().argmax())

def parse_arg():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', default='/home/jsk/data/inference', help='데이터 최상위 경로')
    parser.add_argument('--checkpoint-path', default='/home/jsk/data/model', help='데이터 최상위 경로')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_arg()
    inference(args)
```
<br>

<hr>

지금까지 Pytorch를 활용하여 기초적인 학습과 추론을 시작해보았습니다. 처음 Deep Learning을 접하시는 분들은 데이터를 불러오고 이미지의 Shape을 맞추고 모델을 불러오는 등의 작업에서 많은 장벽을 느낍니다. 또한 Youtube나 강의를 보아도 Jupyter Notebook을 통해 진행되기 때문에 모듈화가 되어있지 않는다는 단점도 있습니다.

<br>

직접 API를 개발하거나 어떠한 서비스를 개발할 목적을 가지고 계신 분이라면 모듈화는 필수입니다. 기존 Tutorial을 토대로 모듈화까지 진행하며 Pytorch에 개괄적인 부분을 살펴보았습니다. 부족한 부분이 많지만 많은 도움이 되었으면 좋겠습니다.