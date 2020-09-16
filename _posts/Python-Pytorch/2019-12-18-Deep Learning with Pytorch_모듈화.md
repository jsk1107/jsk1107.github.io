---
layout: post
title: Tutorial [6] 모듈화
comments: true
categories : [Pytorch, Tutorials]
use_math: true
---

딥러닝의 이론적인 내용은 최대한 배제하고 Pytorch를 활용하여 코딩할 수 있도록 필수로 알아야하는 내용에 대해서만 초점을 두었습니다. Pytorch 1.3.1 버전을 기준으로 공식 홈페이지에 나와있는 튜토리얼을 참고하였으며, 가장 기초적인 CNN모델인 ResNet을 활용하여 classification을 직접 설계해보는 단계까지가 Tutorial의 목표입니다.

<hr>

## 모듈화

지금까지 작성한 코드는 하나의 py 파일에서 실행이 가능합니다. 하지만 이렇게 되면 코드를 수정하거나, 추가기능을 넣을때 매우 복잡해지게 됩니다. 따라서 기능별로 찢어발기는 작업을 진행해 보도록 하겠습니다.

<hr>

#### customdataset.py

CustomDataset은 해당 부분을 그대로 py파일로 만들어 줍니다. 초기 설정도 하드코딩이 된 부분이 없이 모두 parameter로 입력해주었기에 따로 손볼것이 없습니다.

```python
import os
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from PIL import Image

class Customdataset(Dataset):

    def __init__(self, root, transform=None):
        # parameter
        self.root = root
        self.transform = transform

        # label map
        self.category = dict()
        for idx, f in enumerate(os.scandir(self.root)):
            if f.is_dir():
                self.category[idx] = f.name
        self.category_number = list()

        # path 설정
        self.img_file_dirs = [os.path.abspath(d) for d in os.scandir(self.root)]
        self.img_file_path = list()

        # all_img_path생성, 대응되는 category_number 생성
        for idx, img_file_dir in enumerate(self.img_file_dirs):
            for f in os.scandir(img_file_dir):
                self.img_file_path.append(os.path.abspath(f))
                self.category_number.append(idx)

    def __getitem__(self, idx):
        img = Image.open(self.img_file_path[idx]).convert('RGB')
        target = self.category_number[idx]
        target = torch.as_tensor(target, dtype=torch.int64)

        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.img_file_path)
```

<hr>

#### utils.py

utils는 공통으로 사용되는 모듈들을 모아두는 곳입니다. 현재 코드에서는 accuracy를 utils에 빼두는것이 적절한것 같아서 따로 모듈화 하였습니다.

```python
import torch

def accuracy(out, target, topk=(1, )):
    # acc
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = out.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target[None])
        res = []
        for k in topk:
            correct_k = correct[:k].flatten().sum(dtype=torch.float32)
            res.append(correct_k * (100.0 / batch_size))

        return res
```

<hr>

#### train.py

중간중간 시간을 체크하거나 로그를 남길 수 있도록 하는 코드를 추가하고 모두 분리작업을 하였습니다. 학습은 에폭마다 한 사이클이 완료되기 때문에 1에폭을 기준으로 분리를 진행하였습니다(`train_one_epoch`). 또한 데이터를 불러오는 부분은 폴더구조에 따라서 바뀔 수 있기 때문에 따로 분리하였습니다(`load_data`). 평가를 진행하는 부분도 모듈로 분리하였습니다(`evaluate`). 이제 `main`부분에서 각각 모듈화한 부분들을 불러와 학습을 진행하고, 1에폭마다 평가를 한 후 모델을 저장합니다.

```python
import os
import datetime, time
import torch
import torchvision
from torch.utils.data import DataLoader
from customdataset import CustomDataset
from torchvision import transforms as T
from utils import accuracy

def train_one_epoch(model, data_loader, criterion, optimizer, device):
    model.train()

    for i, (img, target) in enumerate(data_loader):
        img, target = img.to(device), target.to(device)
        output = model(img)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(loss.item())

def load_data(data_dir):
    print('Loading data')
    st = time.time()

    transform = T.Compose([T.Resize((128, 128)),
                           T.ToTensor(),
                           T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    dataset = CustomDataset(data_dir, transform=transform)
    train_len = int(len(dataset) * 0.8)
    test_len = int(len(dataset)) - train_len
    train_set, test_set = torch.utils.data.random_split(dataset, lengths=[train_len, test_len])
    print('Took :', time.time() - st, 'sec')

    return train_set, test_set


def evaluate(model, test_loader, criterion, device):
    model.eval()
    with torch.no_grad():
        for i, (img, target) in enumerate(test_loader):
            img, target = img.to(device), target.to(device)
            out = model(img)
            loss = criterion(out, target)
            acc1, acc5 = accuracy(out, target, topk=(1, 2))

            print('loss :{} * Acc@1 {top1:.3f} Acc@5 {top5:.3f}'
          .format(loss, top1=acc1.item(), top5=acc5.item()))


def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    root = 'D:\\data\\img'

    train_set, test_set = load_data(root)

    train_loader = DataLoader(train_set, batch_size=5, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=5, shuffle=True, num_workers=4, pin_memory=True)

    print('Creating Model')
    model = torchvision.models.resnet50()
    model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-04)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=20, gamma=0.1)

    START_EPOCH = 0
    EPOCHS = 5

    resume = False
    if resume:
        checkpoint = torch.load('./checkpoint.tar', map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        START_EPOCH = checkpoint['START_EPOCH']
        EPOCHS = checkpoint['EPOCHS']

    print('Start Training')
    st = time.time()
    for EPOCH in range(START_EPOCH, EPOCHS):
        train_one_epoch(model, train_loader, criterion, optimizer, device)
        lr_scheduler.step()

        if EPOCH % 1 == 0:
            evaluate(model, test_loader, criterion, device)
            checkpoint = {
                'model' : model.state_dict(),
                'optimizier' : optimizer.state_dict(),
                'lr_scheduler' : lr_scheduler.state_dict(),
                'EPOCHS' : EPOCHS
            }
            now = datetime.datetime.now()
            nowDatetime = now.strftime('%Y-%m-%d %H:%M:%S')
            torch.save(checkpoint, './checkpoint_{}.tar'.format(nowDatetime))
    total_time = time.time() - st
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

if __name__ == '__main__':
    main()
```

<hr>

#### argparse

지금까지는 입력옵션들에 대해서 모두 하드코딩을 해주었습니다. 이제는 argparse라는 패키지를 활용해서 하나의 환경변수를 가진 함수로 모두 모아볼것입니다.

```python
def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description='classification 튜토리얼입니다.')
    parser.add_argument('--data-path', default='/home/jsk/data/img', help='데이터의 최상위 경로')
    parser.add_argument('--checkpoint-path', default='/home/jsk/data/model', help='checkpoint path')
    parser.add_argument('--batch-size', type=int, default=2, help='배치사이즈(default:16)')
    parser.add_argument('--num-worker', type=int, default=4, help='CPU 사용 자원 수(default:4)')
    parser.add_argument('--start-epoch', type=int, default=0, help='시작 EPOCH')
    parser.add_argument('--epoch', type=int, default=10, help='전체 EPOCH')
    parser.add_argument('--lr', type=float, default=1e-04, help='leraning_rate(default:1e-04)')
    parser.add_argument('--lr-step', type=int, default=20, help='lr_scheduler_step(default:20)')
    parser.add_argument('--lr-steps', type=int, default=[20, 40], help='lr_scheduler_steps(default:[20, 40])')
    parser.add_argument('--gamma', type=float, default=0.1, help='lr_scheduler_gamma')
    parser.add_argument('--resume', default=False, action='store_true', help='checkpoint 불러오기')

    args = parser.parse_args()
    return args
```

`ArgumentParser`는 Parser의 설명을 기록할 수 있습니다. 그리고 `parser.add_argument`를 통해서 입력 인자와 default로 어떠한 값을 줄것인지, type은 무엇인지 등을 입력 할 수 있습니다. argparse를 통하여 최종적으로 변환된 train.py 코드는 [github](https://github.com/tmdrb0707/pytorch_classification)에 올려두었으니 확인해보시기 바랍니다.