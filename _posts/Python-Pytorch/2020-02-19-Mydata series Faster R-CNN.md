---
layout: post
title: Object Detection [1] 나만의 Data로 실행해보자 - Faster R-CNN
comments: true
categories : [Pytorch, Object Detection]
use_math: true
---

Deep Learning의 다양한 모델(ResNet, Inception 등)을 직접 개발하는것도 굉장한 도움이 되지만, 입문자에게 있어 그것은 좌절만 안겨줄 뿐입니다(심지어 고수들도). 우선 선형대수학에 대한 개념도 필요하고, 알고리즘과 최적화문제 등 매우 복잡한 형태로 이루어져 있기 때문입니다. 또한 엄청난 양의 모듈화로 하나하나 찾아보기란 쉬운일이 아닙니다.

코딩을 함에 있어 기초지식을 쌓았다면, 다른사람이 만든 API를 직접 실행해 보는것이 다음 순서라고 생각합니다. 이를통해 코딩의 역량을 확실히 향상시킬 수 있을것입니다.

<hr>

## 나만의 Data로 실행해보자 Seires 1탄 Faster R-CNN.

가장 먼저 준비해야하는것은 Github에서 Faster R-CNN을 clone하는것입니다. Pytorch에서는 Tutorial 성격으로 제공해주는 코드가 있으므로 이를 활용하겠습니다.
Repository 전체를 Clone하여도 괜찮지만 쓸모없는 파일이 많기때문에, 그냥 코드를 복붙하여 사용하도록 하겠습니다.
[링크](https://github.com/pytorch/vision/tree/master/references/detection)된 Pytorch Github로 들어가면 8개의 파일이 있습니다. 우선 해당 파일을 모두 복붙하여 하나씩 생성하도록 합니다.

그리고 해당 폴더 구조를 조금 변경하도록 하겠습니다.

```Shell
WORKSPACE
    | --- engine.py
    | --- train.py
    | --- coco_eval.py
    | --- coco_util.py
    | --- utils
           |-- group_by_aspect_ratio.py
           |-- transforms.py
           |-- utils.py
```

utils 폴더에는 각종 클래스 및 메서드를 모아둔 폴더입니다. train.py나 engin.py 등에서 실행할때 필요한것들을 관리하기 좋게 모아둔 것입니다. **폴더구조 변경으로 인해 모든 파일에 import utils 부분을 import utils.utils as utils 로 수정해합니다.**

<hr>

<br>

이제 데이터를 수집해 줍니다. 나만의 Data를 수집하기 위해서 약간의 노가다를 해줍니다. 저는 고속도로 자동차 사진을 수집하도록 하겠습니다. Google에서 "고속도로 자동차" 라는 키워드로 검색하여 이미지를 몇장 다운받도록 하겠습니다.

<p align="center">
<img width="400" src="https://www.dropbox.com/s/8b8ac0n6kyq6y2q/1.PNG?raw=1">
</p>

이미지를 수집했으면 Ground Truth Bounding Box를 처리해주어야 합니다. Terminal에서 라이브러리를 하나 인스톨 하고 실행해 줍니다.

```Shell
>>> pip install labelImg
>>> labelImg
```

Bounding Box Tool은 여러종류 있습니다. 그중에서도 labelImg는 오래된 Tool이지만 여전히 많은 사람들이 사용하고 있습니다. labelImg를 Open하면 다음과 같은 창이 하나 뜹니다. 

<p align="center">
<img width="400" alt="labelImg" src="https://www.dropbox.com/s/pu44uactm30dojt/2.png?raw=1">
</p>

좌측의 **Open Dir**을 눌러서 수집한 이미지가 있는 경로를 설정합니다. 그리고 **Change Save Dir**을 눌러서 annotaion을 저장할 경로를 설정합니다. 마지막으로 상단의 **View** 탭에서 **Auto Save mode**를 클릭합니다.

사용법은 간단합니다. **W**키를 눌러 Bounding Box처리를 할 수 있고 **D**룰 누르면 다음 이미지로 건너갑니다. 이제 10장의 이미지에 보이는 모든 차량 사진에 **Car**라고 하는 label를 달아주도록 합니다.

<p align="center">
<img width="400" alt="label" src="https://www.dropbox.com/s/n2kcsor79962rfb/3.png?raw=1">
</p>

데이터의 폴더구조는 아래와 같이 설정하였습니다. 폴더구조는 아래처럼 수집하는것을 추천합니다.

```Shell
data
    | --- img
    |      |-- 001.png
    |      |-- 002.png
    |      ...
    | --- annotation
           |-- 001.xml
           |-- 002.xml
           ...
```

<hr>

<br>

가장먼저 찾아보아야 할 곳은 main이 실행되는곳 입니다. 대부분 파일명은 main.py 또는 train.py 로 명명되어있습니다. 혹시 다르다 할지라도 소스코드를 살펴보면 알 수 있습니다.

#### train.py
```python
r"""PyTorch Detection Training.
To run in a multi-gpu environment, use the distributed launcher::
    python -m torch.distributed.launch --nproc_per_node=$NGPU --use_env \
        train.py ... --world-size $NGPU
The default hyperparameters are tuned for training on 8 gpus and 2 images per gpu.
    --lr 0.02 --batch-size 2 --world-size 8
If you use different number of gpus, the learning rate should be changed to 0.02/8*$NGPU.
On top of that, for training Faster/Mask R-CNN, the default hyperparameters are
    --epochs 26 --lr-steps 16 22 --aspect-ratio-group-factor 3
Also, if you train Keypoint R-CNN, the default hyperparameters are
    --epochs 46 --lr-steps 36 43 --aspect-ratio-group-factor 3
Because the number of images is smaller in the person keypoint subset of COCO,
the number of epochs should be adapted so that we have the same number of iterations.
"""
import datetime
import os
import time

import torch
import torch.utils.data
from torch import nn
import torchvision
import torchvision.models.detection
import torchvision.models.detection.mask_rcnn

from coco_utils import get_coco, get_coco_kp

from group_by_aspect_ratio import GroupedBatchSampler, create_aspect_ratio_groups
from engine import train_one_epoch, evaluate

import utils
import transforms as T


def get_dataset(name, image_set, transform, data_path):
    paths = {
        "coco": (data_path, get_coco, 91),
        "coco_kp": (data_path, get_coco_kp, 2)
    }
    p, ds_fn, num_classes = paths[name]

    ds = ds_fn(p, image_set=image_set, transforms=transform)
    return ds, num_classes


def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


def main(args):
    utils.init_distributed_mode(args)
    print(args)

    device = torch.device(args.device)

    # Data loading code
    print("Loading data")

    dataset, num_classes = get_dataset(args.dataset, "train", get_transform(train=True), args.data_path)
    dataset_test, _ = get_dataset(args.dataset, "val", get_transform(train=False), args.data_path)

    print("Creating data loaders")
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test)
    else:
        train_sampler = torch.utils.data.RandomSampler(dataset)
        test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    if args.aspect_ratio_group_factor >= 0:
        group_ids = create_aspect_ratio_groups(dataset, k=args.aspect_ratio_group_factor)
        train_batch_sampler = GroupedBatchSampler(train_sampler, group_ids, args.batch_size)
    else:
        train_batch_sampler = torch.utils.data.BatchSampler(
            train_sampler, args.batch_size, drop_last=True)

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_sampler=train_batch_sampler, num_workers=args.workers,
        collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1,
        sampler=test_sampler, num_workers=args.workers,
        collate_fn=utils.collate_fn)

    print("Creating model")
    model = torchvision.models.detection.__dict__[args.model](num_classes=num_classes,
                                                              pretrained=args.pretrained)
    model.to(device)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_steps, gamma=args.lr_gamma)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1

    if args.test_only:
        evaluate(model, data_loader_test, device=device)
        return

    print("Start training")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        train_one_epoch(model, optimizer, data_loader, device, epoch, args.print_freq)
        lr_scheduler.step()
        if args.output_dir:
            utils.save_on_master({
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'args': args,
                'epoch': epoch},
                os.path.join(args.output_dir, 'model_{}.pth'.format(epoch)))

        # evaluate after every epoch
        evaluate(model, data_loader_test, device=device)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description=__doc__)

    parser.add_argument('--data-path', default='/datasets01/COCO/022719/', help='dataset')
    parser.add_argument('--dataset', default='coco', help='dataset')
    parser.add_argument('--model', default='maskrcnn_resnet50_fpn', help='model')
    parser.add_argument('--device', default='cuda', help='device')
    parser.add_argument('-b', '--batch-size', default=2, type=int,
                        help='images per gpu, the total batch size is $NGPU x batch_size')
    parser.add_argument('--epochs', default=26, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--lr', default=0.02, type=float,
                        help='initial learning rate, 0.02 is the default value for training '
                        'on 8 gpus and 2 images_per_gpu')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('--lr-step-size', default=8, type=int, help='decrease lr every step-size epochs')
    parser.add_argument('--lr-steps', default=[16, 22], nargs='+', type=int, help='decrease lr every step-size epochs')
    parser.add_argument('--lr-gamma', default=0.1, type=float, help='decrease lr by a factor of lr-gamma')
    parser.add_argument('--print-freq', default=20, type=int, help='print frequency')
    parser.add_argument('--output-dir', default='.', help='path where to save')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, help='start epoch')
    parser.add_argument('--aspect-ratio-group-factor', default=3, type=int)
    parser.add_argument(
        "--test-only",
        dest="test_only",
        help="Only test the model",
        action="store_true",
    )
    parser.add_argument(
        "--pretrained",
        dest="pretrained",
        help="Use pre-trained models from the modelzoo",
        action="store_true",
    )

    # distributed training parameters
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')

    args = parser.parse_args()

    if args.output_dir:
        utils.mkdir(args.output_dir)

    main(args)
```

위 소스코드를 살펴보면 데이터를 불러오는 부분인 get_dataset이라는곳이 있습니다. 이곳은 데이터를 불러와서 return해주는 영역입니다. 따라서 CustomDataset class를 만들고 get_dataset 함수부분을 대체하도록 하겠습니다.

#### data_util.py
```python
import torch
import xml.etree.ElementTree as ET

import os

from utils.data_utils import load_anno
from torch.utils.data import Dataset
from PIL import Image


class CustomDataset(Dataset):
    def __init__(self, img_dir, anno_dir, transforms=None):
        self.img_dir = img_dir
        self.anno_dir = anno_dir
        self.transforms = transforms
        self.img_list_path = [os.path.join(self.img_dir, f.name) for f in os.scandir(img_dir)]
        self.anno_list_path = [os.path.join(self.anno_dir, f.name) for f in os.scandir(anno_dir)]

    def __getitem__(self, idx):
        img = Image.open(self.img_list_path[idx]).convert('RGB')
        target = load_anno(self.anno_list_path[idx])
        if self.transforms is not None:
            img, target = self.transforms(img, target)
        return img, target

    def __len__(self):
        return len(self.img_list_path)


def load_anno(anno_path):

    tree = ET.parse(source=anno_path)
    root = tree.getroot()

    width = int(root.find('size').find('width').text)
    height = int(root.find('size').find('height').text)
    depth = int(root.find('size').find('depth').text)
    object = root.findall('object')

    boxes = []
    label_numbers = []

    for obj in object:
        label = obj.find('name').text
        label_number = label_map(label)
        xmin = int(obj.find('bndbox').find('xmin').text)
        ymin = int(obj.find('bndbox').find('ymin').text)
        xmax = int(obj.find('bndbox').find('xmax').text)
        ymax = int(obj.find('bndbox').find('ymax').text)
        boxes.append([xmin, ymin, xmax, ymax])
        label_numbers.append(label_number)

    target = {}
    target['width'] = torch.as_tensor(width, dtype=torch.int64)
    target['height'] = torch.as_tensor(height, dtype=torch.int64)
    target['depth'] = torch.as_tensor(depth, dtype=torch.int64)
    target['boxes'] = torch.as_tensor(boxes, dtype=torch.float32)
    target['label'] = torch.as_tensor(label_numbers, dtype=torch.int64)

    return target


def label_map(label):

    if label == "Car":
        label_number = 1

    return label_number
```

CustomDataset에서는 img와 xml파일의 경로를 받아 하나씩 Open해준 후 return합니다. xml파일은 Parsing을 해야하기 때문에 load_anno라는 함수를 따로 만들었습니다. Pytorch에서 제공하는 Faster R-CNN을 사용하기 위해서는 xml을 Parsing한 후 Dictionary형태로 만들어주어야 합니다. 필요한 Key는 Bounding Box정보와 Label명입니다. 주의할점은 tensor타입만 받아들이기 때문에, label정보를 미리 숫자로 변환하여 return하는 label_map함수를 만들었습니다.

이제 다시 train.py에서 get_data부분을 수정하도록 합니다.

```python
...
...
 # Data loading code
    print("Loading data")

   # dataset, num_classes = get_dataset(args.dataset, "train", get_transform(train=True), args.data_path)
    #dataset_test, _ = get_dataset(args.dataset, "val", get_transform(train=False), args.data_path)
    dataset = CustomDataset(args.img_data_path, args.anno_data_path, transforms=get_transform(train=True))
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(.2 * dataset_size))
    random_seed = 42
    np.random.seed(random_seed)
    np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    dataset_train = SubsetRandomSampler(train_indices)
    dataset_test = SubsetRandomSampler(val_indices)
...
...
```

CustomDataset에는 전체 Dataset이 들어있기 때문에, 이를 Train/Val로 분리를 해주어야 합니다. 애초에 수집부터 Train, Val을 따로 구분하여 수집하였다면 이 과정은 필요없지만, 지금은 반드시 해주어야 하는 부분입니다. Pytorch에서는 SubsetRandomSampler라는 기능을 제공하고 있습니다. 또한 SubsetRandomSampler를 사용하게되면 BatchSampler가 필요가 없으니 해당 부분은 주석처리를 하던 제거하던지 해줍시다.

마지막부분의 argparse부분만 조금 조정하도록 합시다. 다른것은 수정할것이 없고 data경로와 모델이 저장될 경로만 추가수정 해주면 됩니다.

```python
   import argparse
    parser = argparse.ArgumentParser(
        description=__doc__)
    # parser.add_argument('--data-path', default='/datasets01/COCO/022719/', help='dataset')
    # parser.add_argument('--dataset', default='coco', help='dataset')
    parser.add_argument('--img-data-path', default='./data/img', help='dataset')
    parser.add_argument('--anno-data-path', default='./data/anno', help='dataset')
    parser.add_argument('--model', default='fasterrcnn_resnet50_fpn', help='model')
    ...
    ...
    parser.add_argument('--output-dir', default='./models', help='path where to save')
    ...
```

이제 모든것이 다 끝났습니다. train.py에서 Run을 해주면 실행이 됩니다. 혹은 Linux Terminal에서 실행하고자 한다면 데이터의 경로만 인자로 지정해주면 됩니다. epoch이나 lr-steps등을 조정하고 싶다면 추가로 인자를 입력하면 되고, 입력하지 않으면 기존에 설정해두었던 default값이 실행되게 됩니다.

```Shell
python train.py --data-path ./data/img --anno-data-path ./data/anno (Optinally) --epoch 30 --lr-steps 20 ...
```


#### Detection의 Folder구조 : 

Detection에서는 JSON, XML 등의 메타정보를 가진 파일을 활용합니다. 다행인것은 메타정보에 해당 이미지의 Label정보를 가지고 있기 때문에 쉽게 Folder구조를 설계할 수 있습니다.
```Shell
dataset
    | --- cat0001.jpg
    | --- cat0001.xml
    | --- cat0002.jpg
    | --- cat0002.xml
    | --- dog0001.jpg
    | --- dog0001.xml
    | --- rabbit0001.jpg
    | --- rabbit0001.xml
    | ...
```

망했습니다. 우선 데이터 수집을 할때 cat0001.jpg, cat0002.jpg처럼 이미지의 이름에 label정보는 굳이 없어도 됩니다. 왜냐하면 xml파일이 Label정보를 가지고 있기 때문입니다. classification과 마찬가지로 I/O에도 많은 부담이 갑니다. 아래와 같은 Folder구조로 바꾸어주도록 합니다.

```Shell
dataset
    | --- Image
    |      |-- 0001.jpg
    |      |-- 0002.jpg
    |      |-- 0003.jpg
    |      |-- ...
    |
    | --- Annotations
    |      |-- 0001.xml
    |      |-- 0002.xml
    |      |-- 0003.xml
    |      |-- ...
```

한가지 궁금한점이 생깁니다. MNIST, CIFAR10, COCO 데이터셋등을 보면 Train, Test 폴더가 따로 분리되어있습니다. 그러면 Train과 Test를 나누어서 Folder구조를 가지고가야 하는것일까요?<br>
반드시 나누어야 하는 특수한 상황이 아닌이상 **그럴필요는 없다** 입니다(개인적인 생각). 데이터 수집은 대부분 크롤링을 통해 마구잡이로 데이터를 수집하는경우가 대부분입니다. 이렇게 수집된 이미지파일을 하나하나씩 분리를 하는것은 비효율적이라고 생각합니다. 물론 I/O측면에서 분리를 하는것이 좋기는 합니다만, 전체 데이터를 읽어와서 Train, Test로 Split해주는 편이 더 좋을것같다는 것입니다. 간단한 조작으로도 충분히 분리할 수 있고, sklearn이라는 훌륭한 API를 제공되고 있으니 이를 활용하는것이 바람직하다 봅니다.