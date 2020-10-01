---
layout: post
title: Tutorial[8] - Inference(끝)
comments: true
categories : [Pytorch, Tutorials]
use_math: true
---

딥러닝의 이론적인 내용은 최대한 배제하고 Pytorch를 활용하여 코딩할 수 있도록 필수로 알아야하는 내용에 대해서만 초점을 두었습니다. 

Pytorch 1.4 버전을 기준으로 공식 홈페이지에 나와있는 튜토리얼을 참고하였으며, 데이터 로드, CNN모델 설계, 학습, 추론까지의 일련의 과정을 경험해보는것이 Tutorial의 목표입니다.

<hr>

## Inference

이제 마지막으로 남은것은 추론이다. 앞서 학습을 통해 획득한 모델을 통해 새로운 이미지에 대하여 잘 분류를 해내는지 확인을 하는 작업을 수행한다.

<p align="center">
<img width="400" alt="cat" src="https://www.dropbox.com/s/zuyg6datwuni0eg/92969_25283_5321.jpg?raw=1">
</p>

<center>(이것은 고양이가 아니다...미지의 동물이다...)</center>

### 프로세스 :

```python
import torch
import MyNetwork
import cv2

def classification_img(img_path, model):

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # opencv는 BGR순서로 read한다.
    img = torch.from_numpy(img).float()

    img = img.permute(2, 0, 1).squeeze(0) # (H, W, C) -> (C, H, W) -> (1, C, H, W)
    
    model = model.to(device)
    img = img.to(device)

    model.eval()
    criterion = torch.nn.Softmax()
    
    out = model(img)
    label_idx = torch.argmax(out, dim=1)
    prob = criterion(out)
    
    return prob, label_idx

if __name__ == '__main__':
    import argparse
    import os 

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', default='/home/jsk/data/inference', help='Dir Inference_Img')
    parser.add_argument('--model_path', default='./best_model.pth.tar', help='Path checkpoint_model')
    parser.add_argument('--label_map_path', default='./label_map.name', help='Path label_map')

    args = parser.parse_args()
    
    infer_img_dir = args.input_dir
    model_path = args.model_path
    label_map_path = args.label_map_path

    # MatricTracker편에서 만들어 두었던 label_map 메서드를 활용.
    label_map = label_map(label_map_path)

    state = torch.load(model_path)

    # model편에서 만들어 두었던 MyNetwork를 활용.
    model = MyNetwork(num_class=2)
    model.load_state_dict(state['state_dict'])

    for img in os.scandir(infer_img_dir):
        img_path = os.path.join(infer_img_dir, img.name)

        pred, label_idx = classification_img(img_path, model)

        label_name = label_map[label_idx]
        print(f'Prob of {label_name}: {100 * prob:.4f}%')
```

추론을 할때는 학습 단계에서 처럼 DataLoader를 활용할 필요가없다. 일반적으로 추론은 이미지가 1장씩 Input으로 들어가게 되기 때문이며, 미래에 획득할 이미지를 추론한다는 의미에서 Real-Time을 가정했을때 Batch단위로 들어갈 일이 없기 때문이다.

(수집된 이미지를 저장해두었다가 추론하는 경우는 DataLoader를 활용하는것이 좋음)

<br>

학습시 저장되었던 모델을 초기화해두고 `model.load_state_dict()` 메서드를 통해 가중치들을 불러와야한다. **반드시 체크해야하는것은 학습에 사용되었던 모델을 그대로 호출해야한다는 것이다.** 다른 모델을 사용하는 경우에는 Layer의 순서, 채널, class의 갯수 등이 다르기 때문에 가중치들을 맵핑할 수 없게되어 Error를 발생한다.

<br>

model load가 완료되었다면, 이미지를 한장씩 model에 넣어준다. return받은 output을 통해 가장 높을 확률값과 그에 해당하는 인덱스를 뽑아 return해 주고, label_map의 Dictionary와 인덱스를 맵핑하여 class_name을 출력해주면 완성!!

<hr>

지금까지 Pytorch를 활용하여 기초적인 학습과 추론을 시작해보았다. 처음 Deep Learning을 접하는 분들은 데이터를 불러오고 이미지의 Shape을 맞추고 모델을 불러오는 등의 작업에서 많은 장벽을 느끼기 마련이다. 나또한 이렇게까지 학습하는데 많은 시간이 걸렸다. 간단해 보이기는 하지만.....

<br>

Pytorch를 접하시는 분들은 대부분 연구원(Resercher)들이기 때문에 코딩적인 부분에 어려움을 많이 느낀다고 생각한다. 부족하지만 이 튜토리얼이 그들에게 많은 도움이 되었으면 하는 바람이다.