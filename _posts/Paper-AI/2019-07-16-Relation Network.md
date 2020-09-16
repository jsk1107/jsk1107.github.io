---
layout: post
title: Recognition [1] Relation Network
excerpt: AI Paper Review_Relation Network.
comments: true
categories : [AI Paper Review, Relation Network]
use_math: true
---


## 1. 신경망의 문제

신경망을 활용한 알고리즘들은 분류(Classification)의 문제에는 매우 탁월한 성능을 보였습니다. 개, 고양이, 돼지 등 99%가 넘는 매우 높은 정확도로 분류를 합니다. 이미지에 대해서 __What?__ 이라는 질문에 대한 단순한 추론은 이제 정복했다는 것이죠. 인간은 한단계 더 높은 복합적인 추론을 하기 원했습니다.

<p align="center">
<img width="400" alt="cat" src="https://www.dropbox.com/s/grnpoas6ey7woxq/banana-cat.jpg?raw=1">
</p>


위 사진을 보고 인간은 과연 어떠한 추론을 하게 될까요? __"~~졸귀탱~~ 고양이가 바나나를 먹고 있는 중"__ 이라고 대부분 생각할 것입니다. 하지만 신경망 알고리즘들은 고양이 또는 바나나 라고 단순한 추론을 할것입니다.

이러한 문제를 해결하기위해 많은 시도들이 나왔지만 그다지 성능이 좋지는 않았습니다. 시간이 흘러 2017년 구글 딥마인드에서 이 문제에 대해 탁월한 성능을 낼 수 있는 알고리즘을 발표하게 됩니다.

## 2. Relation Network

**Relation Network[^0]**(이하 RN)는 객체와 객체간의 Relation(관계)를 추론하는 네트워크입니다. 위 사진에서는 고양이와 바나나의 관계를 추론한다는 것이죠. 바로 수식을 통해 알아보도록 하겠습니다.

$$
    RN(O) = f_{\phi}\left(\sum_{i,j}g_{\theta}(o_{i},o_{j}) \right)
$$

여기에서 $O$는 객체들의 집합이고, 함수 $f_{\phi}$, $g_{\theta}$는 MLP(Multi Layer Perceptron)입니다. 딥마인드 팀은 이러한 간단한 식을 통해 관계를 추론할 수 있고 3가지의 탁월한 이점이 있다고 말하고 있습니다.

### 2.1 관계를 학습할 수 있다.

<p align="center">
<img width="500" alt="object_pair" src="https://www.dropbox.com/s/nsqv079unvhjpmp/object_pair.png?raw=1">
</p>


예를들어 4개의 나무에 대해서 관계를 살펴본다고 하면 $\binom{4}{2}$ 만큼의 객체들이 관계가 있습니다. 모든 경우의 수가 함수 $g_{\theta}$에 입력으로 들어가서 관계를 추론한다는 것이죠. 
일반적인 상황에 대해서 적용해본다면, 모든 객체에 대해서 $i$번째 객체와 $j$번째 객체가 쌍(pair)로 $g_{\theta}$의 입력으로 들어가고 모든 관계를 추론할 수 있습니다. 그래서 딥마인드 팀은 함수 $g_{\theta}$를 Relation Function이라 정의했습니다.

### 2.2 데이터에 효율적이다.

Relation Function은 모든 객체에 대해서 단 하나의 단일함수 입니다. 이것은 모든 객체에 대하여 가중치를 공유(Share)하기 때문에 학습을 진행할때 특정 객체에 대해 Overfit되거나 bias가 줄어들 수 있습니다.

### 2.3 객체들의 집합에서 동작한다.

수식을 살펴보면 i와 j의 순서가 뒤바뀌어도 결과에는 영향을 미치지 않음을 알 수 있습니다. 즉, RN은 입력 객체들의 순서에 대해서도 불변(invariant)하고 출력 객체들의 순서에도 불변합니다. 궁극적으로 이러한 불변성은 RN의 출력이 객체에 존재하는 관계의 대표성에 관한 정보를 포함한다는것을 말합니다.

<p align="center">
<img width="550" alt="Relation_network" src="https://www.dropbox.com/s/428v6i5jgqprrfr/Relation_Network_paper.png?raw=1">
</p>


위 그림을 통해 3가지 특징을 살펴볼 수 있습니다. 이미지가 CNN 네트워크를 통해 나온 Feature Map의 형상이 d x d x k라고 하면, 1x1xk만큼의 Feature를 하나의 Object라 합니다. 그리고 관계를 추론하는 질문지에서 LSTM 네트워크를 통해나온 마지막 부분을 2개의 Object 함께 묶어서 한개의 Object Pair를 생성합니다.
Feature Map의 크기는 d x d 이므로 Object Pair는 $\binom{d^{2}}{2}$만큼 존재합니다. 그리고 이것을 Relation Function인 $g_{\theta}$에 입력으로 넣어줍니다. 이때 $g_{\theta}$는 모든 입력에 대해 동일한 $\theta$를 가지고 있기때문에 가중치를 공유하게 되는것을 확인할 수 있습니다.

## 3. Tasks

딥마인드팀은 RN의 퍼포먼스를 증명하기위해 4가지의 각기 다른 도메인 상황(Visual QA, Text-based QA, Dynamic physical system)에서 시뮬레이션을 진행했습니다. 각 데이터셋에 대하여 간단하게 알아보도록 합시다.

### 3.1 CLEVR

<p align="center">
<img width="550" alt="CLEVR" src="https://www.dropbox.com/s/g5jzcjzdeo7s1sw/Relation_Network1.png?raw=1">
</p>

기존의 Visual QA 데이터셋은 객체들간의 관계를 추론하기에는 모호하거나 언어적으로 편향되어있는 이미지였습니다. 이러한점을 보완해서 제작한 데이터셋이 CLEVR입니다. 위 그림과 같이 구, 원통과 같은 3D 랜더링된 객체 이미지를 포함하고 있으며 가장 큰 특징은 많은 질의가 현실세계와 명확한 관계가 있다는 것입니다. "구의 색깔은 무엇인가?", "원통과 같은 물질의 정육면체가 있는가?"와 같은 질문에 각 객체들간의 관계를 명확히 추론할 수 있겠지요.

CLEVR 데이터는 픽셀 형태, 상태묘사 두가지 버전이 있습니다. 픽셀버전은 일반적인 2D 형태의 이미지입니다. 상태묘사 버전은 행렬의 형태로 각 row에는 위치정보(x,y,z), 색깔(r,g,b), 모양(cube,cylinder,etc), 소재(rubber,metal,etc) 크기(small,large,etc)에 대한 메타정보가 있습니다.

### 3.2 Sort-of-CLEVR

RN 아키텍쳐가 일반적인 관계추론에 잘 적합하게위해 만든 데이터셋 입니다. 데이터의 구성은 CLEVR과 유사하게 구성을 했다는 의미에서 "Sort-of-CLEVR"라고 명명했습니다. 이 데이터셋은 relational question과 non-relational question으로 나누어져 있으며, 6개의 객체를 가지고 있습니다. 각 객체는 랜덤한 모양과 6개의 색깔로 구분되어 있어 분명하게 각 객체를 인지할 수 있습니다. 6절 결론부분의 그림을 참고하면 직관적으로 이해하실 수 있을겁니다.

### 3.3 bAbI

<p align="center">
<img width="550" alt="bAbI" src="https://www.dropbox.com/s/7bwcvvbn6plql8i/bAbI.png?raw=1">
</p>

bAbI는 텍스트 기반의 QA 데이터셋 입니다. 20개의 Task가 있으며, 각 질의는 관계의 특별한 종류에 관한것입니다. 예를들면 'Sandra picked up the football' 'Sandra went to the office'라는 사실(fact)가 있을때, 'Where is the football?' 라는 질의에는 'office'라는 답변을 하는식이지요. 

### 3.4 Dynamic physical systems

이것은 MuJoCo라는 물리엔진을 사용한 mass-spring system(스프링-질량계) 데이터셋 입니다. ~~기계공학의 진동해석에서 사용되는 모델로 우리는 그냥 넘어갑시다.~~
각기 다른 색깔을 가지고 있는 10개의 공들이 자유롭게 움직이면서 서로 충돌하거나 벽에 부딪쳐 튕겨나옵니다. 공들은 서로 보이지 않는 스프링으로 연결되어있거나 그에 준하는 강한 제약사항들로 연결되어있습니다. 이러한 연결을 통해 부과된 힘 덕분에 공이 독립적으로 움직이는것을 막아줍니다. 이렇게 되면 공들의 관계에 대해서도 알 수 있게되겠죠. 
딥마인드팀은 2가지 Task에 대해서 시뮬레이션을 했습니다. 첫번째는 공들의 연결의 존재 또는 부재를 추론하는것. 두번째는 계의 수를 집계하는것입니다.

아래 영상을 보시면 직관적으로 이해하실 수 있을겁니다.

{% include youtubeplayer.html id="zGs88bvivCY" %}

## 4. Models

RN은 Raw 데이터의 값을 그대로 입력하지 않습니다. 이미지는 CNN, Text는 LSTM을 거쳐서 나온 output을 하나의 객체로 인식하고 해당 객체를 Pairwise하여 RN에 입력하는 구조입니다. 기존 알고리즘과 입력데이터의 형태에서 차이가 있다는것을 알 수 있겠죠.

<p align="center">
<img width="550" alt="Relation_Network" src="https://www.dropbox.com/s/428v6i5jgqprrfr/Relation_Network.png?raw=1">
</p>

__Dealing with pixels__

Visual QA와 같은 이미지데이터의 처리는 우선적으로 CNN 알고리즘을 사용해야합니다. 4개의 Convolution Layer를 사용하여 d x d x k개(k는 마지막 4번째 Conv layer의 kernel의 개수)의 Feature Map을 획득합니다. 그리고 각각의 Feature에 대응되는 k차원의 Feature 집합에 대해서 object라는 이름표를 달아주게 됩니다. 위 그림에서는 노란색, 빨간색, 파란색이 각각 하나의 object가 되는것이죠.

__Conditioning RNs with question embeddings__

객체간의 관계를 알기위해서는 독립적인 질문이 있어야합니다. 그래서 위 공식의 RN function $g_{\theta}$ 를 다음과 같은 $RN = f_{\phi}(\sum_{i,j}g_{\theta}(o_{i}, o_{j}, q))$로 수정하였습니다. 여기서 q는 질문의 단어들을 LSTM의 입력으로 넣은 후의 마지막 state를 의미합니다. 

__Dealing with state descriptions__

행동묘사는 미리 만들어진 객체 관계가 존재하기 때문에(Visual QA와 같은), RN에 직접적으로 제공할 수 있습니다.

__Dealing with natural language__

bAbI와 같은 자연어 task는 일련의 객체로 변환작업을 해야만 합니다. 사전에 정의한 20개의 서포트셋(bAbI의 20개 task)이 있습니다. 그리고 질문 이전에 나오는 서포트셋에 대한 태깅작업을 해줍니다. 서포트셋이란 "Mary went to the bathroom", "John moved to the hallway", "Mary travelled to the office"와 같은것을 뜻합니다. 만약 질문이 "Where is Mary?"라고 한다면, 서포트셋의 관계를 따져봤을때 office가 적절하므로 태깅작업은 "office"라고 해줍니다. 위쪽의 bAbI 예제그림에서 1번 task를 보시면 알 수 있습니다. 태깅이 bathroom이 아니라 office라고 한점을 미루어보아 서포트셋에는 시간적 순서 혹은 상대적인 위치를 반영하고 있는듯 합니다. 그리고 새롭게 정의된 RN의 공식에서 객체 $O_{i}, O_{j}$를 얻기위해 서포트셋을 각 문장을 LSTM에 입력하고, 출력되어 나온 state의 마지막을 객체로 선정합니다. 또한 질문에 대한 q를 얻기위해 마찬가지로 LSTM을 거쳐 마지막 state를 객체로 선정합니다. 이렇게 $O_{i}, O_{j}, q$를 모두 구하여 페어를 만들어주면 끝입니다.

__Model Configuration details__

이제 모델의 상세내역을 살펴보도록 하겠습니다. CLEVR와 같은 픽셀 task의 작업에 대해서,
1. 24개의 kernel을 가진 4개의 conv layer, 활성함수로는 ReLU, Batch Normalization을 사용.
2. 질문에 대한 task에 대해서는 128개의 unit의 LSTM 프로세스.
3. RN의 $g_{\theta}$는 4개의 MLP Layer로 구성. Layer당 256개의 unit.
4. $f_{\theta}$는 3개의 MLP Layer로 구성. Layer는 256, 256, 29(마지막 Layer)개의 unit.
5. 최종 Layer는 Softmax. 나머지 Layer는 ReLU로 구성.
6. 손실함수는 cross-entropy 사용. learning_rate는 $2.5e^{-4}$
7. Backbone으로는 ResNet 또는 VGG 사용.

## 5. Result

__CLEVR from pixels__

CLEVR은 논문 출시기준(2017.06) 95.5%의 SOTA를 기록했습니다. 사람보다도 높은 수준으로 관계를 설명한다는것입니다. RN은 단지 visual Problem에서만 사용되는것 뿐만아니라, 관계추론을 해야하는 작업단위에서 응용될 수 있다고 합니다.

<p align="center">
<img width="550" alt="Result_CLEVR" src="https://www.dropbox.com/s/dt082bxym0197kb/Result_CLEVR.PNG?raw=1">
</p>

실제로 다음에 Review를 할 Temporal Relation Reasoning이라는 Paper에서는 이미지들의 관계를 추론하여 Action Recognition을 하였습니다.

__Sort-of-CLEVR from pixels__

sort of CLEVR 데이터에 대해서는 relational, non-relational question RN이 적용된 CNN에서는 94%를 상회하는 정확도를 보여주었지만, MLP+CNN은 non-relational question에서 63%의 정확도밖에 가지지 못하였습니다. 이것은 MLP로 구성된 모델은 객체들의 관계를 요구하는 작업에서는 적하지 않다는것을 알 수 있습니다. 

<p align="center">
<img width="550" alt="Result_Sort-of-CLEVR" src="https://www.dropbox.com/s/hy58xemg8ziql5a/Soft-of-CLEVFR_task.PNG?raw=1">
</p>

## 6. Conclusion

해당 논문에서는 CNN, LSTM, MLP를 사용하여 Relational Reaoning에 대한 Task에 대해서 상당한 퍼포먼스를 보여준다는것을 알 수 있었습니다. 제가 생각하기에 주목할만한 2가지는 PlugIn 방식으로 네트워크에 자유롭게 삽입할 수 있다는것과, 관계에 대한 설명이 super-Human performance(인간을 뛰어넘는)라는 점입니다. 세상은 필히 객체과 객체들간의 관계로 이어집니다. 독립접으로 작용하는 것은 지극히 제한적이지요. 그래서 해당 논문은 매우 의미있는 연구였다고 생각합니다. 향후 논문들도 RN을 이용하거나 변형하여 다양한 분야에서 적용이 되고 있고, 그 결과 또한 상당히 개선되었음을 알 수 있습니다.

#### Reference

[^0]: https://arxiv.org/pdf/1706.01427.pdf