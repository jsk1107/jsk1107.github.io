---
layout: post
title: Object Detection [3] Faster R-CNN
excerpt: AI Paper Review_Faster R-CNN
comments: true
categories : [AI Paper Review, Faster R-CNN]
use_math: true
---

R-CNN, Fast R-CNN에서 Region Proposal을 구하는 방법은 Selective Search를 통해 구했다.

하지만 이 방법은 CPU를 통한 연산이었으며 신경망에 포함되어있지 않았다. 따라서 모든 Pixel에 대해서 연산을 하고 병합을 해야하기 때문에 가장 큰 병목이었다. 

Faster R-CNN은 이러한 병목을 해결하였으며, 논문의 제목답게 **Faster R-CNN Towards Real-Time Object Detection with Region Proposal Netwoks**로써 실시간 탐지가 가능하였다.

이 병목을 어떻게 해결하였을까??

## Concept

Faster R-CNN은 두가지 모듈로 구성이 되어있다. 

1. Region Proposal을 만드는 영역.(RPN)
2. Fast R-CNN 영역.

신경망이 두개의 모듈을 포함하고 있기 때문에, 모듈간 FeatureMap의 연결을 어떠한 방법으로 할것인지도 고려해야한다. 논문에서는 3가지 방법을 제안하였다.

1. Alternating training.
2. Approximate joint trainning
3. Non-Approximate joint tranning

<p align="center">
<img width="400" height="400" alt="Faster_R_CNN" src="https://www.dropbox.com/s/8040npeg82p7vc2/Faster_R_CNN.PNG?raw=1">
</p>



### 1. Region Proposal Networks(RPN)

Faster R-CNN의 가장 큰 특징은 **Anchor Box 개념을 도입하여 기존 SS를 대체**하였으며, **Region Proposal을 신경망에 포함**시켰다는 것이다.

신경망 내부에서 역전파를 통해 연산을 수행하게 되므로 Bounding Box Regression을 매우 빠르게 수행할 수 있다.

또한 CPU연산을 GPU연산으로 대체가능할 수도 있기 때문에, GPU 환경에서는 더욱 빠른 속도를 획득할 수 있던 것이다. 

RPN도 네트워크이기 때문에 손실함수가 필요하다. 여기서 사용하는 손실함수는 아래와 같다.

$$
    L({p_{i}}, {t_{i}}) = \frac{1}{N_{cls}}\sum_{i}L_{cls}(p_{i}, p{i}^{*}) + \lambda\frac{1}{N_{reg}}\sum_{i}p_{i}^{*}L_{reg}(t_{i},t_{i}^{*})
$$

Classification Loss에서는 RoI가 객체인지 아닌지를 추정하며, Regression Loss에서는 좌표값은 추정하게 된다. 

자세한 내용은 **Appendix** 참조

### 2. Fast RCNN

RPN을 수행한 후 넘겨받은 좌표값을 활용하여 ROI Pooling을 수행하고, Classification과 Box Regression을 수행한다.

기존 Fast R-CNN에서 하던 방법과 동일하게 동작한다. 즉, 여기에서도 손실함수가 등장한다.

$$
    L(p, u, t^{u}, v) = L_{cls}(p,u) + \lambda[u\geq1]L_{loc}(t^{u},v)
$$


여기까지 읽어보면 무언가 헷갈리는점이 발생한다.

RPN에서도 Loss Function이 존재하고, Fast RCNN에서도 Loss Function이 존재한다는 점이다.

전혀 이상한것이 아니다. 왜냐하면 이것은 **네트워크가 2개**이기 때문이다.

RPN영역 + Fast RCNN 영역을 하나로 묶어서 Faster-RCNN이라고 부른것이므로 각각의 영역에서 Loss Function이 존재하는것은 당연하다.

때문에 Two-Stage 모델이라고 부른다.

자세한 내용은 Appendix 참조

### 3. Sharing Features for RPN and Fast R-CNN

RPN과 Fast R-CNN 두 네트워크를 독립적으로 학습을 진행하는것이 아닌라 convolutional layers를 공유하는 형태로 학습을 진행하게 된다.

1. Alternating training.

우선 RPN 학습을 진행한다. 획득한 Proposals로부터 Fast R-CNN을 학습한다. 

Fast R-CNN의 학습이 끝난 후 Bouding Box Regression을 통해 수정된 좌표값을 통해 RPN의 좌표값을 초기화한다(Update).
 
그리고 이 과정을 반복한다. 논문에서는 이 방법으로 모든 실험을 진행하였으나 과정이 복잡하여 더이상 사용하지 않는 방법이다.

2. Approximate joint trainning

RPN와 Fast R-CNN 네트워크를 하나의 네트워크로 병합한다.

순전파 진행시 RPN으로 부터 획득한 Proposals을 고정된것으로(상수취급) Fast R-CNN 영역에 좌표정보를 전달해주게 된다. 넘겨받은 좌표정보를 이용하여 Fast R-CNN의 Loss를 계산하게 됩니다. 이때 RPN에서 구했던 Loss를 같이 합쳐서 병합된 Loss를 구합니다.

이제 RPN Loss와 Fast R-CNN Loss가 합쳐진 신호가 역전파를 진행하면, Fast R-CNN영역, RPN영역, CNN영역까지 모든 Layer에 가중치를 공유할 수 있습니다(**RoI Pooling을 통해 얻은 k개의 Anchor간 공유가 아니라 Anchor내 공유!!**).

1번 방법보다 25~50%가량 학습 속도를 단축시켰다고 합니다.(Pytorch 내부 코드 형태)

3. Non-Approximate joint tranning

RPN으로부터 예측한 Bounding Box정보는 함수의 형태입니다. Fast R-CNN에서 역전파를 진행했을 때, Bounding Box정보에 대한 값을 미분하여 값을 획득하여 업데이트를 해준다면 보다 정확한 좌표정보를 얻을 수 있을것입니다. 반면 2번의 방법은 Bounding Box 정보를 고정된 값(상수) 형태로 Fast R-CNN에 넘겨주기 때문에 비교적 정확한 값을 얻을 수 없습니다.

이 방법을 사용하려면 Bounding Box에 대해 미분이 가능한 RoI Pooling layer가 필요로 하며 RoI warping이라는 방법을 사용하여 해결할 수 있다고만 제시하였습니다.



## Appendix

### Region Proposal Networks(RPN)

RPN은 Anchor라고 부르는 사각형 모양의 Box를 slinding window방식으로 탐색을 합니다. Anchor는 Size, Aspect Ratio가 각기 다른 k개의 Anchor들이 있습니다. 각각의 Anchor를 통해 4개의 좌표정보를 가진 reg layer와 2개의 label을 가진 cls layer를 획득합니다.

<p align="center">
<img width="400" alt ="Anchor" src="https://www.dropbox.com/s/fe91khui6mwbwmw/Anchor.PNG?raw=1">
</p>

위 그림에서는 CNN을 통해 나온 Feature Map의 Parameter수가 256임을 알 수 있습니다. 이 Feature Map에서 얻은 256개의 Parameter에서 각각 4+2개의 정보를 뽑아내고 Anchor가 총 9개 있으므로, $256 \times (4+2) \times 9$ 만큼의 Parameter가 생성이 됩니다. 계산되어야 하는 Parameter의 수가 상당히 줄어듬으로써 연산량도 빨라지고 Overfitting에 대한 위험도 피할 수 있다고 합니다.

#### 2.1.1 Loss Function

RPN을 학습하기 위해서 각각의 Anchor에 binary class label(Object, Not Object)을 달아줘야 합니다. 그리고 Ground-Truth Box와 IOU를 계산하는데 두가지 조건에 대하여 label을 달아줍니다.

##### Positive Label
1. Anchor와 GT의 IOU가 가장 높은 Anchor
2. Anchor와 GT의 IOU가 0.7이상인 Anchor

##### Negative Label
1. Anchor와 GT의 IOU가 0.3이하인 Anchor

이렇게 Label을 달아주었다면, RoI Pooling을 시행하여 Feature Map의 size를 동일하게 만들어준 후, Loss Function으로는 Multi-task loss를 사용해줍니다.

$$
    L({p_{i}}, {t_{i}}) = \frac{1}{N_{cls}}\sum_{i}L_{cls}(p_{i}, p{i}^{*}) + \lambda\frac{1}{N_{reg}}\sum_{i}p_{i}^{*}L_{reg}(t_{i},t_{i}^{*})
$$

i는 위에서 적어놓은 조건을 만족한 Positive, Negative Label의 갯수입니다. Anchor가 Positive일 경우 Ground-Truth label $p_{i}^{\ast}$ 는 1, Negative일 경우에는 0이 됩니다. $t_{i}$ 는 예측한 4개의 Bounding Box 좌표를 의미하고, $t_{i}^{\ast}$ 는 Ground-Truth의 좌표를 의미합니다.

$L_{cls}$는 object인지 아닌지에 대한 log loss이고 $L_{reg}$는 smooth L1 Loss입니다. 또한 $L_{reg}$ 텀에 곱하기로 붙어있는 $p_{i}^{\ast}$는 Anchor가 Positive일 경우($p_{i}^{\ast}=1$)에만 활성화 되게 됩니다. 즉 Not object일 경우에는 $L_{reg}$의 Loss자체가 0이 되버려 가중이 업데이트에 영향을 주지 않게되는 구조입니다. 각각의 텀 앞에 붙어있는 $N_{cls}, N_{reg}$는 정규화를 하기위한 것입니다. 해당 논문에서는 mini-batch size가 256이므로 $N_{cls}=256$이 되고, Anchor의 갯수가 약 2400개 이므로 $N_{reg}=2400$이 됩니다. $N_{reg}$텀 앞에 붙어있는 $\lambda$를 10으로 저정해주면 대량 256에 근사한 값을 가지게 됨으로써, $L_{cls}, L_{reg}$ 두 텀의 균형이 맞춰지게 되는 구조입니다.

Box Regression을 보면 기존 Fast R-CNN과 다른점이 하나 있습니다. Fast R-CNN에서는 댜양한 크기의 Proposals로부터 RoI Pooling하여 Bounding Box Regression을 수행했습니다. 그렇기 때문에 역전파를 하게되면 가중치가 모든 Proposals에 공유되었습니다. 하지만 Anchor를 활용하면 3(size) x 3(aspect ratio)의 고정된 크기로 RoI Pooling하여 Regression을 진행하기 때문에 9개(3x3이므로)의 Regressor를 각각 학습할 수 있습니다. Regressor간에는 가중치를 공유하지 않으므로 더욱 정확한 좌표정보를 획득할 수 있습니다.

#### 2.1.2 Training RPN

각각의 mini-batch는 1장의 이미지로부터 256개의 Anchor를 획득합니다. 128개는 Positive, 128개는 Negative로 1:1비율로 선정합니다. 만약 Positive Anchor가 128개가 안된다면, Negative Anchor를 추가합니다.(**이 방법을 적용하더라도 Positive Anchor의 수는 압도적으로 적다. 이 방법을 보완하기위해 RetinaNet탄생**)



### 3. Implementation Details

실제로 학습의 진행은 다음과 같이 진행하였습니다. 이미지의 W, H중 작은 쪽의 길이가 600이 되도록 re-size를 해줍니다. CNN의 마지막 conv layer의 Sliding Window stride는 16을 설정하였습니다. stride의 크기를 줄이면 정확도는 향상되기는 하지만, 16으로도 좋은 결과를 보여주었다고 합니다. Anchor에 대해서는 128, 256, 512 size와 1:1, 1:2, 2:1의 aspect ratios를 적용하였습니다.

일반적으로 1000x600으로 re-size된 이미지에 대하여 Sliding Window의 stride가 16이므로 $W=1000/16=62.5$이고 $H=600/16=37.5$가 되므로 약 $60 \times 40 \times 9 = 20000$ 개의 Anchor가 생성됩니다.

이미지의 경계부분에 Anchor가 가로질러나가는 것이 존재하는데, 이를 cross-boundary anchor라고 합니다. 학습시에 Loss에 기여하지 못하므로 이 부분은 제외해주어야 합니다. 이러한 부분을 제거해주면 남아있는 Anchor는 약 6,000개가 됩니다.

<p align="center">
<img width="400" alt="cross-boundary-anchors" src="https://www.dropbox.com/s/sa44v866nakp8zg/cross_boundary_anchor.png?raw=1">
</p>
<center>cross-boundary-anchor</center>
<br>

마지막으로 쓸데없이 겹치는 Anchor들이 상당히 많을것인데, 이것을 감소시키기 위해서 IOU가 0.7 이상인 조건하에서 NMS를 적용합니다. 여기까지 진행하면 약 2000개의 Anchor가 완성되게 됩니다(즉, NMS를 조절하면 2000개보다 훨씬 적은 Anchor가 만들 수 있습니다).

### 4. Result

Selective Search는 CPU로 연산처리를 할 수 밖에 없지만 RPN은 GPU로 작업이 가능하기 때문에 상당한 fps를 보여줍니다. proposal의 수도 상당히 줄었으며 mAP는 보다 증가되었음을 보여줍니다. PASCAL VOC 2007 데이터셋으로는 73.2%의 mAP와 이미지당 198ms(약 0.2s)의 Inference Time을 내었다고 합니다.


<p align="center">
<img width="600" alt="result" src="https://www.dropbox.com/s/8ifeodma3ma05lo/result.PNG?raw=1">
</p>
<p align="center">
<img width="600" alt="result" src="https://www.dropbox.com/s/ac0rg78ewe1r6h3/result_1.PNG?raw=1">
</p>
<center>RPN의 효과는 대단했다...!!</center>
<br>


### 6. Conclusion

Faster R-CNN은 RPN영역과 Fast R-CNN영역의 두 네트워크를 합친구조로 Two-Stage Detection으로 알려져 있습니다. 비록 현재는 구시대의 유물로 사라져가고 있지만, 진정한 실시간 객체 검출을 향한 발판을 마련해주었습니다.

이후 사람들은 두개의 Stage를 하나로 합쳐서 계산할수는 없을까? RPN영역에서 아예 분류까지 모두 처리할 수 있지 않을까? 라는 의문을 가지게 되었고, 이러한점을 개선시켜 바로 YoLo가 탄생하게 되었습니다.


#### Reference

[^0]: https://arxiv.org/pdf/1506.01497.pdf