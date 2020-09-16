---
layout: post
title: 케라스 창시자에게 배우는 딥러닝-1
comments: true
categories : [Keras, Tutorials]
use_math: true
---

딥러닝을 학습하는 기본서 중 하나로 추천되는 교재입니다. 해당 교재를 통해 학습하며 관련 내용을 정리해보겠습니다. 교재와 깃허브 내용인 https://github.com/rickiepark/deep-learning-with-python-notebooks 의 자료를 활용하여 스스로 학습한 내용입니다.
따라서 교재와는 구성이 다를 수 있으며, 내용에 따라 개념이 미흡한 부분이 있을 수 있습니다
<hr>

## 1. 딥러닝이란 무엇인가?

### 1.1 인공지능과 머신러닝, 딥러닝

<p align="center">
<img width="400" src="https://www.dropbox.com/s/bjer2o4g1rj9h9k/img1-1.PNG?raw=1">
</p>

#### 1.1.1 인공지능

- 보통의 사람이 수행하는 지능적인 작업을 자동화하기 위한 연구 활동
- 머신러닝과 딥러닝을 포괄하는 종합적인 분야

#### 1.1.2 머신러닝

- 프로그래머가 직접 만든 데이터 처리 규칙 대신 컴퓨터가 데이터를 보고 자동으로 규칙을 학습 하는것
- 명시적으로 프로그램되는 것이  아니라 훈련(training)됨
- 작업과 관련있는 많은 샘플을 제공하여 통계적 구조를 찾아 그 작업을 자동화하기 위한 규칙을 만들어 냄