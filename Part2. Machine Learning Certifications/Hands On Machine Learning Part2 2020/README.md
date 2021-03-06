# Hands-On-Machine Learning Part2. 2020
![Hands-On-Machine-Learning](https://image.aladin.co.kr/product/23767/71/cover500/k532639960_1.jpg)

<code>해당 Repository는 한빛 미디어의 '핸즈온 머신러닝 (2판)을 학습하며 정리한 내용을 담았습니다</code> 

## 시작하며

### 필요한 기술

- Numpy, Pandas, Matplotlib 숙련도
- 대학 수준의 수학 지식
    - 미적분, 선형대수, 확률, 통계
- 파이썬에 대한 기본 지식

## 이 책의 구성

- 머신러닝에 대여
    * 머신러닝.문제.머신러닝 시스템의 종류와 기본 개념
- 전형적인 머신러닝 프로젝트 단계
- 데이터를 사용한 모델 학습
- 비용 함수 최적화하기(Loss Function 최적화인듯 싶다)
- 데이터 처리, 정제, 준비
- 특성 선택과 특성 공학
- 모델 선택과 교차 검증을 사용한 하이퍼파라미터 튜닝
- 머신러닝 도전 과제, 과소적합과 과대적합
- 차원의 저주와 차원 감소
- 학습 알고리즘: 선형 회귀, 다항 회귀, 로지스틱 회귀, K-NN, SVM, Decision Tree, Ensemble
- 데이터 차원 축소
- 군집, 밀도 추정, 이상치 탐지 방법

### 2부 - 신경망과 딥러닝

- 신경망에 대하여
- 텐서플로와 케라스를 사용한 신경망 학습
- 피드포워드 신경망, 합성곱 신경망, 순환 신경망, 인코더/디코더, 트랜스포머, 오토인코더, GAN
- 심층 신경망을 훈련시키기 위한 기법
- 강화학습
- 대용량 데이터 적재 및 전처리
- 대규모 텐서플로 모델 훈련 및 배포

## 주의사항

> 깊은 곳으로 성급하게 뛰어들지 마세요
> 

저자의 말을 빌려 머신러닝/딥러닝 학습시 주의사항에 대해 살펴보도록 하겠다. 

딥러닝이 머신러닝에서 가장 흥미진진한 분야임에는 의심의 여지가 없지만 먼저 기초적인 것들을 마스터해야 한다. 또한 대부분의 문제는 랜덤 포레스트나 앙상블 방법 같은 좀 더 간단한 기법을 사용하여 해결할 수 있다. 

딥러닝은 이미지 인식, 음성 인식, 자연어 처리와 같은 복잡한 문제에 가장 적합하다. 충분한 데이터와 컴퓨팅 성능이 있어야하며 인내도 필요하다.

나는 딥러닝에서도 어렵다는 AutoEncoder, Variational AutoEncoder부터 뛰어들었었다. OMG... :-)

## 2부에서 바뀐 내용들(Update)

1. ML 관련 주제를 추가하였습니다
    - 비지도 학습 기법(군집, 이상치 탐지, 밀도 추정, 혼합 모델 등)
    - 심층 신경망을 훈련하기 위한 다양한 방법
    - 추가적인 컴퓨터 비전 기법(Xception, SENet, YOLO를 사용한 객체 탐지, R-CNN을 사용한 시맨틱 분할 등)
    - 합성곱 신경망을 사용한 시퀀스 다루기
    - CNN과 트랜스포머
    - 순환 신경망을 사용한 언어 처리
    - GAN
2. 추가적인 라이브러리와 API, 분산 전략 API를 사용한 대규모 TF 모델 훈련 및 배포, TF 서빙, TF Addons/Seq2Seq, TensorFlow.js
    - Keras
    - Data API
    - 강화학습을 위한 TF-Agents
    - 이외의 다양한 내용들 추가
3. 최근 중요한 딥러닝 연구 결과 설명
4. 텐서플로 2사용 및 케라스 API 구현 사용
5. 기존 예제 코드 업데이트

## Reference
* Scikit-Learn** : http://scikit-learn.org
* TensorFlow** : https://www.tensorflow.org