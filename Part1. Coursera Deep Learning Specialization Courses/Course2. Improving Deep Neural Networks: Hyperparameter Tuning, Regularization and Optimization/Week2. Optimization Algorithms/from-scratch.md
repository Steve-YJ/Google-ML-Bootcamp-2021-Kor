## From scratch Chap6. 학습 관련 기술들

- 밑바닥부터 시작하는 딥러닝 - 사이토 고키저
- 신경망 학습의 핵심 개념 재정리

## Agenda

- 가중치 매개변수 최적값 탐색 최적화 방법
- 가중치 매개변수 초깃값
- 하이퍼파라미터 튜닝
- weight decay
- dropout
- batch normalization

## 6.1. 매개변수 갱신

Q. 신경망 학습의 목적은 무엇일까?

- A. 신경망 학습의 목적은 손실 함수의 값을 가능한 한 낮추는 매개변수를 찾는 것이다
- 즉, 매개변수의 최적값을 찾는 문제이다. 이를 최적화(Optimization)라고 한다

Q. 지금까지 배운 최적화 방법은 어떤 방법이 있을까?

- A. 지금까지는 최적의 매개변수를 찾는 단서로 매개변수의 기울기(미분)을 이용했다
- 매개변수에 대한 손실함수의 기울기를 구해, 기울어진 방향으로 매개변수 값을 갱신하는 일을 계속해서 반복하여 최적의 값에 다가갔다
- 이는 확률적 경사 하강법(SGD)라고 한다

### 6.1.2 확률적 경사 하강법(SGD)

- SGD
    - W ← w - alpha * dl/dw

    ```python
    class SGD:
    		def __init__(self, lr=0.01):
    				self.lr = lr

    		def update(self, params, grads):
    				for key in params.keys():
    						params[key] -= self.lr * grads[key]
    ```

- pseudo code

    ```python
    network = TwoLayerNet(...)
    optimizer = SGD()

    for i in range(10000):
    		...
    		x_batch, t_batch = get_mini_batch(...). # 미니배치
    		grads = network(x_batch, t_batch)
    		params = network.params
    		optimizer.update(params, grads)
    		...
    ```

### 6.1.3. SGD의 단점

> *Q.SGD의 단점은 무엇일까?*

SGD는 단순하고 구현도 쉽지만, 문제에 따라서는 비효율적일 때가 있다. SGD의 단점을 알아보도록 한다

- SGD의 단점은 비등방성(anistropy)함수에서는 탐색 경로가 비효율적이라는 것이다
    - 비등방성 함수란 방향에 따라 성질이 달라지는 함수를 의미한다
- SGD와 같이 무작정 기울어진 방향으로 탐색하는 방식보다는 더 좋은 방안이 필요하다

```python
SGD의 단점을 개선해주는 모멘텀, AdaGrad, Adam 방법을 알아보도록 한다
```

### 6.1.4. 모멘텀

모멘텀(Momentum)은 '운동량'을 뜻하는 단어로, 물리와 관계가 있다

- v = av - alpha*dl - [식 6.3]
- W -= W+v - [식 6.4]

모멘텀의 수식에서 v는 물리에서 말하는 속도(velocity)에 해당한다

식 6.3은 기울어진 방향으로 힘을 받아 물체가 가속된다는 물리 법칙을 나타낸다. av는 물체가 아무런 힘을 받지 않을 때 서서히 하강시키는 역할을 한다. 물리에서 지면 마찰이나 공기 저항에 해당한다

```python
class Momentum:
		def __init__(self, lr=0.01, momentum=0.9):
				self.lr = lr
				self.momentum = momentum
				self.v = None
		
		def update(self, params, grads):
				if self.v is None:
						self.v = {}
						for key, val in params.items():
								self.v[key] = np.zeros_like(val)

				for key in params.key():
						self.v[key] = self.momentum*self.v[key] - self.lr*grads[key]
						params[key] += self.v[key]
```
