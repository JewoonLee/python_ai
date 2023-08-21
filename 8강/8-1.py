# 합성곱
# 합성곱은 마치 입력 데이터에 마법의 도장을 찍어서 유용한 특성만 드러나게 하는 것
# 합성곱은 밀집층과 다르게 일부에 가중치를 곱함
# 처음에 일부만 만들어진 가중치랑 절편이 그 다음 일부에서도 쓰임
# 밀집층 뉴런은 입력 개수만큼 10의 가중치를 가지고 1개의출력을 함
# 합성곱 뉴런은 3개의 가중치를 가지고 8개의 출력을 함
# 합성곱 신경망(convolution neural network,CNN)
# 여기서는 뉴런을 필터 혹은 커널이라고 부름
# 합성곱은 2차에서도 적용할 수 있음
# 2차원은 도장도 2차원이야함
# 이렇게 4*4 행렬을 2*2 행렬로 만들 수 있음 => 특성 맵이라고 부름
# 필터를 여러개 만들 수 있음
# 이러면 2차원 배열의 필터가 3개 있으면 합성곱도 3개 생기기때문에 3차원이 됨

# 케라스 합성곱 층
# Conv2D는 왼->오->위->아래 순서
from tensorflow import keras
keras.layers.Conv2D(10,kernel_size=(3,3),activation='relu')
# 첫번째 매개변수 => 필터 개수
# 두번재 매배변수 => 필터의 크기
# 세번째 매개변수 => 활성화 함수
# 합성곱 신경망: 1개 이상의 합성곱 층을 쓴 인공 신경망

# 패딩, 스트라이드 
# 합성곱을 이용하면 입력층과 출력층의 크기가 달라짐 => 따라서 더 큰 출력 크기가 있는 것 처럼 만들어 입력층과 츨력층 크기를 같게 할 수 있음
# 이렇게 입력 배열 주위를 가상 원소로 채우는 것을 패딩이라고 함
# 특성 맵 크기를 동일하게 만들기 위해 입력 주위에 0으로 패딩 하는 것: 세임 패딩
# 패딩 없이 순수 입력배열만 사용 하는 것: 밸리드 패딩 => 문제는 픽셀의 사용 빈도가 달라짐
# 세임 패딩을 이용하면 가에 있는 픽셀도 많이 사용
keras.layers.Conv2D(10,kernel_size=(3,3),activation='relu',padding='same')   

# 스트라이드는 커널도장이 움직이는 크기를 나타냄
# 스트라이드는 거의 1로 통일

# 풀링: 풀링은 특성 맵 가로세로 크기를 줄이는 역할 
# 최대 쿨링: 영역에서 가장 큰 값
# 평균 쿨링: 영역 평균값 
# 거의 크기 2를 사용 => 크기가 절감됨

# 합성곱 신경망 전체 구조
# 1. 합성곱 층(세임 페딩)
# 2. 폴링 층
# 3. 이전에 사용했던 밀집층 이용

# 컬러 이미지를 사용한 합성곱
# 컬러 이미지는 RGB 3개의 값이 필요하기 때문에 3차원임
# 커널 배열의 깊이는 항상 입력 깊이와 같음
# 사실 케라스의 합성곱은 3차원 입력을 기대함
# 비슷한 경우 => 합성곱 층- 풀링 층 다음에 또 합성곱 층이 올때
# 예를 들어 첫번째 합성곱 층의 필터 개수가 5=> 특성맵:(4,4,5) => 그럼 2번째 핕터는 3,3,5가 되야함
# 따라서 이렇게 계속하면 신경망 너비와 높이는 줄어드는데 깊이는 계속 깊어짐
# 마지막에 출력층 전에 특성 맵을 모두 펼처서 밀집층으로 만듦
# 합성곱 신경망에서 필터는 이미지에 어떤 특징을 찾는다고 생각하면 됨
# 처음에는 기본적인 특징 찾고, 층이 깊어질수록 다양하고 구체적인 특징 찾도록 필터 개수 늘림
# 또 어떤 특징 이미지 어디 위치해 있더라고 쉽게 알 수 있게 너비와 높이 차원을 줄임


