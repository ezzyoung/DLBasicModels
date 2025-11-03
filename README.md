# MLModels

이 프로젝트는 다양한 머신러닝 모델과 딥러닝 아키텍처를 구현하고 실험하는 종합적인 머신러닝 프로젝트입니다.

## 📁 프로젝트 구조

```
MLModels/
├── CNN1/                    # CNN 기본 구현 및 Fashion-MNIST 실험
│   ├── cnn.py              # Fashion-MNIST CNN 모델
│   ├── transferlearning.py  # 전이학습 구현
│   ├── checkpoints/        # 모델 체크포인트
│   └── data/               # Fashion-MNIST 데이터셋
├── CNN2/                    # 고급 CNN 아키텍처 구현
│   ├── alexnet.py          # AlexNet 구현
│   ├── vggnet.py           # VGGNet 구현
│   ├── resnet.py           # ResNet 구현
│   ├── lenet5.py           # LeNet-5 구현
│   └── dogs-vs-cats/       # 개/고양이 분류 데이터셋
├── NLP/                     # 자연어처리 프로젝트
│   ├── preprocessing.py    # 텍스트 전처리
│   ├── preprocessing2.py   # 추가 전처리 스크립트
│   └── data/               # 네이버 영화 리뷰 데이터
├── Time-Series/             # 시계열 분석
│   ├── ARIMA.py            # ARIMA 모델
│   ├── LSTM.py             # LSTM 시계열 예측
│   ├── RNN.py              # RNN 구현
│   └── chap07-data/        # 시계열 데이터
├── requirements.txt         # 프로젝트 의존성
└── README.md               # 프로젝트 문서
```

## 🚀 주요 기능

### 1. CNN (Convolutional Neural Networks)
- **CNN1**: Fashion-MNIST 데이터셋을 이용한 기본 CNN 구현
- **CNN2**: 고급 CNN 아키텍처들
  - **AlexNet**: 2012년 ImageNet 우승 모델
  - **VGGNet**: 깊은 네트워크 구조 (VGG16/19)
  - **ResNet**: 잔차 연결을 통한 깊은 네트워크
  - **LeNet-5**: 초기 CNN 아키텍처

### 2. 자연어처리 (NLP)
- 한국어 텍스트 전처리 (형태소 분석, 토큰화)
- 네이버 영화 리뷰 감정 분석
- Word2Vec 임베딩 모델

### 3. 시계열 분석
- **ARIMA**: 전통적인 시계열 예측 모델
- **LSTM**: 장단기 메모리 네트워크
- **RNN**: 순환 신경망

## 🛠️ 설치 및 실행

### 환경 요구사항
- Python 3.7+
- PyTorch 2.8.0
- CUDA 지원 (선택사항)

### 의존성 설치
```bash
pip install -r requirements.txt
```

### 주요 의존성
- `torch==2.8.0` - PyTorch 딥러닝 프레임워크
- `torchvision==0.23.0` - 컴퓨터 비전 라이브러리
- `numpy==2.2.6` - 수치 계산
- `matplotlib==3.10.7` - 시각화
- `opencv-python==4.12.0.88` - 이미지 처리
- `tqdm==4.67.1` - 진행률 표시

## 📊 실험 결과

### CNN 모델들
- **Fashion-MNIST**: 10개 의류 카테고리 분류
- **Dogs vs Cats**: 이진 분류 문제
- 각 모델별 성능 비교 및 시각화

### 시계열 예측
- 주식 데이터 예측 (SBUX.csv)
- 판매 데이터 분석 (sales.csv)
- ARIMA vs LSTM 성능 비교

## 🔬 모델 아키텍처 상세

### ResNet (Residual Network)
- **핵심 아이디어**: Skip Connection을 통한 기울기 소실 문제 해결
- **Basic Block**: ResNet-18/34용
- **Bottleneck Block**: ResNet-50/101/152용
- 깊은 네트워크에서도 안정적인 학습 가능

### VGGNet vs AlexNet
- **VGGNet**: 3x3 커널만 사용한 깊은 네트워크
- **AlexNet**: 다양한 커널 크기 (11x11, 5x5, 3x3)
- 파라미터 효율성과 표현력의 균형

### LSTM 시계열 모델
- 시퀀스 길이에 따른 예측 성능
- 다층 LSTM 구조
- 시계열 데이터의 장기 의존성 학습

## 📈 사용법

### CNN 모델 실행
```bash
# Fashion-MNIST CNN 학습
cd CNN1
python cnn.py

# 전이학습 실험
python transferlearning.py
```

### 시계열 분석
```bash
# ARIMA 모델
cd Time-Series
python ARIMA.py

# LSTM 예측
python LSTM.py
```

### NLP 전처리
```bash
# 텍스트 전처리
cd NLP
python preprocessing.py
```

## 🎯 주요 특징

1. **교육용 설계**: 각 모델의 구조와 원리를 이해할 수 있도록 상세한 주석 포함
2. **실무 적용**: 실제 데이터셋을 사용한 실험
3. **성능 비교**: 다양한 아키텍처의 성능 비교
4. **한국어 지원**: 한국어 NLP 실험 포함
5. **시각화**: 학습 과정과 결과의 시각화

## 📝 라이선스

이 프로젝트는 교육 및 연구 목적으로 제작되었습니다.

## 🤝 기여

프로젝트 개선이나 새로운 모델 추가에 대한 기여를 환영합니다.

## 📞 문의

프로젝트에 대한 질문이나 제안사항이 있으시면 이슈를 생성해 주세요.
