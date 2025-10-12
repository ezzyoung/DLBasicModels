'''
GoogLeNet 에 대한 간단 설명
LeNet-5를 시작으로 CNN은 일반적인 표준 구조를 가지게 됨
 - Convolutional layer가 쌓이고 그 뒤에 1개 또는 그 이상의 FC layer가 따라오는 구조
 - ImageNet과 같이 대용량 데이터에서의 요즘 트렌트는 layer의 수와 사이즈를 늘림
 - 오버 피팅을 해결하기 위해 dropout을 적용하는 것 (GoogLeNet 도 마찬가지)

기본적으로 기존 CNN(VGG, AlexNet)은 “모든 층이 같은 커널(3×3)”만 썼다. 
BUT 실제로 ‘어떤 필터 크기가 최적일지는 데이터마다 다르다’ -> 다양한 크기 (1*1, 3*3, 5*5) 병렬로 연결해서 다양한 특징 잡아냄 (concat)

그러나 바로 병렬로 사용하면 연산량이 너무 커진다. 그래서 먼저 1×1 필터로 채널 수를 줄이고 난 후 3*3, 5*5 등 적용. 

'''
'''
ResNet

Residual Connection

입력 x ───────────────► (+) ──► ReLU ──► 출력 y
        │             ▲
        ▼             │
   Conv → BN → ReLU → Conv → BN
           ↑
          └─ 이 부분이 F(x)
# =============================================================================
# ResNet(Residual Network) 구조 설명 — 주석용 요약 (복붙 가능)
# -----------------------------------------------------------------------------
# 1) 핵심 아이디어 (Residual / Skip Connection)
#    - 일반 신경망은 y = H(x)를 직접 학습한다.
#    - ResNet은 H(x) 대신 "잔차(residual)" F(x) = H(x) - x 를 학습하도록 설계한다.
#    - 출력은 y = F(x) + x  (여기서 "+ x"가 바로 Skip(Shortcut) Connection)
#    - 효과: 기울기(gradient)가 Skip 경로로 직접 전파되어 깊은 네트워크에서도
#            기울기 소실(vanishing gradient)이 완화되고 학습이 안정화됨.
#
# 2) Residual Block (잔차 블록) 두 종류
#    (A) Basic Block (ResNet-18/34)
#        x ───────────────► (+) ──► ReLU ──► y
#             │            ▲
#             ▼            │
#         Conv3x3 → BN → ReLU → Conv3x3 → BN
#       - 간단한 3×3 합성곱 두 개로 F(x)를 구성.
#       - 얕은 깊이(18, 34층)에 적합.
#
#    (B) Bottleneck Block (ResNet-50/101/152)
#        x ───────────────────────────► (+) ──► ReLU ──► y
#             │                       ▲
#             ▼                       │
#         Conv1x1 → BN → ReLU → Conv3x3 → BN → ReLU → Conv1x1 → BN
#       - 1×1(채널 축소) → 3×3(특징 추출) → 1×1(채널 복원)
#       - 연산/파라미터 효율↑, 매우 깊은 네트워크에 적합.
#
# 3) 차원 불일치(채널/공간 크기) 해결: Projection Shortcut
#    - F(x) 출력과 x를 더하려면 텐서 크기가 같아야 한다.
#    - 다운샘플링(stride=2)이나 채널 수가 바뀌는 첫 블록은 x에 1×1 Conv(+BN)를 적용해
#      크기를 맞춘다: y = F(x) + W_s x  (W_s: 1×1 Conv로 구현된 projection)
#
# 4) 네트워크 전체 구성 (예: ResNet-34, 입력 224×224 기준)
#    - Stem(초기): Conv7×7(stride=2) → BN → ReLU → MaxPool3×3(stride=2)  → 56×56
#    - Stage1: BasicBlock × 3                         (채널 64 , 56×56)
#    - Stage2: BasicBlock × 4 (첫 블록 stride=2)      (채널 128, 28×28)
#    - Stage3: BasicBlock × 6 (첫 블록 stride=2)      (채널 256, 14×14)
#    - Stage4: BasicBlock × 3 (첫 블록 stride=2)      (채널 512,  7×7)
#    - Head(출력): Global AvgPool(7×7→1×1) → FC(클래스 수)
#    ※ ResNet-50/101/152는 각 Stage가 Bottleneck Block으로 구성되고 블록 수만 다름.
#
# 5) Pre-Activation ResNet(개선형, He et al. 2016)
#    - BN→ReLU→Conv 순서(프리액티베이션)로 바꿔 잔차 경로로의 신호/기울기 흐름을 더 원활하게 함.
#    - 일반형(Original)보다 학습이 더 안정적이라는 보고가 많음.
#
# 6) Global Average Pooling(GAP)과 FC
#    - 마지막 Convolution 출력(예: N×C×7×7)을 GAP로 N×C×1×1로 축소 후 (N×C)로 펼쳐
#      Linear(FC)로 클래스 점수(logit) 산출. 파라미터 수/과적합을 줄이는 효과.
#
# 7) 학습 팁
#    - He 초기화(Kaiming), BN 사용, SGD+모멘텀(또는 AdamW), Cosine LR 등 널리 사용.
#    - 데이터 증강(RandomResizedCrop, Flip, ColorJitter 등)과 Weight Decay 권장.
#
# 8) 용어 설명 (Glossary)
#    - Residual(잔차): H(x)-x, 즉 "입력에서 얼마나 바꿀지"라는 변화량.
#    - Skip/Shortcut Connection: x를 뒤로 직접 전달하는 경로(보통 identity, 필요 시 1×1 conv).
#    - Projection Shortcut: 1×1 conv(+BN)으로 x의 크기를 F(x)에 맞춰주는 단축 경로.
#    - Basic Block: 3×3 Conv 두 개로 F(x)를 구성하는 단순 블록(ResNet-18/34).
#    - Bottleneck Block: 1×1-3×3-1×1로 채널 축소/복원하는 효율형 블록(ResNet-50/101/152).
#    - BN(BatchNorm): 배치 통계로 정규화하여 학습 안정화/수렴 가속.
#    - Pre-Activation: (BN→ReLU→Conv) 순서. Original은 (Conv→BN→ReLU).
#    - Downsampling: 해상도(H×W)를 절반으로 줄이는 것(stride=2). Stage 전환 시 자주 사용.
#    - GAP(Global Average Pooling): 공간 평균으로 채널만 남기는 풀링(7×7→1×1 등).
#
# 9) 한 줄 요약
#    - ResNet은 "입력을 그대로 더하는" Skip 연결로 잔차 F(x)를 학습하여 깊은 네트워크도
#      안정적으로 훈련되게 만든 구조. Basic/Bottleneck 블록을 Stage별로 쌓아 구성한다.
| 단계     | Conv/Pooling 작용                         | 공간 크기   | 채널 수 | 의미              |
| ------ | --------------------------------------- | ------- | ---- | --------------- |
| 입력     | 224×224×3                               | 224×224 | 3    | RGB 이미지         |
| Stem   | Conv7×7(stride=2), MaxPool3×3(stride=2) | 56×56   | 64   | 크기 1/4로 축소, 채널↑ |
| Stage1 | BasicBlock ×3                           | 56×56   | 64   | 특징 추출 시작        |
| Stage2 | stride=2                                | 28×28   | 128  | 공간 ↓, 특징 더 풍부   |
| Stage3 | stride=2                                | 14×14   | 256  | 고수준 특징 학습       |
| Stage4 | stride=2                                | 7×7     | 512  | 추상적 특징 학습       |
| Head   | GAP(7×7→1×1)                            | 1×1     | 512  | 요약(평균 풀링) 후 FC  |

추가 : 데이터 입출력 특성
| 실제 데이터  | 딥러닝 표현       | 의미           |
| ------- | ------------ | ------------ |
| 이미지 평면  | 2D 공간 (H×W)  | 픽셀 배치        |
| 색상 채널   | 3D (C×H×W)   | RGB 특징       |
| 배치 여러 장 | 4D (N×C×H×W) | 여러 이미지 묶음    |
| 문장      | 2D (L×D)     | 단어 시퀀스 × 의미축 |
| 문장 여러 개 | 3D (N×L×D)   | 배치 × 단어 × 의미 |

딥러닝에서 “차원”이란 공간의 축(axis) 이며,
실제 공간의 차원이 아니라 데이터의 구조와 특징(feature)을 표현하기 위한 수학적 축입니다.

이미지: 공간(H×W) + 색상(C) + 배치(N) → 4D

언어: 시퀀스(L) + 의미벡터(D) + 배치(N) → 3D

즉, “고차원”은 복잡한 공간이 아니라 정보가 표현되는 방향이 많은 공간을 뜻함 -> 들어가는 데이터는 정해져있음

데이터는 현실의 구조(이미지·텍스트·음성) 자체이고,
텐서는 그걸 딥러닝이 다루기 위해 수학적으로 “표현한 형태(요약 버전)”
# =============================================================================
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

import matplotlib.pyplot as plt
import numpy as np

import copy
from collections import namedtuple
import os
import random
import time

import cv2
from torch.utils.data import DataLoader, Dataset
from PIL import Image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#이미지 데이터 전처리
class ImageTransform():    
    def __init__(self, resize, mean, std):
        self.data_transform = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(resize, scale=(0.5, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ]),
            'val': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(resize),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
        }
        
    def __call__(self, img, phase):
        return self.data_transform[phase](img)

#변수 정의
size = 224
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)
batch_size = 32

#훈련 테스트 데이터셋
cat_directory = r'C:\Users\ilsai\MLModels\CNN2\dogs-vs-cats\Cat'
dog_directory = r'C:\Users\ilsai\MLModels\CNN2\dogs-vs-cats\Dog'

cat_images_filepaths = sorted([os.path.join(cat_directory, f) for f in os.listdir(cat_directory)])   
dog_images_filepaths = sorted([os.path.join(dog_directory, f) for f in os.listdir(dog_directory)])
images_filepaths = [*cat_images_filepaths, *dog_images_filepaths]    
correct_images_filepaths = [i for i in images_filepaths if cv2.imread(i) is not None] 

random.seed(42)    
random.shuffle(correct_images_filepaths)
#train_images_filepaths = correct_images_filepaths[:20000] #성능을 향상시키고 싶다면 훈련 데이터셋을 늘려서 테스트해보세요   
#val_images_filepaths = correct_images_filepaths[20000:-10] #훈련과 함께 검증도 늘려줘야 합니다
train_images_filepaths = correct_images_filepaths[:400]    
val_images_filepaths = correct_images_filepaths[400:-10]  
test_images_filepaths = correct_images_filepaths[-10:]    
print(len(train_images_filepaths), len(val_images_filepaths), len(test_images_filepaths))

#이미지에 대한 레이블 구분 - 개 이면 1, 고양이면 0

class DogvsCatDataset(Dataset):    
    def __init__(self, file_list, transform=None, phase='train'):    
        self.file_list = file_list
        self.transform = transform
        self.phase = phase
        
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):       
        img_path = self.file_list[idx]
        img = Image.open(img_path)        
        img_transformed = self.transform(img, self.phase)
        
        label = img_path.split('/')[-1].split('.')[0]
        if label == 'dog':
            label = 1
        elif label == 'cat':
            label = 0
        return img_transformed, label

#이미지 데이터셋 정의
train_dataset = DogvsCatDataset(train_images_filepaths, transform=ImageTransform(size, mean, std), phase='train')
val_dataset = DogvsCatDataset(val_images_filepaths, transform=ImageTransform(size, mean, std), phase='val')

index = 0
print(train_dataset.__getitem__(index)[0].size())
print(train_dataset.__getitem__(index)[1])

#데이터셋의 데이터를 메모리로 불러오기
train_iterator  = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_iterator = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
dataloader_dict = {'train': train_iterator, 'val': valid_iterator}

batch_iterator = iter(train_iterator)
inputs, label = next(batch_iterator)
print(inputs.size())
print(label)

#Basic Block
class BasicBlock(nn.Module):    
    expansion = 1
    
    def __init__(self, in_channels, out_channels, stride = 1, downsample = False):
        super().__init__()                
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size = 3, 
                               stride = stride, padding = 1, bias = False) #3X3 conv
        self.bn1 = nn.BatchNorm2d(out_channels)        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size = 3, #3X3 conv
                               stride = 1, padding = 1, bias = False)
        self.bn2 = nn.BatchNorm2d(out_channels)        
        self.relu = nn.ReLU(inplace = True)
        
        if downsample: #“공간 크기(Height × Width)를 줄이는 것”
            conv = nn.Conv2d(in_channels, out_channels, kernel_size = 1, 
                             stride = stride, bias = False)
            bn = nn.BatchNorm2d(out_channels)
            downsample = nn.Sequential(conv, bn)
        else:
            downsample = None        
        self.downsample = downsample
        
    def forward(self, x):       
        i = x       
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)        
        x = self.conv2(x)
        x = self.bn2(x)
        
        if self.downsample is not None:
            i = self.downsample(i)
                        
        x += i
        x = self.relu(x)
        
        return x

#병목 블록 정의
class Bottleneck(nn.Module):    
    expansion = 4
    
    def __init__(self, in_channels, out_channels, stride = 1, downsample = False):
        super().__init__()    
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size = 1, stride = 1, bias = False)
        self.bn1 = nn.BatchNorm2d(out_channels)        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = stride, padding = 1, bias = False)
        self.bn2 = nn.BatchNorm2d(out_channels)        
        self.conv3 = nn.Conv2d(out_channels, self.expansion * out_channels, kernel_size = 1, #채널 늘린다가 expansion
                               stride = 1, bias = False)
        self.bn3 = nn.BatchNorm2d(self.expansion * out_channels)        
        self.relu = nn.ReLU(inplace = True)
        
        if downsample:
            conv = nn.Conv2d(in_channels, self.expansion * out_channels, kernel_size = 1, 
                             stride = stride, bias = False) #의미가 응축된 형태 - downsample. bottleneck 의 경우 줄여서 연산 효율 늘리고 다시 확장해 표현력 유지
            bn = nn.BatchNorm2d(self.expansion * out_channels)
            downsample = nn.Sequential(conv, bn)
        else:
            downsample = None            
        self.downsample = downsample
        
    def forward(self, x):        
        i = x        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)        
        x = self.conv3(x)
        x = self.bn3(x)
                
        if self.downsample is not None:
            i = self.downsample(i)
            
        x += i
        x = self.relu(x)
    
        return x

#ResNet 모델 네트워크
class ResNet(nn.Module):
    def __init__(self, config, output_dim, zero_init_residual=False):
        super().__init__()
                
        block, n_blocks, channels = config
        self.in_channels = channels[0]            
        assert len(n_blocks) == len(channels) == 4
        
        self.conv1 = nn.Conv2d(3, self.in_channels, kernel_size = 7, stride = 2, padding = 3, bias = False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace = True)
        self.maxpool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
        
        self.layer1 = self.get_resnet_layer(block, n_blocks[0], channels[0])
        self.layer2 = self.get_resnet_layer(block, n_blocks[1], channels[1], stride = 2)
        self.layer3 = self.get_resnet_layer(block, n_blocks[2], channels[2], stride = 2)
        self.layer4 = self.get_resnet_layer(block, n_blocks[3], channels[3], stride = 2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(self.in_channels, output_dim)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)
        
    def get_resnet_layer(self, block, n_blocks, channels, stride = 1):   
        layers = []        
        if self.in_channels != block.expansion * channels:
            downsample = True
        else:
            downsample = False
        
        layers.append(block(self.in_channels, channels, stride, downsample))
        
        for i in range(1, n_blocks):
            layers.append(block(block.expansion * channels, channels))

        self.in_channels = block.expansion * channels            
        return nn.Sequential(*layers)
        
    def forward(self, x):        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)        
        x = self.avgpool(x)
        h = x.view(x.shape[0], -1)
        x = self.fc(h)        
        return x, h

ResNetConfig = namedtuple('ResNetConfig', ['block', 'n_blocks', 'channels'])

resnet18_config = ResNetConfig(block = BasicBlock,
                               n_blocks = [2,2,2,2],
                               channels = [64, 128, 256, 512])

resnet34_config = ResNetConfig(block = BasicBlock,
                               n_blocks = [3,4,6,3],
                               channels = [64, 128, 256, 512])

resnet50_config = ResNetConfig(block = Bottleneck,
                               n_blocks = [3, 4, 6, 3],
                               channels = [64, 128, 256, 512])

resnet101_config = ResNetConfig(block = Bottleneck,
                                n_blocks = [3, 4, 23, 3],
                                channels = [64, 128, 256, 512])

resnet152_config = ResNetConfig(block = Bottleneck,
                                n_blocks = [3, 8, 36, 3],
                                channels = [64, 128, 256, 512])


#모델 구조 출력
OUTPUT_DIM = 2
model = ResNet(resnet50_config, OUTPUT_DIM)
print(model)

'''
Global Average Pooling : 각 채널당 하나의 값만 남기는 과정

GAP 전 출출력:  (배치, 2048, 7, 7)
↓
Global Average Pooling
↓
출력:  (배치, 2048, 1, 1)
↓
Flatten → (배치, 2048) -> 차원수를 1차로 낮춰서 class 나누기 위해 전처리
↓
FC → (배치, num_classes)
'''