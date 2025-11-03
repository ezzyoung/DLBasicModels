"""
[AlexNet vs VGGNet 한눈 비교]

1) 커널 크기
   - AlexNet: 초반에 큰 커널(11x11 stride 4, 5x5 등) 사용.
   - VGGNet : 일관적으로 작은 커널(3x3, stride=1, padding=1)만 여러 번 쌓음.

2) 깊이(Depth)
   - AlexNet: 상대적으로 얕음(5개의 Conv + 3개의 FC).
   - VGGNet : 훨씬 깊음(VGG16/19 = 16/19개의 weight layer), 표현력 증가.

3) 파라미터 효율 & 비선형성
   - 큰 커널 1개(예: 7x7) vs 작은 3x3 커널 여러 개(예: 3개): receptive field는 비슷하지만,
     3x3을 여러 번 쓰면 ReLU 비선형성이 여러 번 들어가 표현력이 올라가고,
     파라미터 수가 줄어드는 경향이 있어 학습/일반화에 유리.
     (예시: 7x7*C^2 = 49C^2, 3x3*3회 = 27C^2)

4) 구조적 단순성
   - VGG는 "Conv(3x3) x N → MaxPool(2x2, s=2)" 블록을 반복하는 매우 규칙적인 설계.
   - 구현/이식/변형이 쉽고, 이후 모델(ResNet, Faster R-CNN 백본 등)들에 큰 영향.

5) 입력 크기 전처리
   - 둘 다 일반적으로 224x224(또는 227x227) 근방을 사용하지만,
     VGG 논문 구현은 224x224를 표준 입력으로 사용.
   - 분류기는 '복원'이 아니라 '요약'이므로, 입력 크기를 정하면 마지막 FC 입력 차원이 고정됨.

"""
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as Datasets

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#VGG 모델 정의
class VGG(nn.Module):
    def __init__(self, features, output_dim):
        super().__init__()        
        self.features = features        
        self.avgpool = nn.AdaptiveAvgPool2d(7)        
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace = True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace = True),
            nn.Dropout(0.5),
            nn.Linear(4096, output_dim),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        h = x.view(x.shape[0], -1)
        x = self.classifier(h)
        return x, h

#config 지정
vgg11_config = [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']

vgg13_config = [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']

vgg16_config = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 
                512, 'M']

vgg19_config = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 
                512, 512, 512, 512, 'M']


#vggnet 계층 정의
def get_vgg_layers(config, batch_norm):    
    layers = []
    in_channels = 3
    
    for c in config: #config 값 가져오기기
        assert c == 'M' or isinstance(c, int) #maxpooling
        if c == 'M':
            layers += [nn.MaxPool2d(kernel_size = 2)]
        else: #숫자면 합성곱층
            conv2d = nn.Conv2d(in_channels, c, kernel_size = 3, padding = 1)
            if batch_norm: #정규화 설정하면 정규화 + RELU
                layers += [conv2d, nn.BatchNorm2d(c), nn.ReLU(inplace = True)]
            else: #아니면 그냥 RELU
                layers += [conv2d, nn.ReLU(inplace = True)]
            in_channels = c
            
    return nn.Sequential(*layers) #계층반환

#모델 계층 생성
vgg11_layers = get_vgg_layers(vgg11_config, batch_norm=True)
print(vgg11_layers) #vgg layer 확인

#vgg11 전체 네트워크

OUTPUT_DIM=2 #개, 고양이 두 클래스 분류
model=VGG(vgg11_layers, OUTPUT_DIM)
print(model)