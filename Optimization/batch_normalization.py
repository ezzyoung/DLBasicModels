import torch
import matplotlib.pyplot as plt
import numpy as np

import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.optim as optim

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

#MNIST DATA Download
trainset = torchvision.datasets.FashionMNIST(root='./data', train=True, 
                                        download=True, 
                                        transform=transforms.ToTensor())

#내려받은 데이터셋 메모리로 가져옴
batch_size = 4
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

#데이터셋 분리
dataiter = iter(trainloader)
images, labels = next(dataiter)

print(images.shape)
print(images[0].shape) #첫번째 이미지 한장
print(labels[0].item()) #첫번째 이미지의 정답 라벨

'''
기본적으로 파이토치는 이미지 데이터셋을 [배치 크기, 채널, 너비, 높이] 4차원 형태로 저장. -> 이미지 출력하려면 [너비, 높이, 채널]
그런데 matplotlib 으로  출력 위해서는 이미지가 [너비, 높이' 채널] 형태여야 함
transpose() 사용
np.transpose(img, (1, 2, 0)) -> 이미지 텐서를 인덱스 순서를 기존거에서 바꿀때 1,2,0 순서로 만든다는 뜻
'''

#배치 정규화 미적용 vs 배치 정규화 적용
#미적용
class NormalNet(nn.Module):
    def __init__(self): 
        super(NormalNet, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(784, 48),  # 28 x 28 = 784
            nn.ReLU(),
            nn.Linear(48, 24),
            nn.ReLU(),
            nn.Linear(24, 10) #클래스 10개로 분류
        )
             
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

#적용
class BNNet(nn.Module):
    def __init__(self): 
        super(BNNet, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(784, 48),
            nn.BatchNorm1d(48), #배치 단위로 정규화. 한 배치 안에 있는 여러 샘플들 각각이 '48차원 벡터'를 가지며,각각의 feature(=차원)별로 정규화를 수행
            nn.ReLU(),
            nn.Linear(48, 24),
            nn.BatchNorm1d(24), #배치 단위로 정규화
            nn.ReLU(),
            nn.Linear(24, 10)
        )
             
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

model = NormalNet()
print(model)

model_bn = BNNet()
print(model_bn)

#데이터셋 메모리로 불러오기
batch_size = 512
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

#옵티마이저 손실 함수 지정
loss_fn = nn.CrossEntropyLoss().to(device)
opt = optim.SGD(model.parameters(), lr=0.01)
opt_bn = optim.SGD(model_bn.parameters(), lr=0.01)

#모델 학습 및 결과 시각화
loss_arr = []
loss_bn_arr = []
max_epochs = 20

for epoch in range(max_epochs):
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        opt.zero_grad()
        outputs = model(inputs).to(device)        
        loss = loss_fn(outputs, labels)
        loss.backward()
        opt.step()
        
        opt_bn.zero_grad()
        outputs_bn = model_bn(inputs)
        loss_bn = loss_fn(outputs_bn, labels)
        loss_bn.backward()
        opt_bn.step()
        
        loss_arr.append(loss.item())
        loss_bn_arr.append(loss_bn.item())
           
    plt.plot(loss_arr, 'yellow', label='Normal')
    plt.plot(loss_bn_arr, 'blue', label='BatchNorm')    
    plt.legend()
    plt.show()