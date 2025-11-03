#정규화 (Normalization)
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

df = pd.read_csv(r'C:\Users\ilsai\MLModels\NLP\data\diabetes.csv')
X = df[df.columns[:-1]]
y = df['Outcome'] #당뇨병인가 아닌가 레이블

X = X.values
y = torch.tensor(y.values)
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.33)

#정규화 과정
# 데이터의 값 범위를 일정한 기준으로 맞추는 작업

'''
| 구분        | MinMaxScaler                   | StandardScaler        |
| --------- | ------------------------------ | --------------------- |
| 변환 방식     | 0~1 범위로 압축                     | 평균=0, 표준편차=1로 변환      |
| 수식        | (x - 최소값) / (최대값 - 최소값)        | (x - 평균) / 표준편차       |
| 범위        | [0, 1]                         | 제한 없음                 |
| 이상치 민감도   | 매우 민감                          | 덜 민감                  |
| 주로 쓰는 케이스 | 이미지 픽셀 데이터, 0~255 같은 bounded 값 | 정규분포 근처 데이터, 딥러닝/선형모델 |
| 예시        | CNN 입력                         | 회귀, SVM, PCA, KNN     |
'''
#훈련과 테스트 데이터 정규화

ms = MinMaxScaler()
ss = StandardScaler()

X_train = ss.fit_transform(X_train)
X_test = ss.fit_transform(X_test)
y_train =y_train.reshape(-1, 1)
y_test =y_test.reshape(-1, 1)
y_train = ms.fit_transform(y_train)
y_test = ms.fit_transform(y_test)

#대량 데이터 처리 위해 데이터로더 가지고 옴

class customdataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.len = len(self.X)
    def __getitem__(self, index):
        return self.X[index], self.y[index]
    def __len__(self):
        return self.len

#데이터 담기
train_data = customdataset(torch.FloatTensor(X_train), 
                       torch.FloatTensor(y_train))
test_data = customdataset(torch.FloatTensor(X_test), 
                       torch.FloatTensor(y_test))

train_loader = DataLoader(dataset=train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset=test_data, batch_size=64, shuffle=False)

class binaryClassification(nn.Module):
    def __init__(self):
        super(binaryClassification, self).__init__()
        self.layer_1 = nn.Linear(8, 64, bias=True) 
        self.layer_2 = nn.Linear(64, 64, bias=True)
        self.layer_out = nn.Linear(64, 1, bias=True)         
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1)
        self.batchnorm1 = nn.BatchNorm1d(64)
        self.batchnorm2 = nn.BatchNorm1d(64)
        
    def forward(self, inputs):
        x = self.relu(self.layer_1(inputs))
        x = self.batchnorm1(x)
        x = self.relu(self.layer_2(x))
        x = self.batchnorm2(x)
        x = self.dropout(x)
        x = self.layer_out(x)        
        return x

#모델이 사용한 손실 함수와 옵티마이저 지정
'''
모델 출력(logit) → 확률(0~1) 로 바꾸고 → 그 확률과 정답레이블(0 또는 1) 을 비교해 오차를 계산한다는 뜻
'''
epochs = 1000+1
print_epoch = 100
LEARNING_RATE = 1e-2

model = binaryClassification()
model.to(device)
print(model)

BCE = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

#성능 측정 함수
def accuracy(y_pred, y_test):
    y_pred_tag = torch.round(torch.sigmoid(y_pred))
    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum/y_test.shape[0]
    acc = torch.round(acc * 100)    
    return acc

#모델 학습
for epoch in range(epochs):    
    iteration_loss = 0.
    iteration_accuracy = 0.
    
    model.train()
    for i, data in enumerate(train_loader):
        X, y = data
        X, y = X.to(device), y.to(device)
        y_pred = model(X.float()).to(device)
        loss = BCE(y_pred, y.reshape(-1,1).float())     
      
        iteration_loss += loss
        iteration_accuracy += accuracy(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if(epoch % print_epoch == 0):
        print('Train: epoch: {0} - loss: {1:.5f}; acc: {2:.3f}'.format(epoch, iteration_loss/(i+1), iteration_accuracy/(i+1)))
    
    iteration_loss = 0.
    iteration_accuracy = 0.
    model.eval()
    for i, data in enumerate(test_loader):
        X, y = data
        X, y = X.to(device), y.to(device)
        y_pred = model(X.float()).to(device)
        loss = BCE(y_pred, y.reshape(-1,1).float())
        iteration_loss += loss
        iteration_accuracy += accuracy(y_pred, y)
    if(epoch % print_epoch == 0):
        print('Test: epoch: {0} - loss: {1:.5f}; acc: {2:.3f}'.format(epoch, iteration_loss/(i+1), iteration_accuracy/(i+1)))