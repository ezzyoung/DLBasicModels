import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

#cpu 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#,fashion_mnist 데이터 내려받기
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = torchvision.datasets.FashionMNIST(
    root='./data',
    train=True,
    transform=transform,
    download=True
)

test_dataset = torchvision.datasets.FashionMNIST(
    root='./data',
    train=False,
    transform=transform,
    download=True
)

#데이터 로더 설정
train_loader = DataLoader(dataset=train_dataset, batch_size=100)
test_loader = DataLoader(dataset=test_dataset, batch_size=100)

#분류에 사용될 클래스 정의
labels_map = {0: "T-shirt", 1: "Trouser", 2: "Pullover", 3: "Dress", 4: "Coat", 5: "Sandal", 6: "Shirt", 7: "Sneaker", 8: "Bag", 9: "Ankle boot"}

fig = plt.figure(figsize=(8,8))
columns = 4;
rows = 5; 
for i in range(1, columns * rows + 1):
    img, label = train_dataset[i]
    fig.add_subplot(rows, columns, i)
    plt.title(labels_map[label])
    plt.axis('off')
    plt.imshow(img.squeeze(0), cmap=plt.cm.gray)
plt.show()

#심층 신경망 모델 생성
# 이 부분 주석
# class CNN(nn.Module)::
#     def __init__(self):
#         super(FashionDNN, self).__init__()
#         self.fc1 = nn.Linear(in_features=784, out_features=256)
#         self.drop = nn.Dropout(p=0.25)
#         self.fc2 = nn.Linear(in_features=256, out_features=128)
#         self.fc3 = nn.Linear(in_features=128, out_features=10)
#     def forward(self, input_data):
#         out = input_data.view(-1, 784)
#         out = F.relu(self.fc1(out))
#         out = self.drop(out)
#         out = F.relu(self.fc2(out))
#         out = self.fc3(out)
#         return out

class FashionCNN(nn.Module):
    def __init__(self):
        super(FashionCNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3,padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer2 = nn.Sequential( #계층을 차례로 쌓을 수 있게 만들어줌
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc1 = nn.Linear(in_features=64*6*6, out_features=600)
        self.drop = nn.Dropout(p=0.25)
        self.fc2 = nn.Linear(in_features=600, out_features=120)
        self.fc3 = nn.Linear(in_features=120, out_features=10) #마지막 계층의 out_features 는 클래스개수

    def forward(self,x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.drop(out)
        out = self.fc2(out)
        out = self.fc3(out)
        return out

#합성곱 위한 파라미터 정의
learning_rate = 0.001
model=FashionCNN()
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
print(model)

#모델 학습 및 평가
num_epochs = 5
count = 0
loss_list = []
iteration_list = []
accuracy_list = []

predictions_list = []
labels_list = []

for epoch in range(num_epochs):
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        train = Variable(images.view(100,1,28,28))
        labels = Variable(labels)

        outputs = model(train)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        count += 1

        # 50번째 iteration마다 평가
        if not (count % 50):
            total = 0
            correct = 0
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                test = Variable(images.view(100,1,28,28))
                outputs = model(test)
                predictions = torch.max(outputs, 1)[1].to(device)
                predictions_list.append(predictions)
                correct += (predictions == labels).sum()
                total += len(labels)
            
            accuracy = (100 * correct / total)
            loss_list.append(loss.data)
            iteration_list.append(count)
            accuracy_list.append(accuracy)
            
            # accuracy 계산 직후 출력
            print("Epoch: {} Iteration: {} Loss: {:.4f} Accuracy: {:.2f}%".format(
                epoch+1, count, loss.item(), accuracy))