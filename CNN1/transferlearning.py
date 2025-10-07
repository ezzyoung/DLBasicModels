import glob
import os
import time
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# ==================== 함수 정의 ====================
def train_model(model, dataloaders, criterion, optimizer, device, num_epochs=13):
    since = time.time()    
    acc_history = []
    loss_history = []
    best_acc = 0.0

    # 모델 저장 폴더 자동 생성
    save_dir = "./checkpoints"
    os.makedirs(save_dir, exist_ok=True)

    for epoch in range(num_epochs):
        print(f"Epoch {epoch}/{num_epochs - 1}")
        print('-' * 10)

        running_loss = 0.0
        running_corrects = 0

        # 학습 루프
        for inputs, labels in dataloaders:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1) #outputs에서 가장 큰 값(가장 높은 확률) 의 인덱스를 예측된 클래스(preds)로 뽑는다
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0) #배치의 손실값을 누적합에 더함
            running_corrects += torch.sum(preds == labels.data) #정답을 맞춘 개수를 누적

        # 에폭별 결과
        epoch_loss = running_loss / len(dataloaders.dataset)
        epoch_acc = running_corrects.double() / len(dataloaders.dataset)

        print(f"Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} ({epoch_acc*100:.2f}%)")

        # 최고 정확도 갱신
        if epoch_acc > best_acc:
            best_acc = epoch_acc

        acc_history.append(epoch_acc.item())
        loss_history.append(epoch_loss)

        # 에폭별 모델 가중치 저장
        save_path = os.path.join(save_dir, f"epoch_{epoch:02d}.pth")
        torch.save(model.state_dict(), save_path)
        print(f"✅ Model saved to: {save_path}\n")

    # 학습 완료 후 요약
    time_elapsed = time.time() - since
    print(f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    print(f"Best Acc: {best_acc:.4f} ({best_acc*100:.2f}%)")
    print(f"모델 가중치는 모두 {save_dir} 폴더에 저장되었습니다.")
    
    return acc_history, loss_history


# ==================== 메인 실행 부분 ====================
if __name__ == '__main__':
    # 디바이스 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    # 이미지 데이터 전처리 방법 정의
    data_path = 'C:/Users/ilsai/Downloads/data01/train'
    transform = transforms.Compose([
        transforms.Resize([256, 256]),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    
    train_dataset = torchvision.datasets.ImageFolder(
        data_path,
        transform=transform
    )
    
    # ✅ num_workers를 0으로 변경 (Windows 안정성)
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        num_workers=0,  # ← 8에서 0으로 변경!
        shuffle=True
    )

    print(f"Total training images: {len(train_dataset)}")
    print(f"Classes: {train_dataset.classes}\n")

    # ResNet18 모델 생성
    resnet18 = models.resnet18(pretrained=True)

    # 기존 파라미터 고정 (전이 학습)
    for param in resnet18.parameters():
        param.requires_grad = False

    # 마지막 FC 레이어만 교체 (2개 클래스)
    resnet18.fc = nn.Linear(512, 2)

    # 학습 가능한 파라미터 확인
    print("학습 가능한 파라미터:")
    for name, param in resnet18.named_parameters():
        if param.requires_grad:
            print(f"  {name}: {param.shape}")

    # 학습 가능한 파라미터만 옵티마이저에 전달
    params_to_update = []
    for name, param in resnet18.named_parameters():
        if param.requires_grad:
            params_to_update.append(param)

    # 옵티마이저 및 손실함수
    optimizer = optim.Adam(params_to_update, lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # 모델을 디바이스로 이동
    resnet18.to(device)

    print("\n" + "="*60)
    print("학습 시작")
    print("="*60 + "\n")

    # 모델 학습 실행
    train_acc_hist, train_loss_hist = train_model(
        resnet18, 
        train_loader, 
        criterion, 
        optimizer, 
        device,
        num_epochs=13
    )

    print("\n" + "="*60)
    print("학습 완료!")
    print("="*60)

#test data

# 디바이스 설정
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ResNet18 모델 정의 (학습 시와 동일하게)
resnet18 = models.resnet18(pretrained=False)
num_ftrs = resnet18.fc.in_features
resnet18.fc = torch.nn.Linear(num_ftrs, 2)  # 2개 클래스로 가정


#test data
def eval_model(model, dataloaders, device):
    """테스트 데이터 평가 함수"""
    since = time.time()    
    acc_history = []
    best_acc = 0.0

    saved_models = glob.glob('./checkpoints/*.pth')
    saved_models.sort()
    print('saved_model', saved_models)
    
    if len(saved_models) == 0:
        print("체크포인트 폴더에 저장된 모델이 없습니다!")
        return acc_history

    for model_path in saved_models:
        print(f'Loading model {model_path}')

        # 모델 로드
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        model.to(device)
        
        running_corrects = 0

        for inputs, labels in dataloaders:
            inputs = inputs.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                outputs = model(inputs)

            # 가장 높은 확률의 클래스 선택
            _, preds = torch.max(outputs, 1)
            
            # 정확도 계산
            running_corrects += (preds == labels).sum().item()
            
        epoch_acc = running_corrects / len(dataloaders.dataset)
        print(f'Acc: {epoch_acc:.4f}\n')
        
        if epoch_acc > best_acc:
            best_acc = epoch_acc

        acc_history.append(epoch_acc)

    time_elapsed = time.time() - since
    print(f'Validation complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best Acc: {best_acc:.4f}')
    
    return acc_history

# ============================================================
# 메인 실행 부분 - 반드시 if __name__ == '__main__': 안에!
# ============================================================
if __name__ == '__main__':
    # 디바이스 설정
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ResNet18 모델 정의 (학습 시와 동일하게)
    resnet18 = models.resnet18(weights=None)  # pretrained 대신 weights 사용
    num_ftrs = resnet18.fc.in_features
    resnet18.fc = torch.nn.Linear(num_ftrs, 2)  # 2개 클래스

    # 테스트 데이터 전처리
    test_path = 'C:/Users/ilsai/Downloads/data01/test'

    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])

    test_dataset = torchvision.datasets.ImageFolder(
        root=test_path,
        transform=transform
    )


    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=32,
        num_workers=0,  
        shuffle=False
    )

    print(f"Test dataset size: {len(test_dataset)}")
    print(f"Test classes: {test_dataset.classes}")
    print()

    # 정확도 평가
    val_acc_hist = eval_model(resnet18, test_loader, device)
    