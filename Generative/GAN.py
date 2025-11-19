#디즈니 공주 이미지 생성
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
import os

# 디바이스 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 이미지 크기 설정 정규화
IMAGE_SIZE = 128

# RGB 이미지를 위한 정규화 (3채널)
transforms_train = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),  # 튜플로 크기 지정
    transforms.ToTensor(),  # PIL Image를 Tensor로 변환 (0~1 범위)
    # Normalize(mean=0.5, std=0.5)를 사용하는 이유:
    # 1. ToTensor() 후: 픽셀 값이 [0, 1] 범위
    # 2. Normalize 공식: (x - mean) / std
    # 3. (x - 0.5) / 0.5 = 2x - 1 → [0, 1] → [-1, 1]로 변환
    # 4. GAN의 Generator 출력층에서 tanh 활성화 함수 사용 시 [-1, 1] 범위가 적합
    #    - tanh 출력 범위: [-1, 1]
    #    - 실제 이미지도 [-1, 1]로 정규화하면 학습이 안정적
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # RGB 3채널을 [-1, 1] 범위로 정규화
])

# 손상된 이미지를 건너뛰는 커스텀 데이터셋 클래스
class SafeImageFolder(torchvision.datasets.ImageFolder):
    """손상된 이미지를 자동으로 건너뛰는 ImageFolder"""
    def __getitem__(self, index):
        while True:
            try:
                return super().__getitem__(index)
            except Exception as e:
                print(f"Warning: 이미지 로딩 실패 (인덱스 {index}), 건너뜁니다: {e}")
                # 다음 인덱스로 이동 (마지막 인덱스면 처음으로)
                index = (index + 1) % len(self.samples)
                # 무한 루프 방지: 모든 이미지가 손상된 경우를 대비
                if index == 0:
                    raise RuntimeError("모든 이미지 로딩에 실패했습니다.")

# princess 폴더 경로 설정
princess_folder = os.path.join(os.path.dirname(__file__), "princess")

# SafeImageFolder를 사용하여 princess 폴더의 모든 이미지 로드 (손상된 이미지 자동 건너뛰기)
train_dataset = SafeImageFolder(
    root=princess_folder,
    transform=transforms_train
)

# DataLoader 설정
BATCH_SIZE = 128
dataloader = DataLoader(
    train_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=True, 
    num_workers=0  # 데이터 로딩에 사용할 서브프로세스(worker) 개수
    # 0: 메인 프로세스에서 직접 로딩 (Windows에서 안정적, 디버깅 용이)
    # 1 이상: 병렬로 데이터 로딩 (Linux/Mac에서 성능 향상, Windows에서는 오류 발생 가능)
    # 권장: Windows=0, Linux/Mac=4~8 (CPU 코어 수에 따라 조정)
)

#Generator 모델 정의

# 잠재 공간 차원(Latent Dimension) 설정
# latent_dim: Generator의 입력으로 사용되는 랜덤 노이즈 벡터의 크기
# - 100은 일반적으로 사용되는 값이지만, 반드시 100이어야 하는 것은 아님
# - 하이퍼파라미터로 조정 가능 (32, 64, 100, 128, 256 등)
# 
# latent_dim의 역할:
# 1. 랜덤 노이즈 벡터의 차원: Generator는 이 랜덤 벡터를 받아 이미지를 생성
# 2. 잠재 공간의 표현력: 값이 클수록 더 복잡한 패턴을 표현 가능하지만 학습이 어려울 수 있음
# 3. 일반적인 선택:
#    - 작은 이미지(28x28, 32x32): 64~100
#    - 중간 이미지(64x64, 128x128): 100~256
#    - 큰 이미지(256x256 이상): 256~512
# 
# 100을 선택하는 이유:
# - GAN 논문에서 널리 사용된 표준 값
# - 작은 이미지부터 중간 크기 이미지까지 적절한 균형
# - 학습 안정성과 표현력의 좋은 절충점
latent_dim = 100  # 필요에 따라 64, 128, 256 등으로 변경 가능


# 생성자(Generator) 클래스 정의
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        # 하나의 블록(block) 정의
        def block(input_dim, output_dim, normalize=True):
            layers = [nn.Linear(input_dim, output_dim)]
            if normalize:
                # 배치 정규화(Batch Normalization)
                # nn.BatchNorm1d(num_features, eps=1e-05, momentum=0.1, ...)
                # - num_features: 입력 특징의 개수 (output_dim)
                # - eps: 분모에 더해지는 작은 값 (0으로 나누기 방지, 기본값 1e-05)
                # - momentum: 이동 평균(running mean/std) 업데이트에 사용 (기본값 0.1)
                #   * momentum이 높을수록 (1에 가까울수록) 이전 통계를 더 많이 유지
                #   * momentum=0.8은 매우 높은 값 (일반적으로 0.1~0.2 사용)
                layers.append(nn.BatchNorm1d(output_dim, momentum=0.8)) 
            # LeakyReLU 활성화 함수
            # nn.LeakyReLU(negative_slope=0.2, inplace=True)
            # - negative_slope=0.2: 음수 입력에 대한 기울기 (기본값 0.01)
            #   * x >= 0: f(x) = x (ReLU와 동일)
            #   * x < 0: f(x) = 0.2 * x (작은 기울기로 음수 값도 전달)
            # - inplace=True: 메모리 효율을 위해 입력 텐서를 직접 수정 
            #   * 메모리 절약 효과, 하지만 역전파 시 주의 필요
            # GAN에서 LeakyReLU를 사용하는 이유:
            # - 음수도 반환가능
            # - Generator와 Discriminator 모두에서 안정적인 학습
            layers.append(nn.LeakyReLU(0.2, inplace=True)) #리키렐루 기울기 명시
            return layers

        # 생성자 모델은 연속적인 여러 개의 블록을 가짐
        # RGB 이미지 생성: 3채널 (Red, Green, Blue)
        self.model = nn.Sequential(
            *block(latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, 3 * 128 * 128),  # RGB 3채널: 3 * 128 * 128 = 49,152
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        # img.view(): 텐서의 형태(shape)를 변경하는 reshape 연산
        # img.size(0): 배치 크기 (첫 번째 차원)
        # img.view(img.size(0), 3, 128, 128)의 의미:
        # - 입력 형태: [배치크기, 49152] (1차원 벡터, 3*128*128 = 49,152)
        # - 출력 형태: [배치크기, 3, 128, 128] (4차원 텐서: 배치, 채널, 높이, 너비)
        # 
        # 예시 (배치 크기가 32인 경우):
        # - 입력: [32, 49152] → 평탄화된 1차원 벡터
        # - 출력: [32, 3, 128, 128] → RGB 이미지 형태 (32개 이미지, 3채널, 128x128 크기)
        # 
        # 왜 reshape가 필요한가?
        # - Generator의 마지막 레이어는 1차원 벡터를 출력 (nn.Linear(1024, 3*128*128))
        # - 이미지로 사용하려면 2D 형태(높이 x 너비)로 변환해야 함
        # - PyTorch의 이미지 텐서는 [배치, 채널, 높이, 너비] 형태를 사용
        # - RGB 이미지는 채널이 3개 (Red, Green, Blue)
        img = img.view(img.size(0), 3, 128, 128)  # RGB 3채널
        return img


# 판별자(Discriminator) 클래스 정의
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            # nn.Linear(input_features, output_features)
            # 첫 번째 레이어: 3 * 128 * 128 = 49,152 (평탄화된 RGB 이미지) → 512 (은닉층 뉴런 개수)
            # 512: 첫 번째 은닉층의 뉴런(노드) 개수
            # - 입력 이미지(49,152차원)를 512차원으로 압축/변환
            # - 특징 추출을 위한 중간 표현 공간
            # - 값이 클수록 표현력이 높지만 파라미터 수와 계산량 증가
            # - RGB 이미지는 3채널이므로 3 * 128 * 128 = 49,152
            nn.Linear(3 * 128 * 128, 512),  # RGB 3채널
            nn.LeakyReLU(0.2, inplace=True),
            # 두 번째 레이어: 512 → 256 (은닉층 뉴런 개수)
            # 256: 두 번째 은닉층의 뉴런 개수 (점진적으로 차원 축소)
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            # 출력 레이어: 256 → 1 (진짜/가짜 판별 확률)
            nn.Linear(256, 1),
            nn.Sigmoid(),  # 출력을 [0, 1] 범위로 변환 (확률값)
        )

    # 이미지에 대한 판별 결과를 반환
    def forward(self, img):
        # img.view(img.size(0), -1): 이미지를 1차원 벡터로 평탄화(flatten)
        # - img.size(0): 배치 크기 (첫 번째 차원 유지)
        # - -1: 나머지 차원을 자동으로 계산하여 1차원으로 평탄화
        # 
        # 변환 과정:
        # - 입력 형태: [배치크기, 3, 128, 128] (4차원 RGB 이미지 텐서)
        # - 출력 형태: [배치크기, 49152] (2차원: 배치, 평탄화된 벡터)
        # 
        # 예시 (배치 크기가 32인 경우):
        # - 입력: [32, 3, 128, 128] → RGB 이미지 형태
        # - 출력: [32, 49152] → 평탄화된 벡터 (3*128*128 = 49,152)
        # 
        # 왜 평탄화가 필요한가?
        # - Discriminator의 첫 번째 레이어는 nn.Linear(3*128*128, 512)
        # - Linear 레이어는 1차원 입력을 받으므로 이미지를 벡터로 변환해야 함
        # - Generator의 reshape와 반대 작업 (이미지 → 벡터)
        # - RGB 이미지는 3채널이므로 3 * 128 * 128 = 49,152 차원
        flattened = img.view(img.size(0), -1)
        output = self.model(flattened)

        return output


# 생성자(generator)와 판별자(discriminator) 초기화
generator = Generator()
discriminator = Discriminator()

# device로 모델 이동 (CUDA가 있으면 GPU, 없으면 CPU 사용)
generator.to(device)
discriminator.to(device)

# 손실 함수(loss function)
adversarial_loss = nn.BCELoss()
# 손실 함수는 device 이동이 필요 없음 (자동으로 처리됨)

# 학습률(learning rate) 설정
lr = 0.0002

# 생성자와 판별자를 위한 최적화 함수
# Adam 옵티마이저의 betas 파라미터:
# betas=(beta1, beta2): 모멘텀 계수 (기본값: (0.9, 0.999))
# 
# - beta1 (첫 번째 값, 0.5): 1차 모멘텀 계수
#   * gradient의 지수 이동 평균(Exponential Moving Average)에 사용
#   * 값이 작을수록 (0.5) 최근 gradient를 더 많이 반영 → 빠른 적응
#   * 값이 클수록 (0.9) 이전 gradient를 더 많이 유지 → 안정적
#   * GAN에서는 0.5를 사용 (논문에서 제안된 값, 학습 안정성 향상)
# 
# - beta2 (두 번째 값, 0.999): 2차 모멘텀 계수
#   * gradient의 제곱의 지수 이동 평균에 사용 (분산 추정)
#   * 일반적으로 0.999 사용 (거의 변경하지 않음)
#   * 학습률의 적응적 조정에 사용
# 
# GAN에서 beta1=0.5를 사용하는 이유:
# - Generator와 Discriminator의 균형잡힌 학습
# - 빠른 수렴과 안정성의 균형
# - 원래 Adam의 0.9보다 더 안정적인 학습 (GAN 논문에서 제안)
optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

#학습
import time
import matplotlib.pyplot as plt

# 생성된 이미지 저장 폴더 생성
output_dir = "generated_images"
os.makedirs(output_dir, exist_ok=True)
print(f"생성된 이미지는 '{output_dir}' 폴더에 저장됩니다.")

# 이미지 생성 개수 정보
print(f"\n{'='*60}")
print("이미지 생성 정보:")
print(f"{'='*60}")
print(f"- 학습 중 매 배치마다 생성: {BATCH_SIZE}개 (학습용, 저장 안 됨)")
print(f"- 저장되는 이미지: sample_interval마다 25개씩 하나의 PNG 파일로 저장")
print(f"- 각 PNG 파일: 5x5 격자로 25개 이미지 포함")
print(f"{'='*60}\n")

n_epochs = 400 # 학습의 횟수(epoch) 설정
sample_interval = 2000 # 몇 번의 배치(batch)마다 결과를 출력할 것인지 설정

# 예상 저장 파일 수 계산
total_images = len(train_dataset)
batches_per_epoch = (total_images + BATCH_SIZE - 1) // BATCH_SIZE  # 올림 계산
total_batches = n_epochs * batches_per_epoch
expected_saved_files = (total_batches // sample_interval) + 1  # +1은 학습 완료 후 최종 이미지

print(f"데이터셋 크기: {total_images}개 이미지")
print(f"예상 배치 수: {total_batches}개")
print(f"예상 저장 파일 수: 약 {expected_saved_files}개 (각 파일에 25개 이미지 포함)")
print(f"총 저장될 이미지 개수: 약 {expected_saved_files * 25}개\n")

start_time = time.time()

for epoch in range(n_epochs):
    for i, (imgs, _) in enumerate(dataloader):

        # 진짜(real) 이미지와 가짜(fake) 이미지에 대한 정답 레이블 생성
        # device를 사용하여 CUDA/CPU 자동 선택
        real = torch.ones(imgs.size(0), 1, device=device)  # 진짜(real): 1
        fake = torch.zeros(imgs.size(0), 1, device=device)  # 가짜(fake): 0

        real_imgs = imgs.to(device)

        """ 생성자(generator)를 학습합니다. """
        optimizer_G.zero_grad()

        # 랜덤 노이즈(noise) 샘플링
        z = torch.normal(mean=0, std=1, size=(imgs.shape[0], latent_dim), device=device)

        # 이미지 생성
        generated_imgs = generator(z)

        # 생성자(generator)의 손실(loss) 값 계산
        g_loss = adversarial_loss(discriminator(generated_imgs), real)

        # 생성자(generator) 업데이트
        g_loss.backward()
        optimizer_G.step()

        """ 판별자(discriminator)를 학습합니다. """
        optimizer_D.zero_grad()

        # 판별자(discriminator)의 손실(loss) 값 계산
        real_loss = adversarial_loss(discriminator(real_imgs), real)
        fake_loss = adversarial_loss(discriminator(generated_imgs.detach()), fake)
        d_loss = (real_loss + fake_loss) / 2

        # 판별자(discriminator) 업데이트
        d_loss.backward()
        optimizer_D.step()

        done = epoch * len(dataloader) + i
        if done % sample_interval == 0:
            # 생성된 이미지 중에서 25개만 선택하여 5 X 5 격자 이미지에 저장 및 출력
            # 이미지를 [-1, 1] 범위에서 [0, 1] 범위로 변환 (정규화 해제)
            img_to_save = (generated_imgs.data[:25] + 1) / 2.0  # [-1, 1] -> [0, 1]
            img_to_save = torch.clamp(img_to_save, 0, 1)  # 값 범위 제한
            
            # 파일로 저장
            save_path = os.path.join(output_dir, f"epoch_{epoch}_batch_{done}.png")
            save_image(img_to_save, save_path, nrow=5, normalize=False)
            print(f"\n이미지 저장됨: {save_path}")
            
            # 화면에 출력 (선택사항)
            try:
                # 첫 5개 이미지만 화면에 출력
                # img_to_save[idx]: 배치에서 idx번째 이미지 선택
                # - img_to_save 형태: [25, 3, 128, 128] (25개 이미지, RGB 3채널, 128x128)
                # - img_to_save[0]: 첫 번째 이미지 [3, 128, 128]
                # - img_to_save[1]: 두 번째 이미지 [3, 128, 128]
                fig, axes = plt.subplots(1, 5, figsize=(15, 3))
                for idx in range(min(5, len(generated_imgs))):
                    img = img_to_save[idx].permute(1, 2, 0).cpu().numpy()
                    axes[idx].imshow(img)
                    axes[idx].axis('off')
                    axes[idx].set_title(f'Sample {idx+1}')
                plt.suptitle(f'Epoch {epoch}, Batch {done}', fontsize=14)
                plt.tight_layout()
                plt.show(block=False)  # block=False로 설정하면 학습이 계속 진행됨
                plt.pause(0.1)  # 짧은 시간 표시
            except Exception as e:
                print(f"이미지 출력 중 오류 (계속 진행): {e}")

    # 하나의 epoch이 끝날 때마다 로그(log) 출력
    print(f"[Epoch {epoch}/{n_epochs}] [D loss: {d_loss.item():.6f}] [G loss: {g_loss.item():.6f}] [Elapsed time: {time.time() - start_time:.2f}s]")

# 학습 완료 후 최종 이미지 생성 및 출력 함수
def generate_and_save_samples(generator, num_samples=25, save_path=None):
    """학습된 Generator를 사용하여 이미지를 생성하고 저장/출력"""
    generator.eval()  # 평가 모드로 전환
    
    with torch.no_grad():  # gradient 계산 비활성화 (메모리 절약)
        # 랜덤 노이즈 생성
        z = torch.normal(mean=0, std=1, size=(num_samples, latent_dim), device=device)
        
        # 이미지 생성
        generated_imgs = generator(z)
        
        # 이미지를 [-1, 1] 범위에서 [0, 1] 범위로 변환
        img_to_save = (generated_imgs + 1) / 2.0
        img_to_save = torch.clamp(img_to_save, 0, 1)
        
        # 파일로 저장
        if save_path is None:
            save_path = os.path.join(output_dir, "final_generated_samples.png")
        
        save_image(img_to_save, save_path, nrow=5, normalize=False)
        print(f"\n최종 생성 이미지 저장됨: {save_path}")
        
        # 화면에 출력
        try:
            fig, axes = plt.subplots(5, 5, figsize=(15, 15))
            axes = axes.flatten()
            
            # img_to_save[idx]의 의미:
            # - img_to_save: 배치 텐서, 형태는 [num_samples, 3, 128, 128]
            #   예: num_samples=25일 때 [25, 3, 128, 128] (25개 이미지, 각각 RGB 3채널, 128x128 크기)
            # - idx: 인덱스 (0부터 num_samples-1까지)
            # - img_to_save[idx]: 배치에서 idx번째 이미지를 선택
            #   예: img_to_save[0] → 첫 번째 이미지 [3, 128, 128]
            #       img_to_save[1] → 두 번째 이미지 [3, 128, 128]
            #       img_to_save[24] → 25번째 이미지 [3, 128, 128]
            # - .permute(1, 2, 0): 차원 순서 변경 [3, 128, 128] → [128, 128, 3]
            #   (PyTorch: [채널, 높이, 너비] → matplotlib: [높이, 너비, 채널])
            # - .cpu().numpy(): GPU 텐서를 CPU로 이동 후 NumPy 배열로 변환
            for idx in range(num_samples):
                img = img_to_save[idx].permute(1, 2, 0).cpu().numpy()
                axes[idx].imshow(img)
                axes[idx].axis('off')
                axes[idx].set_title(f'Sample {idx+1}', fontsize=8)
            
            plt.suptitle('Final Generated Images', fontsize=16)
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"이미지 출력 중 오류: {e}")
    
    generator.train()  # 다시 학습 모드로 전환


print("학습 완료!")
print(f"생성된 이미지 저장 위치: {os.path.abspath(output_dir)}")
print("\n최종 샘플 이미지를 생성합니다...")
generate_and_save_samples(generator, num_samples=25)
print("\n완료!")
