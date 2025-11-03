import torch
import torch.nn as nn
import torch.nn.functional as F


# LSTM 네트워크 (업데이트 버전)
class LSTM(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers, seq_length):
        super(LSTM, self).__init__()
        self.num_classes = num_classes     # 분류 클래스 수
        self.num_layers = num_layers       # LSTM 계층 수
        self.input_size = input_size 
        self.hidden_size = hidden_size 
        self.seq_length = seq_length

        # LSTM 계층 선언
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True  # (batch, seq_len, input_size) 입력 형식
        )

        # fully connected layer (classifier head)
        self.fc_1 = nn.Linear(hidden_size, 128)
        self.fc   = nn.Linear(128, num_classes)

        self.relu = nn.ReLU()

    def forward(self, x):
        batch_size = x.size(0)

        # ✅ Variable 제거 + 입력 x.device 따라가기
        h_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size,
                          device=x.device, dtype=x.dtype)
        c_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size,
                          device=x.device, dtype=x.dtype)

        # LSTM 출력
        output, (hn, cn) = self.lstm(x, (h_0, c_0))
        # hn shape: (num_layers, batch_size, hidden_size)

        # ✅ 보통은 마지막 레이어의 마지막 hidden만 사용
        hn = hn[-1]  # (batch_size, hidden_size)

        out = self.relu(hn)
        out = self.fc_1(out)
        out = self.relu(out)
        out = self.fc(out)

        return out
