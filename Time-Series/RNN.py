# from sympy.core import parameters   # ★ 수정: 사용 안 함. Adam에 model.parameters()를 써야 하므로 제거
import time
import string
import random
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# ★ 수정: torchtext 0.6.0에서는 legacy가 아니라 data / datasets 사용
from torchtext import data as tt_data            # ★ 수정
from torchtext import datasets as tt_datasets    # ★ 수정

start = time.time()

# ★ 수정: legacy 제거 → data.Field / data.LabelField 사용
TEXT  = tt_data.Field(lower=True, fix_length=200, batch_first=False)    # ★ 수정
LABEL = tt_data.LabelField(dtype=torch.long)                             # ★ 수정

# ★ 수정: IMDB.splits는 Field/LabelField를 받는 구버전 API
train_data, test_data = tt_datasets.IMDB.splits(TEXT, LABEL)            # ★ 수정

# 전처리: 토큰 리스트에 대해 소문자/구두점 제거/공백 제거
for example in train_data.examples:
    text = [x.lower() for x in vars(example)['text']]
    text = [x.replace("<br", "") for x in text]
    text = [''.join(c for c in s if c not in string.punctuation) for s in text]
    text = [s for s in text if s]
    vars(example)['text'] = text
    
for example in test_data.examples:
    text = [x.lower() for x in vars(example)['text']]
    text = [x.replace("<br", "") for x in text]
    text = [''.join(c for c in s if c not in string.punctuation) for s in text]
    text = [s for s in text if s]
    vars(example)['text'] = text

# 훈련/검증 분리
train_data, valid_data = train_data.split(random_state=random.seed(0), split_ratio=0.8)

# 단어/라벨 어휘 구축
TEXT.build_vocab(train_data, max_size=10000, min_freq=10, vectors=None)
LABEL.build_vocab(train_data)

print(f"Unique tokens in TEXT vocabulary: {len(TEXT.vocab)}")
print(f"Unique tokens in LABEL vocabulary: {len(LABEL.vocab)}")

# 이터레이터 준비 (0.6.0에서는 BucketIterator 사용)
BATCH_SIZE = 64
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

train_iterator, valid_iterator, test_iterator = tt_data.BucketIterator.splits(  # ★ 수정
    (train_data, valid_data, test_data),
    batch_size=BATCH_SIZE,
    device=device,
    sort=False,                  # ★ 수정: 길이정렬 비활성(간단하게)
    sort_within_batch=False
)

# 하이퍼파라미터
embedding_dim = 100
hidden_size   = 300

# RNNCell 인코더
class RNNCell_Encoder(nn.Module):
    def __init__(self, input_dim, hidden_size):
        super(RNNCell_Encoder, self).__init__()
        self.rnn = nn.RNNCell(input_dim, hidden_size)
        self.hidden_size = hidden_size                          # ★ 수정: 나중에 참조 위해 저장
    
    def forward(self, inputs):  # inputs: (seq_len, batch, emb_dim)
        bz = inputs.shape[1]
        # ★ 수정: hidden_size/디바이스 안전 처리
        ht = torch.zeros((bz, self.hidden_size),                # ★ 수정
                         device=inputs.device, dtype=inputs.dtype)
        for word in inputs:            # word: (batch, emb_dim)
            ht = self.rnn(word, ht)    # ht: (batch, hidden_size)
        return ht                      # 마지막 타임스텝 hidden

# 분류기
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # ★ 수정: 임베딩 크기 → len(TEXT.vocab) (stoi 길이와 동일하지만 더 직관적)
        self.em  = nn.Embedding(len(TEXT.vocab), embedding_dim, padding_idx=TEXT.vocab.stoi[TEXT.pad_token] if TEXT.pad_token in TEXT.vocab.stoi else 0)  # ★ 수정(패딩 안전)
        self.rnn = RNNCell_Encoder(embedding_dim, hidden_size)
        self.fc1 = nn.Linear(hidden_size, 256)
        self.fc2 = nn.Linear(256, 2)    # ★ 수정: IMDB는 이진 분류 → 출력 2
    def forward(self, x):                # x: (seq_len, batch)
        x = self.em(x)                   # (seq_len, batch, emb_dim)
        x = self.rnn(x)                  # (batch, hidden_size)
        x = F.relu(self.fc1(x))          # (batch, 256)
        x = self.fc2(x)                  # (batch, 2)
        return x

model = Net().to(device)

loss_fn  = nn.CrossEntropyLoss()
# ★ 수정: Adam에 넘길 파라미터는 model.parameters()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)      # ★ 수정

# 학습 함수
def training(epoch: int, model: nn.Module,
             trainloader: tt_data.BucketIterator,
             validloader: tt_data.BucketIterator) -> Tuple[float, float, float, float]:
    model.train()
    total, correct, loss_sum = 0, 0, 0.0

    for b in trainloader:
        x, y = b.text, b.label
        x, y = x.to(device), y.to(device)

        logits = model(x)
        loss   = loss_fn(logits, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            preds = logits.argmax(dim=1)
            total += y.size(0)
            correct += (preds == y).sum().item()
            loss_sum += loss.item() * y.size(0)

    epoch_loss = loss_sum / total if total > 0 else 0.0         # ★ 수정: 안전한 평균
    epoch_acc  = correct / total if total > 0 else 0.0

    # 검증
    model.eval()
    v_total, v_correct, v_loss_sum = 0, 0, 0.0
    with torch.no_grad():
        for b in validloader:
            x, y = b.text, b.label
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss   = loss_fn(logits, y)
            preds  = logits.argmax(dim=1)

            v_total    += y.size(0)
            v_correct  += (preds == y).sum().item()
            v_loss_sum += loss.item() * y.size(0)

    epoch_valid_loss = v_loss_sum / v_total if v_total > 0 else 0.0  # ★ 수정
    epoch_valid_acc  = v_correct / v_total if v_total > 0 else 0.0   # ★ 수정

    print('epoch:', epoch,
          'loss:', round(epoch_loss, 3),
          'accuracy:', round(epoch_acc, 3),
          'valid_loss:', round(epoch_valid_loss, 3),
          'valid_accuracy:', round(epoch_valid_acc, 3))
    return epoch_loss, epoch_acc, epoch_valid_loss, epoch_valid_acc

# 학습 실행
epochs = 5
train_loss, train_acc = [], []
valid_loss, valid_acc = [], []

for epoch in range(1, epochs + 1):                                      # ★ 수정: 1부터 보기 좋게
    ep_l, ep_a, ep_vl, ep_va = training(epoch, model, train_iterator, valid_iterator)
    train_loss.append(ep_l); train_acc.append(ep_a)
    valid_loss.append(ep_vl); valid_acc.append(ep_va)

mid = time.time()
print("train time:", round(mid - start, 2), "sec")

# 테스트 평가
def evaluate(epoch: int, model: nn.Module, testloader: tt_data.BucketIterator) -> Tuple[float, float]:
    model.eval()
    t_total, t_correct, t_loss_sum = 0, 0, 0.0
    with torch.no_grad():
        for b in testloader:
            x, y = b.text, b.label
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss   = loss_fn(logits, y)
            preds  = logits.argmax(dim=1)

            t_total    += y.size(0)
            t_correct  += (preds == y).sum().item()
            t_loss_sum += loss.item() * y.size(0)

    epoch_test_loss = t_loss_sum / t_total if t_total > 0 else 0.0   # ★ 수정
    epoch_test_acc  = t_correct / t_total if t_total > 0 else 0.0    # ★ 수정
    print('epoch:', epoch, 'test_loss:', round(epoch_test_loss, 3), 'test_accuracy:', round(epoch_test_acc, 3))
    return epoch_test_loss, epoch_test_acc

test_loss, test_acc = [], []
for epoch in range(1, 6):                                             # ★ 수정: 1~5로 출력 정렬
    tl, ta = evaluate(epoch, model, test_iterator)
    test_loss.append(tl); test_acc.append(ta)

end = time.time()
print("total time:", round(end - start, 2), "sec")

#RNN 계층 (전체 타임스텝)
class BasicRNN(nn.Module):
    def __init__ (self, n_layers,hidden_dim, n_vocab, embed_dim, n_classes, dropout_p=0.2):
        super(BasicRNN, self).__init__()
        self.n_layers = n_layers #RNN 계층 수
        self.embed = nn.Embedding(n_vocab, embed_dim) #워드 임베딩 적용
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(dropout_p)
        self.rnn = nn.RNN(embed_dim, self.hidden_dim, num_layers=self.n_layers, batch_first=True)
        self.out = nn.Linear(self.hidden_dim, n_classes)
        
    def forward(self,x):
        x=self.embed(x) #문자를 벡터로 변환
        h_0=self._init_state(batch_size=x.size(0))
        x, _ =self.rnn(x, h_0)
        h_t = x[:,-1,:] #가장 마지막에 나온 임베딩 값
        self.dropout(h_t)
        logit = torch.sigmoid(self.out(h_t))
        return logit
    def _init_state(self, batch_size=1):
        weight = next(self.parameters()).data
        return weight.new(self.n_layers, batch_size, self.hidden_dim).zero_()
