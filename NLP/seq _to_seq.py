#seq to seq 모델 구현
'''
teacher forcing - 정답을 넣어주는 방식 = > 빠르게 수렴 
'''
#imports
from __future__ import unicode_literals, print_function, division
from gensim.parsing import read_file
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
import pandas as pd

import os
import re
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SOS_token = 0
EOS_token = 1
MAX_LENGTH = 20

#데이터 준비
'''
문장 → 단어 나눔 → 단어를 숫자로 변환하는 사전 생성
'''
class Lang:
    def __init__(self):
        self.word2index = {}              # 단어 → 인덱스 매핑
        self.word2count = {}              # 단어 등장 빈도 저장
        self.index2word = {0: "SOS", 1: "EOS"}  # 인덱스 → 단어 매핑 (기본 토큰 포함)
        self.n_words = 2                  # 단어 개수 (초기값: SOS, EOS 포함 2개)

    def addSentence(self, sentence):
        for word in sentence.split(' '): #공백 기준으로 단어 나누고
            self.addWord(word) #단어 추가
    
    def addWord(self, word):
        if word not in self.word2index:   # 아직 등록되지 않은 단어라면
            self.word2index[word] = self.n_words     # 새 인덱스 부여
            self.word2count[word] = 1                # 빈도 1로 등록
            self.index2word[self.n_words] = word     # 인덱스 → 단어 등록
            self.n_words += 1                        # 전체 단어 개수 +1
        else:
            self.word2count[word] += 1               # 이미 있으면 빈도 증가

#데이터 정규화
def normalizeString(df, lang):
    sentence = df[lang].str.lower()   # ① 모두 소문자로 변환
    sentence = sentence.str.replace('[^A-Za-z\s]+', '')   # ② 영문자 + 공백 외 문자 제거 (구두점, 숫자, 기호 삭제)
    sentence = sentence.str.normalize('NFD')   # ③ 유니코드 정규화 (예: á → a + ´)
    sentence = sentence.str.encode('ascii', errors='ignore').str.decode('utf-8')   # ④ ASCII 외 문자 제거 (악센트 제거)
    return sentence

#두 언어 문장 각각 정규화 해서 반환
def read_sentence(df, lang1, lang2):
    sentence1 = normalizeString(df, lang1)
    sentence2 = normalizeString(df, lang2)
    return sentence1, sentence2

def read_file(loc, lang1, lang2):
    df = pd.read_csv(loc, delimiter='\t', header=None, names=[lang1, lang2])
    return df
    
def process_data(lang1,lang2):
    df = read_file('data/eng-fra.txt', lang1, lang2)
    sentence1, sentence2 = read_sentence(df, lang1, lang2)

    input_lang = Lang()
    output_lang = Lang()
    pairs = []
    for i in range(len(df)):
        if len(sentence1[i].split(' ')) < MAX_LENGTH and len(sentence2[i].split(' ')) < MAX_LENGTH:
            full = [sentence1[i], sentence2[i]]
            input_lang.addSentence(sentence1[i])
            output_lang.addSentence(sentence2[i])
            pairs.append(full)

    return input_lang, output_lang, pairs

#텐서로 변환 
# 문장을 단어로 분리하고 인덱스 반환
def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]
#딕셔너리에서 단어에 대한 인덱스 가져오고 문장 끝에 토큰 추가
def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1) #텐서 형태를 하나씩 들어가게 만들어야 해서 수정
#입력과 출력 문장을 텐서로 변환
def tensorsFromPair(input_lang, output_lang, pair):
    input_tensor = tensorFromSentence(input_lang, pair[0])
    target_tensor = tensorFromSentence(output_lang, pair[1])
    return (input_tensor, target_tensor)


#인코더
class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, embbed_dim, num_layers):
        super(Encoder, self).__init__()       
        self.input_dim = input_dim #입력층
        self.embbed_dim = embbed_dim #임베딩 계층
        self.hidden_dim = hidden_dim #은닉층
        self.num_layers = num_layers #계층 수
        self.embedding = nn.Embedding(input_dim, self.embbed_dim) #임베딩 계층 초기화화
        self.gru = nn.GRU(self.embbed_dim, self.hidden_dim, num_layers=self.num_layers) #임베딩 계층, 은닉층 차원, gru 계층 수 이용해 초기화
              
    def forward(self, src):      
        embedded = self.embedding(src).view(1,1,-1)
        outputs, hidden = self.gru(embedded)
        return outputs, hidden

#디코더
class Decoder(nn.Module):
    def __init__(self, output_dim, hidden_dim, embbed_dim, num_layers):
        super(Decoder, self).__init__()

        self.embbed_dim = embbed_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers

        self.embedding = nn.Embedding(output_dim, self.embbed_dim)
        self.gru = nn.GRU(self.embbed_dim, self.hidden_dim, num_layers=self.num_layers)
        self.out = nn.Linear(self.hidden_dim, output_dim)
        self.softmax = nn.LogSoftmax(dim=1)
      
    def forward(self, input, hidden):
        input = input.view(1, -1) #입력을 (1, 배치크기)
        embedded = F.relu(self.embedding(input))
        output, hidden = self.gru(embedded, hidden)       
        prediction = self.softmax(self.out(output[0]))      
        return prediction, hidden

#네트워크 정의
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device, MAX_LENGTH=MAX_LENGTH):
        super().__init__()
      
        self.encoder = encoder                  # 인코더 모듈 (예: GRU/LSTM 기반)
        self.decoder = decoder                  # 디코더 모듈 (예: GRU/LSTM 기반)
        self.device = device                    # 텐서를 올릴 디바이스(CPU/GPU)

    def forward(self, input_lang, output_lang, teacher_forcing_ratio=0.5):
        # input_lang: (input_seq_len, batch_size) 형태의 인덱스 텐서
        # output_lang: (target_seq_len, batch_size) 형태의 정답(타겟) 인덱스 텐서
        # teacher_forcing_ratio: Teacher Forcing 적용 확률 (0~1)

        input_length = input_lang.size(0)       # 입력 시퀀스 길이(토큰 수)
        batch_size = output_lang.shape[1]       # 배치 크기 (보통 1)
        target_length = output_lang.shape[0]    # 타겟 시퀀스 길이(토큰 수)
        vocab_size = self.decoder.output_dim    # 디코더가 예측해야 하는 단어 집합 크기(softmax 차원)

        # 최종 예측을 담을 텐서 초기화: (target_len, batch_size, vocab_size)
        outputs = torch.zeros(target_length, batch_size, vocab_size).to(self.device)

        # =========================
        # 1) ENCODER: 입력을 한 토큰씩 인코더에 넣어 hidden state를 누적 업데이트
        # =========================
        for i in range(input_length):
            # input_lang[i]: (batch_size,)  → i번째 시점의 입력 토큰 인덱스들
            # encoder_output: 보통 (batch_size, hidden_dim)
            # encoder_hidden: 다음 단계로 넘길 은닉 상태(스택 LSTM 등일 경우 튜플)
            encoder_output, encoder_hidden = self.encoder(input_lang[i])

        # 마지막 시점의 encoder_hidden을 디코더의 초기 hidden으로 사용
        # ⚠️ 여기서 device는 지역변수가 아니므로 self.device를 쓰는 편이 안전합니다.
        decoder_hidden = encoder_hidden.to(self.device)

        # 디코더의 첫 입력은 SOS 토큰
        # ⚠️ 마찬가지로 device 대신 self.device 권장, 그리고 보통 (batch_size,) 모양이어야 함
        decoder_input = torch.tensor([SOS_token], device=self.device)  # (1,) → batch_size가 1일 때 OK

        # =========================
        # 2) DECODER: 한 시점씩 다음 토큰의 분포를 예측
        # =========================
        for t in range(target_length):
            # decoder_input: 직전 시점의 입력(정답 토큰 또는 모델 예측 토큰)
            # decoder_hidden: 직전 시점의 은닉 상태
            # decoder_output: (batch_size, vocab_size) - 현재 시점의 단어 분포(로짓/확률)
            # decoder_hidden: 다음 시점에 넘길 은닉 상태
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)

            # 현재 시점 t의 예측 분포를 outputs에 저장
            outputs[t] = decoder_output

            # Teacher Forcing 적용 여부 결정 (확률적으로)
            teacher_force = random.random() < teacher_forcing_ratio

            # topk(1): 가장 확률 높은 단어의 인덱스를 얻음
            # topi 모양: (batch_size, 1) → 보통 다음 입력 토큰으로 사용
            topv, topi = decoder_output.topk(1)

            # 다음 시점의 디코더 입력 선택:
            #  - Teacher Forcing 사용 시: 정답 토큰(output_lang[t])을 그대로 사용
            #  - 미사용 시: 모델이 방금 예측한 토큰(topi)을 사용
            # ⚠️ input 변수명은 파이썬 내장함수와 겹치므로 next_input 등으로 바꾸는 것을 권장
            input = (output_lang[t] if teacher_force else topi)

            # Teacher Forcing 미사용 상태에서 EOS 예측 시 조기 종료
            # ⚠️ input이 텐서이고 배치 차원이 있을 수 있으므로 .item() 사용은 batch_size=1 가정
            if (teacher_force == False and input.item() == EOS_token):
                break

            # 다음 루프에서 decoder_input으로 사용하기 위해 모양 맞추기
            # - teacher_force일 때: output_lang[t]는 보통 (batch_size,) → 그대로 가능
            # - 아닐 때: topi는 (batch_size, 1) → 필요 시 squeeze로 (batch_size,)
            # 여기서는 배치 1 가정이므로 아래처럼 간단히 처리 가능
            decoder_input = input.squeeze(1) if input.dim() == 2 else input  # (batch_size,1)→(batch_size,)

        # 모든 시점(or 조기 종료) 처리 후, 전체 예측 분포 반환
        # 반환 모양: (target_length, batch_size, vocab_size)
        return outputs

# 모델 오차 계산 함수 정의
teacher_forcing_ratio = 0.5

def Model(model, input_tensor, target_tensor, model_optimizer, criterion):
    model_optimizer.zero_grad()
    input_length = input_tensor.size(0)
    loss = 0
    epoch_loss = 0
    output = model(input_tensor, target_tensor)
    num_iter = output.size(0)

    for ot in range(num_iter):
        loss += criterion(output[ot], target_tensor[ot])

    loss.backward()
    model_optimizer.step()
    epoch_loss = loss.item() / num_iter
    return epoch_loss

#모델 훈련 함수 정의
def trainModel(model, input_lang, output_lang, pairs, num_iteration=20000):
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.NLLLoss()
    total_loss_iterations = 0

    training_pairs = [tensorsFromPair(input_lang, output_lang, random.choice(pairs))
                      for i in range(num_iteration)]
  
    for iter in range(1, num_iteration+1):
        training_pair = training_pairs[iter - 1]
        input_tensor = training_pair[0]
        target_tensor = training_pair[1]
        loss = Model(model, input_tensor, target_tensor, optimizer, criterion)
        total_loss_iterations += loss

        if iter % 5000 == 0:
            average_loss= total_loss_iterations / 5000
            total_loss_iterations = 0
            print('%d %.4f' % (iter, average_loss))
          
    torch.save(model.state_dict(), '../chap10/data/mytraining.pt')
    return model

#모델 평가
def evaluate(model, input_lang, output_lang, sentences, max_length=MAX_LENGTH):
    with torch.no_grad():
        input_tensor = tensorFromSentence(input_lang, sentences[0])
        output_tensor = tensorFromSentence(output_lang, sentences[1])  
        decoded_words = []  
        output = model(input_tensor, output_tensor)
  
        for ot in range(output.size(0)):
            topv, topi = output[ot].topk(1)

            if topi[0].item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(output_lang.index2word[topi[0].item()])
    return decoded_words

def evaluateRandomly(model, input_lang, output_lang, pairs, n=10):
    for i in range(n):
        pair = random.choice(pairs)
        print('input {}'.format(pair[0]))
        print('output {}'.format(pair[1]))
        output_words = evaluate(model, input_lang, output_lang, pair)
        output_sentence = ' '.join(output_words)
        print('predicted {}'.format(output_sentence))

#모델 훈련
lang1 = 'eng'
lang2 = 'fra'
input_lang, output_lang, pairs = process_data(lang1, lang2)

randomize = random.choice(pairs)
print('random sentence {}'.format(randomize))

input_size = input_lang.n_words
output_size = output_lang.n_words
print('Input : {} Output : {}'.format(input_size, output_size))

embed_size = 256
hidden_size = 512
num_layers = 1
num_iteration = 75000

encoder = Encoder(input_size, hidden_size, embed_size, num_layers)
decoder = Decoder(output_size, hidden_size, embed_size, num_layers)

model = Seq2Seq(encoder, decoder, device).to(device)
 
print(encoder)
print(decoder)

model = trainModel(model, input_lang, output_lang, pairs, num_iteration)

evaluateRandomly(model, input_lang, output_lang, pairs)

#어텐션 적용된 디코더
class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size         # 디코더/임베딩/GRU의 은닉 차원 H
        self.output_size = output_size         # 출력 단어 집합 크기 |V|
        self.dropout_p = dropout_p             # 드롭아웃 비율
        self.max_length = max_length           # 인코더 타임스텝(시퀀스 길이)의 최대치

        # (어휘수 |V| → 은닉차원 H) 임베딩 테이블
        self.embedding = nn.Embedding(self.output_size, self.hidden_size)

        # 어텐션 스코어를 만들기 위한 선형층: [embedded(H) ; hidden(H)] → 길이 max_length
        # 즉 (1, 2H) → (1, max_length) : 각 인코더 타임스텝별 점수
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)

        # 컨텍스트(어텐션 적용 결과 H)와 임베딩(H)을 합친 2H를 다시 H로 축소
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)

        self.dropout = nn.Dropout(self.dropout_p)

        # 입력 크기 H, 은닉 크기 H의 1-레이어 GRU (입력은 attn_combine 결과)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)

        # 최종 단어 분포를 위한 선형층: H → |V|
        self.out = nn.Linear(self.hidden_size, self.output_size)

        '''
        Tensor shape: (32, 20, 256)

32 sentences ┐
              │  each sentence has 20 tokens ┐
              │                               │ each token is a 256-dim vector
              ▼                               ▼
[
  [ [0.12, 0.88, ..., 256 dims],   # sentence 1, word 1
    [0.31, 0.02, ..., 256 dims],   # sentence 1, word 2
     ...
  ],
  [ ... sentence 2 ... ],
  ...
]

    '''
        
    def forward(self, input, hidden, encoder_outputs):
        # 1) 임베딩: 토큰 인덱스 → (1, 1, H)
        #   - input: (1,) 같은 스칼라 인덱스 텐서
        embedded = self.embedding(input).view(1, 1, -1) #(seq_len, batch_size, embedding_dim) 이게 attention, rnn 기반 텐서 형티라고ㅏㅁ
        #transformer 은 (batch_size, seq_len, embedding_dim)
        # 2) 드롭아웃 적용 (학습 안정화/일반화)
        embedded = self.dropout(embedded)

        # 3) 어텐션 가중치 계산
        #   - embedded[0]: (1, H) / hidden[0]: (1, H)
        #   - torch.cat(..., dim=1) → (1, 2H)
        #   - self.attn(...) → (1, max_length) : 각 타임스텝별 "에너지/점수"
        #   - softmax(dim=1) → (1, max_length) : 확률 가중치(합=1)
        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)

        # 4) 가중합(컨텍스트) 계산
        #   - attn_weights: (1, max_length) → unsqueeze(0)로 (1, 1, max_length)
        #   - encoder_outputs: (max_length, H) → unsqueeze(0)로 (1, max_length, H)
        #   - 배치 행렬곱 bmm: (1, 1, max_length) @ (1, max_length, H) = (1, 1, H)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        # 5) 임베딩과 컨텍스트를 결합
        #   - embedded[0]: (1, H), attn_applied[0]: (1, H)
        #   - cat → (1, 2H)
        output = torch.cat((embedded[0], attn_applied[0]), 1)

        # 6) 2H → H로 축소 & GRU 입력 형태(1, 1, H)로 차원 추가
        output = self.attn_combine(output).unsqueeze(0)

        # 7) 비선형 활성화 (ReLU)
        output = F.relu(output)

        # 8) GRU 한 스텝 실행
        #   - input: (1, 1, H)  hidden: (1, 1, H)
        #   - output: (1, 1, H)  hidden: (1, 1, H)
        output, hidden = self.gru(output, hidden)

        # 9) 현재 시점의 단어 분포 로짓 → 로그-소프트맥스
        #   - output[0]: (1, H) → self.out → (1, |V|) → log_softmax
        output = F.log_softmax(self.out(output[0]), dim=1)

        # 반환:
        #  - output: (1, |V|) 현재 시점 단어의 로그확률
        #  - hidden: (1, 1, H) 다음 시점으로 넘길 은닉 상태
        #  - attn_weights: (1, max_length) 인코더 각 타임스텝에 대한 가중치
        return output, hidden, attn_weights
