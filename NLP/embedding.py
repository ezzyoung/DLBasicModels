'''
TF-IDF = 문서 내부에서 많이 등장하는 단어(TF) × 전체 문서에서 드물게 등장하는 단어(IDF)
'''
from sklearn.feature_extraction.text import TfidfVectorizer
doc = ['I like coffee','I like tea and latte', 'I like running every day']
tfidf_vectorizer = TfidfVectorizer(min_df = 1) #최소 한개 문서에만 등장해도 포함
tfidf_matrix = tfidf_vectorizer.fit_transform(doc) #행렬 생성
doc_distance = (tfidf_matrix * tfidf_matrix.T) #문서간 유사도  확인
print('유사도를 위한', str(doc_distance.get_shape()[0]), 'x', str(doc_distance.get_shape()[1]),'행렬을 만들엇습니다')
print(doc_distance.toarray()) #단어 분포 비슷한 두 문서는 높은 점수 얻음

from nltk.tokenize import word_tokenize, sent_tokenize
import warnings
warnings.filterwarnings(action='ignore')
import gensim
from gensim.models import Word2Vec


sample = open('data/peter.txt', "r", encoding='UTF8')
s = sample.read() 

f = s.replace("\n", " ")
data = [] 
  
for i in sent_tokenize(f):
    temp = [] 
    for j in word_tokenize(i):
        temp.append(j.lower())
    data.append(temp) 

print(data) #토큰화 완료

#word2vec
'''
단어의 의미는 그 단어가 등장하는 주변 단어들에 의해 정의된다
참고로 이와 다르게 transformer 기반 임베딩인 BERT 나 GPT는 전체 문서를 보고 임베딩 모델을 학습하기에 효과가 더 좋다
'''
#CBOW
'''
“주변 단어들(context)을 보고, 가운데 단어(target word)를 예측"
(context words)  →  [embedding 평균]  →  (output softmax) → target word
CBOW는 여러 개의 주변 단어(Context)를 입력으로 주고,
그 문맥 속에서 등장해야 할 중심 단어(Target)를 예측하도록 학습하는 방식이다.
예측된 단어와 실제 정답(진짜 중심 단어)을 비교하여 오차(loss)를 줄이는 방향으로
임베딩(단어 벡터)을 업데이트
'''
model1 = gensim.models.Word2Vec(data, vector_size=100, window=5, sg=0)
print("Cosine similarity between peter and wendy - CBBOW", model1.wv.similarity('peter', 'wendy'))

#Skip-gram
'''
중심 단어 하나를 보고, 주변 단어들을 예측하는 방식
(target word) → [embedding] → (output softmax for each context word)
입력된 중심 단어를 기준으로, 주변 단어들이 등장할 확률을 계산하고,
실제 정답(진짜 주변 단어)과 비교해 오차를 줄이는 방향으로 임베딩을 학습
'''
model2 = gensim.models.Word2Vec(data, vector_size=100, window=5, sg=1)
print("Cosine similarity between peter and wendy - Skip-gram", model2.wv.similarity('peter', 'wendy'))

#FastText
'''
FastText는 Word2Vec의 단점을 보완하기 위해 개발된 모델로,
단어를 그대로 학습하는 것이 아니라 문자 단위 subword(n-gram)로 쪼개어 학습함으로써
희귀 단어·신조어·오타도 유사 벡터를 만들 수 있게 한 단어 임베딩 모델
'''
from gensim.test.utils import common_texts
from gensim.models import FastText
model = FastText('data/peter.txt', vector_size=4, window=3, min_count=1, epochs=10)
sim_scores = model.wv.similarity('peter', 'wendy')
print(sim_scores)

#Glove
'''
GloVe는 단어 쌍이 같은 문맥(window) 안에서 함께 등장한 횟수(co-occurrence)를 기반으로
통계 행렬을 만들고, 이를 수학적으로 분해(factorization)하여 단어 벡터를 학습하는 모델
'''
