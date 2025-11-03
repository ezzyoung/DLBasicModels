from nltk.tokenize import sent_tokenize, word_tokenize
text_sample = "Natural Language Processing, or NLP, is the process of extracting that meaning or intent, behind human language. In the field of Conversational artificial intelligence, NLP allows machines and applications to understand the intent of human language imputs."
tokenized_sentences = sent_tokenize(text_sample)

print(tokenized_sentences) #문장 단위 토큰 나누기

sentence = "This book is about Harry Potter"
words = word_tokenize(sentence)
print(words) #단어 토큰 나누기

#한글 토큰화
import csv
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from gensim.models import word2vec
from konlpy.tag import Okt


with open('NLP/data/ratings_train.txt', 'r', encoding='utf-8') as f:
    rdr = csv.reader(f, delimiter='\t')
    rdw = list(rdr)


twitter= Okt()
result = []

for line in rdw:
    malist = twitter.pos(line[1], norm=True, stem=True) #형태소 분석
    r = []
    for word in malist:
        if not word[1] in ["Josa","Eomi","Production"]:
            r.append(word[0])
    
    rl = (" ".join(r)).strip() #형태소 사이에 공백 넣고 양쪽 공백 삭제
    result.append(rl)
    print(rl)

with open("NaverMovie.nlp", 'w', encoding='utf-8') as fp:
    fp.write("\n".join(result)) #형태소 별도 파일 저장장

#word2vec 모델 생성
mdata = word2vec.LineSentence("NaverMovie.nlp")
mModel = word2vec.Word2Vec(mdata, vector_size=200, window=10, hs=1, min_count=2, sg=1)
mModel.save("NaverMovie.model") #모델 저장

#불용어 제거 - 문장 내 빈번하게 발생하여 의미를 부여하기 어려운 단어 의미
import nltk 
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('punkt')
from nltk.tokenize import word_tokenize

sample_text = "The land of Narnia is a magical realm hidden beyond an ordinary wardrobe, where animals speak and ancient prophecies come to life. Guided by Aslan, the great lion"
text_tokens = word_tokenize(sample_text)

tokens_without_sw = [word for word in text_tokens if not word in stopwords.words('english')]
print('불용어 제거 미적용:', text_tokens, '\n')
print('불용어 제거 적용', tokens_without_sw)

#어간 추출, 표제어 추출
'''
어간 추출 : 단어에서 어미나 접사 등을 기계적으로 잘라내어 어간 형태만 남김 studies -> studi
표제어 추출: 단어를 사전에 등재된 기본형(표제어)로 되돌림 studies -> study
'''
#어간 추출
from nltk.stem import PorterStemmer, LancasterStemmer
stemmer = PorterStemmer()

print(stemmer.stem('obsess'), stemmer.stem('obsessed'))
print(stemmer.stem('standardizes'), stemmer.stem('standardization'))

stemmer1 = LancasterStemmer()

print(stemmer1.stem('tribalical'). stemmer1.stem('tribalicalized')) #없는 단어

#표제어 추출
import nltk
nltk.download('wordnet')

from nltk.stem import WordNetLemmatizer #표제어 추출 라이브러리
lemma = WordNetLemmatizer()

print(lemma.lemmatize('standarizes'), lemma.lemmatize('standarization'))

#정확도 더 높이려면 해당 단어 품사 정보를 넣어줌

print(lemma.lemmatize('obsesses','v'), lemma.lemmatize('obsessed','a'))
