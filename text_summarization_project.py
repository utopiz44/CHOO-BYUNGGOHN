# %% [markdown]
# # 텍스트 요약 프로젝트 (Text Summarization)
# - **Abstractive**: seq2seq (3-stacked LSTM Encoder + LSTM Decoder) + Bahdanau Attention
# - **Extractive** : TextRank (summa)
# - **Dataset**    : Amazon Fine Food Reviews (Reviews.csv)
#
# 루브릭 매핑
# 1. 전처리(분석→정제→정규화/불용어→분리→인코딩) ............ Cell 2~7
# 2. 학습 확인(loss 그래프 + 핵심단어 포함) .................. Cell 8~11
# 3. Extractive vs Abstractive 비교(표) ..................... Cell 12~13

# %%
# =========================================================
# Cell 0. 라이브러리
# =========================================================
import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup

import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')

from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Layer
from tensorflow.keras import backend as K
import tensorflow as tf

import warnings
warnings.filterwarnings('ignore')

stop_words = set(stopwords.words('english'))


# %%
# =========================================================
# Bahdanau Attention Layer
#  - 출처: thushv89/attention_keras (Aiffel 노드에서 사용하는 그 레이어)
#  - 별도 attention.py 로 분리해도 됨. TF 2.x 기준.
# =========================================================
class AttentionLayer(Layer):
    """Bahdanau attention (https://arxiv.org/pdf/1409.0473.pdf)."""
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        assert isinstance(input_shape, list)
        self.W_a = self.add_weight(name='W_a',
                                   shape=tf.TensorShape((input_shape[0][2], input_shape[0][2])),
                                   initializer='uniform', trainable=True)
        self.U_a = self.add_weight(name='U_a',
                                   shape=tf.TensorShape((input_shape[1][2], input_shape[0][2])),
                                   initializer='uniform', trainable=True)
        self.V_a = self.add_weight(name='V_a',
                                   shape=tf.TensorShape((input_shape[0][2], 1)),
                                   initializer='uniform', trainable=True)
        super(AttentionLayer, self).build(input_shape)

    def call(self, inputs, verbose=False):
        assert type(inputs) == list
        encoder_out_seq, decoder_out_seq = inputs

        def energy_step(inputs, states):
            assert isinstance(states, (list, tuple)), "States must be an iterable."
            W_a_dot_s = K.dot(encoder_out_seq, self.W_a)
            U_a_dot_h = K.expand_dims(K.dot(inputs, self.U_a), 1)
            reshaped_Ws_plus_Uh = K.tanh(W_a_dot_s + U_a_dot_h)
            e_i = K.squeeze(K.dot(reshaped_Ws_plus_Uh, self.V_a), axis=-1)
            e_i = K.softmax(e_i)
            return e_i, [e_i]

        def context_step(inputs, states):
            c_i = K.sum(encoder_out_seq * K.expand_dims(inputs, -1), axis=1)
            return c_i, [c_i]

        def create_inital_state(inputs, hidden_size):
            fake_state = K.zeros_like(inputs)
            fake_state = K.sum(fake_state, axis=[1, 2])
            fake_state = K.expand_dims(fake_state)
            fake_state = K.tile(fake_state, [1, hidden_size])
            return fake_state

        fake_state_c = create_inital_state(encoder_out_seq, encoder_out_seq.shape[-1])
        fake_state_e = create_inital_state(encoder_out_seq, encoder_out_seq.shape[1])

        last_out, e_outputs, _ = K.rnn(energy_step, decoder_out_seq, [fake_state_e])
        last_out, c_outputs, _ = K.rnn(context_step, e_outputs, [fake_state_c])
        return c_outputs, e_outputs

    def compute_output_shape(self, input_shape):
        return [
            tf.TensorShape((input_shape[1][0], input_shape[1][1], input_shape[1][2])),
            tf.TensorShape((input_shape[1][0], input_shape[1][1], input_shape[0][1]))
        ]


# %%
# =========================================================
# Cell 1. (전처리 1단계 - 분석) 데이터 로드 & 둘러보기
# =========================================================
# Kaggle: https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews
data = pd.read_csv("Reviews.csv", nrows=100000)   # 메모리 고려해 10만행만 사용
data = data[['Text', 'Summary']]
print("원본 샘플 수:", len(data))

# 중복/결측 제거
data.drop_duplicates(subset=['Text'], inplace=True)
data.dropna(axis=0, inplace=True)
print("중복/결측 제거 후:", len(data))
data.head()


# %%
# =========================================================
# Cell 2. (전처리 2단계 - 정제 & 3단계 - 정규화/불용어 제거)
# =========================================================
# 약어(contraction) 사전
contractions = {
    "ain't": "is not", "aren't": "are not", "can't": "cannot", "could've": "could have",
    "couldn't": "could not", "didn't": "did not", "doesn't": "does not", "don't": "do not",
    "hadn't": "had not", "hasn't": "has not", "haven't": "have not", "he'd": "he would",
    "he'll": "he will", "he's": "he is", "how'd": "how did", "how's": "how is",
    "i'd": "i would", "i'll": "i will", "i'm": "i am", "i've": "i have",
    "isn't": "is not", "it'd": "it would", "it'll": "it will", "it's": "it is",
    "let's": "let us", "ma'am": "madam", "mightn't": "might not", "mustn't": "must not",
    "shan't": "shall not", "she'd": "she would", "she'll": "she will", "she's": "she is",
    "shouldn't": "should not", "should've": "should have", "that's": "that is",
    "there's": "there is", "they'd": "they would", "they'll": "they will",
    "they're": "they are", "they've": "they have", "wasn't": "was not", "we'd": "we would",
    "we'll": "we will", "we're": "we are", "we've": "we have", "weren't": "were not",
    "what'll": "what will", "what're": "what are", "what's": "what is", "what've": "what have",
    "where's": "where is", "who'll": "who will", "who's": "who is", "won't": "will not",
    "wouldn't": "would not", "would've": "would have", "you'd": "you would",
    "you'll": "you will", "you're": "you are", "you've": "you have"
}

def preprocess_sentence(sentence, remove_stopwords=True):
    sentence = sentence.lower()                                   # 소문자화
    sentence = BeautifulSoup(sentence, "lxml").text              # HTML 태그 제거
    sentence = re.sub(r'\([^)]*\)', '', sentence)               # 괄호 내용 제거
    sentence = re.sub('"', '', sentence)                        # 따옴표 제거
    sentence = ' '.join([contractions[t] if t in contractions else t
                         for t in sentence.split(" ")])          # 약어 정규화
    sentence = re.sub(r"'s\b", "", sentence)                    # 소유격 제거
    sentence = re.sub("[^a-zA-Z]", " ", sentence)              # 영문 외 문자 제거
    sentence = re.sub('[m]{2,}', 'mm', sentence)               # mmm -> mm
    if remove_stopwords:   # 본문(Text)은 불용어 제거
        tokens = ' '.join(w for w in sentence.split() if w not in stop_words and len(w) > 1)
    else:                  # 요약(Summary)은 문장 형태 유지 위해 불용어 보존
        tokens = ' '.join(w for w in sentence.split() if len(w) > 1)
    return tokens

# 본문은 불용어 제거, 요약은 불용어 유지
clean_text    = [preprocess_sentence(s, True)  for s in data['Text']]
clean_summary = [preprocess_sentence(s, False) for s in data['Summary']]

data = pd.DataFrame({'text': clean_text, 'summary': clean_summary})
data.replace('', np.nan, inplace=True)   # 정제 후 빈 문자열 -> 결측 처리
data.dropna(axis=0, inplace=True)
print("정제 후 샘플 수:", len(data))


# %%
# =========================================================
# Cell 3. (전처리 - 길이 분석) 최대 길이 결정
# =========================================================
text_len    = [len(s.split()) for s in data['text']]
summary_len = [len(s.split()) for s in data['summary']]

print('본문 길이  평균/최대: %.1f / %d' % (np.mean(text_len), max(text_len)))
print('요약 길이  평균/최대: %.1f / %d' % (np.mean(summary_len), max(summary_len)))

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1); plt.hist(text_len, bins=40);    plt.title('Text length')
plt.subplot(1, 2, 2); plt.hist(summary_len, bins=40); plt.title('Summary length')
plt.tight_layout(); plt.show()

# 분포를 보고 결정 (Amazon Reviews 표준값)
text_max_len    = 50
summary_max_len = 8

def below_threshold_len(max_len, lengths):
    cnt = sum(1 for l in lengths if l <= max_len)
    print('전체 중 %d 이하 비율: %.4f' % (max_len, cnt / len(lengths)))

below_threshold_len(text_max_len, text_len)
below_threshold_len(summary_max_len, summary_len)

# 최대 길이 이하 샘플만 사용
data = data[data['text'].apply(lambda x: len(x.split()) <= text_max_len)]
data = data[data['summary'].apply(lambda x: len(x.split()) <= summary_max_len)]
print("길이 필터 후 샘플 수:", len(data))


# %%
# =========================================================
# Cell 4. 디코더용 시작/종료 토큰 부착
# =========================================================
data['summary'] = data['summary'].apply(lambda x: 'sostoken ' + x + ' eostoken')
data.head()


# %%
# =========================================================
# Cell 5. (전처리 4단계 - 데이터셋 분리)
# =========================================================
encoder_input = np.array(data['text'])
decoder_input = np.array(data['summary'])

indices = np.arange(encoder_input.shape[0])
np.random.shuffle(indices)
encoder_input = encoder_input[indices]
decoder_input = decoder_input[indices]

n_test = int(len(encoder_input) * 0.2)
encoder_input_train, encoder_input_test = encoder_input[n_test:], encoder_input[:n_test]
decoder_train_raw,  decoder_test_raw    = decoder_input[n_test:], decoder_input[:n_test]
print('train:', len(encoder_input_train), ' / test:', len(encoder_input_test))


# %%
# =========================================================
# Cell 6. (전처리 5단계 - 인코딩) 정수 인코딩 + 희귀단어 제거 + 패딩
# =========================================================
# ---- 본문 토크나이저 ----
src_tokenizer = Tokenizer()
src_tokenizer.fit_on_texts(encoder_input_train)

threshold = 7
total_cnt = len(src_tokenizer.word_index)
rare_cnt = sum(1 for _, c in src_tokenizer.word_counts.items() if c < threshold)
src_vocab = total_cnt - rare_cnt + 1
print('본문 단어집합 크기:', src_vocab)

src_tokenizer = Tokenizer(num_words=src_vocab)
src_tokenizer.fit_on_texts(encoder_input_train)
encoder_input_train = src_tokenizer.texts_to_sequences(encoder_input_train)
encoder_input_test  = src_tokenizer.texts_to_sequences(encoder_input_test)

# ---- 요약 토크나이저 ----
tar_tokenizer = Tokenizer()
tar_tokenizer.fit_on_texts(decoder_train_raw)

threshold = 6
total_cnt = len(tar_tokenizer.word_index)
rare_cnt = sum(1 for _, c in tar_tokenizer.word_counts.items() if c < threshold)
tar_vocab = total_cnt - rare_cnt + 1
print('요약 단어집합 크기:', tar_vocab)

tar_tokenizer = Tokenizer(num_words=tar_vocab)
tar_tokenizer.fit_on_texts(decoder_train_raw)
decoder_input_train = tar_tokenizer.texts_to_sequences(decoder_train_raw)
decoder_input_test  = tar_tokenizer.texts_to_sequences(decoder_test_raw)

# ---- 요약이 사실상 비어있는(토큰 sos/eos만 남은) 샘플 제거 ----
def drop_empty(enc, dec):
    keep = [i for i, s in enumerate(dec) if len(s) > 2]   # sos,eos 외 단어 1개 이상
    return [enc[i] for i in keep], [dec[i] for i in keep]

encoder_input_train, decoder_input_train = drop_empty(encoder_input_train, decoder_input_train)
encoder_input_test,  decoder_input_test  = drop_empty(encoder_input_test,  decoder_input_test)
print('정제 후 train:', len(encoder_input_train), ' / test:', len(encoder_input_test))

# ---- 패딩 ----
encoder_input_train = pad_sequences(encoder_input_train, maxlen=text_max_len, padding='post')
encoder_input_test  = pad_sequences(encoder_input_test,  maxlen=text_max_len, padding='post')
decoder_padded_train = pad_sequences(decoder_input_train, maxlen=summary_max_len, padding='post')
decoder_padded_test  = pad_sequences(decoder_input_test,  maxlen=summary_max_len, padding='post')

# ---- teacher forcing: 입력(마지막 토큰 제외) / 정답(첫 토큰 제외) ----
decoder_input_train  = decoder_padded_train[:, :-1]
decoder_target_train = decoder_padded_train[:, 1:]
decoder_input_test   = decoder_padded_test[:, :-1]
decoder_target_test  = decoder_padded_test[:, 1:]
print('shape:', encoder_input_train.shape, decoder_input_train.shape, decoder_target_train.shape)


# %%
# =========================================================
# Cell 7. 모델 정의 (seq2seq + Attention)
# =========================================================
embedding_dim = 128
hidden_size   = 256

# ----- Encoder: 3-stacked LSTM -----
encoder_inputs = Input(shape=(text_max_len,))
enc_emb = Embedding(src_vocab, embedding_dim)(encoder_inputs)

encoder_lstm1 = LSTM(hidden_size, return_sequences=True, return_state=True,
                     dropout=0.4, recurrent_dropout=0.4)
output1, h1, c1 = encoder_lstm1(enc_emb)

encoder_lstm2 = LSTM(hidden_size, return_sequences=True, return_state=True,
                     dropout=0.4, recurrent_dropout=0.4)
output2, h2, c2 = encoder_lstm2(output1)

encoder_lstm3 = LSTM(hidden_size, return_sequences=True, return_state=True,
                     dropout=0.4, recurrent_dropout=0.4)
encoder_outputs, state_h, state_c = encoder_lstm3(output2)

# ----- Decoder -----
decoder_inputs = Input(shape=(None,))
dec_emb_layer = Embedding(tar_vocab, embedding_dim)
dec_emb = dec_emb_layer(decoder_inputs)

decoder_lstm = LSTM(hidden_size, return_sequences=True, return_state=True,
                    dropout=0.4, recurrent_dropout=0.2)
decoder_outputs, _, _ = decoder_lstm(dec_emb, initial_state=[state_h, state_c])

# ----- Attention -----
attn_layer = AttentionLayer(name='attention_layer')
attn_out, attn_states = attn_layer([encoder_outputs, decoder_outputs])
decoder_concat_input = Concatenate(axis=-1, name='concat_layer')([decoder_outputs, attn_out])

decoder_softmax_layer = Dense(tar_vocab, activation='softmax')
decoder_softmax_outputs = decoder_softmax_layer(decoder_concat_input)

model = Model([encoder_inputs, decoder_inputs], decoder_softmax_outputs)
model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy')
model.summary()


# %%
# =========================================================
# Cell 8. (루브릭2) 학습 - EarlyStopping
# =========================================================
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=2)

history = model.fit(
    x=[encoder_input_train, decoder_input_train],
    y=decoder_target_train,
    validation_data=([encoder_input_test, decoder_input_test], decoder_target_test),
    batch_size=256, epochs=50, callbacks=[es]
)


# %%
# =========================================================
# Cell 9. (루브릭2) train / validation loss 그래프
# =========================================================
plt.figure(figsize=(7, 4))
plt.plot(history.history['loss'],     label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.title('Training / Validation Loss')
plt.xlabel('epoch'); plt.ylabel('loss'); plt.legend(); plt.show()


# %%
# =========================================================
# Cell 10. 추론용 인코더/디코더 모델 구성
# =========================================================
src_index_to_word = src_tokenizer.index_word
tar_word_to_index = tar_tokenizer.word_index
tar_index_to_word = tar_tokenizer.index_word

# Encoder inference
encoder_model = Model(inputs=encoder_inputs, outputs=[encoder_outputs, state_h, state_c])

# Decoder inference
decoder_state_input_h = Input(shape=(hidden_size,))
decoder_state_input_c = Input(shape=(hidden_size,))
decoder_hidden_state_input = Input(shape=(text_max_len, hidden_size))

dec_emb2 = dec_emb_layer(decoder_inputs)
decoder_outputs2, state_h2, state_c2 = decoder_lstm(
    dec_emb2, initial_state=[decoder_state_input_h, decoder_state_input_c])

attn_out_inf, attn_states_inf = attn_layer([decoder_hidden_state_input, decoder_outputs2])
decoder_inf_concat = Concatenate(axis=-1)([decoder_outputs2, attn_out_inf])
decoder_outputs2 = decoder_softmax_layer(decoder_inf_concat)

decoder_model = Model(
    [decoder_inputs, decoder_hidden_state_input, decoder_state_input_h, decoder_state_input_c],
    [decoder_outputs2, state_h2, state_c2])


def decode_sequence(input_seq):
    e_out, e_h, e_c = encoder_model.predict(input_seq, verbose=0)
    target_seq = np.zeros((1, 1))
    target_seq[0, 0] = tar_word_to_index['sostoken']

    stop = False
    decoded = ''
    while not stop:
        output_tokens, h, c = decoder_model.predict([target_seq, e_out, e_h, e_c], verbose=0)
        idx = np.argmax(output_tokens[0, -1, :])
        word = tar_index_to_word.get(idx, '')
        if word != 'eostoken' and word != '':
            decoded += ' ' + word
        if word == 'eostoken' or len(decoded.split()) >= (summary_max_len - 1):
            stop = True
        target_seq = np.zeros((1, 1)); target_seq[0, 0] = idx
        e_h, e_c = h, c
    return decoded.strip()


def seq2text(seq):
    return ' '.join(src_index_to_word[i] for i in seq if i != 0)

def seq2summary(seq):
    return ' '.join(tar_index_to_word[i] for i in seq
                    if i != 0
                    and i != tar_word_to_index['sostoken']
                    and i != tar_word_to_index['eostoken'])


# %%
# =========================================================
# Cell 11. (루브릭2) Abstractive 결과 확인 - 핵심단어 포함 여부 점검
# =========================================================
for i in range(5):
    print("원문      :", seq2text(encoder_input_test[i]))
    print("실제 요약 :", seq2summary(decoder_padded_test[i]))
    print("예측 요약 :", decode_sequence(encoder_input_test[i].reshape(1, text_max_len)))
    print("-" * 60)


# %%
# =========================================================
# Cell 12. (루브릭1) Extractive 요약 - TextRank (summa)
#  - summa 는 여러 문장으로 된 '긴' 원문에서 잘 동작 (Amazon 리뷰는 짧으므로
#    원본 Text 중 긴 샘플로 시연). 뉴스 데이터셋이면 더 자연스럽게 동작.
# =========================================================
# !pip install summa
from summa.summarizer import summarize

raw = pd.read_csv("Reviews.csv", nrows=100000)[['Text', 'Summary']].dropna()
long_reviews = raw[raw['Text'].apply(lambda x: len(x.split('.')) >= 5)].reset_index(drop=True)

for i in range(3):
    text = long_reviews['Text'][i]
    print("[원문]\n", text[:500], "...\n")
    print("[Extractive(TextRank)]\n", summarize(text, ratio=0.3), "\n")
    print("[Abstractive(seq2seq)]\n",
          decode_sequence(
              pad_sequences(src_tokenizer.texts_to_sequences([preprocess_sentence(text)]),
                            maxlen=text_max_len, padding='post')), "\n")
    print("=" * 70)


# %%
# =========================================================
# Cell 13. (루브릭1) 비교 결과 표로 정리
#  - 아래 표의 '문법 완성도', '핵심단어 포함' 칸은 본인 실제 출력으로 채울 것
# =========================================================
compare = pd.DataFrame({
    '비교 항목': ['문법 완성도', '핵심단어 포함', '요약 길이/압축', '새로운 표현 생성'],
    'Extractive (TextRank)': [
        '원문 문장을 그대로 추출 → 항상 문법적으로 완전',
        '문장 통째로 추출 → 핵심어 누락 거의 없음(불필요 정보도 포함)',
        '문장 단위라 상대적으로 길다',
        '불가능 (원문 문장만 선택)'
    ],
    'Abstractive (seq2seq+Attn)': [
        '단어를 새로 생성 → 짧으면 자연스럽지만 반복/비문 가능',
        '핵심 명사 위주 압축 → 학습 잘되면 포함, 희귀어/OOV는 누락 가능',
        '매우 짧게 압축 (몇 단어)',
        '가능 (원문에 없는 단어 생성)'
    ],
})
pd.set_option('display.max_colwidth', None)
print(compare.to_string(index=False))
