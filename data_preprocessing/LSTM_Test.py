'''
최초의 LSTM 모델을 형성하여 정확도를 도출한 코드
'''

import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer


#데이터셋 로드
data=pd.read_csv(f'{os.getcwd()}/preprocessing.csv')

# 데이터 전처리
#격자 경로를 숫자 시퀀스로 변환
data['grid_path']=data['grid_path'].apply(lambda x: x.strip("[]").replace("'", "").split(','))

# 격자 경로를 숫자 값으로 변환하기 위한 토크나이저 생성
tokenizer = Tokenizer(char_level=True)  # 문자 단위 토크나이저
tokenizer.fit_on_texts(data['grid_path'])
sequences = tokenizer.texts_to_sequences(data['grid_path'])


# 시퀀스 패딩
max_sequence_length = max(len(seq) for seq in sequences)
X = pad_sequences(sequences, maxlen=max_sequence_length, padding='post')
X_dataframe=pd.DataFrame(X)
X_dataframe.to_csv(f'{os.getcwd()}/test.csv')

# 라벨 인코딩
y = data['F1'].map({'N': 0, 'AN': 1}).values  # 비정상 경로가 'AN'인 경우
y = to_categorical(y)                         # 원-핫 인코딩으로 모델이 이해할 수 있는 숫자 형태로 변환한다.(이진 벡터로 변환)

# 훈련/테스트 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 모델 생성
model = Sequential([
    Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=64, input_length=max_sequence_length),    #임베딩 레이어 
    LSTM(64, return_sequences=False),                                                                       #LSTM 레이어 
    Dense(2, activation='softmax'),                                                                         #Dense 레이어(출력층)
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# 모델 훈련
history = model.fit(X_train, y_train, epochs=15, validation_split=0.2, batch_size=3)

# 모델 평가
loss, accuracy = model.evaluate(X_test, y_test)
print(f'테스트 정확도: {accuracy}')
print('----------------------------------------------')

# 데이터셋에서 일부 입력 텍스트를 가져와서 예측 수행
example_text = data['grid_path'][356]  # 데이터셋에서 첫 번째 샘플을 사용
input_sequence = tokenizer.texts_to_sequences([example_text])
input_padded = pad_sequences(input_sequence, maxlen=max_sequence_length, padding='post')

# 모델을 사용하여 예측 수행
predictions = model.predict(input_padded)

# 예측 결과를 해석
predicted_class = np.argmax(predictions, axis=1)
predicted_probabilities = predictions[0]

#print(example_text)
print(f"입력 시퀀스: {input_sequence}")
print(f"패딩된 시퀀스: {input_padded}")
print(f"예측된 클래스: {predicted_class}")
print(f"클래스 확률: {predicted_probabilities}")
