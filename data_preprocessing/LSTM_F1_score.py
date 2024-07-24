'''
그리드 문자열을 LSTM 모델로 학습한 뒤, confusion matrix 및 F1 score 도출
'''
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
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
    Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=32, input_length=max_sequence_length),    #임베딩 레이어  #LSTM 레이어
    LSTM(64, return_sequences=True),
    LSTM(32, return_sequences=False),
    Dense(2, activation='softmax'),                                                                         #Dense 레이어(출력층)
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# 모델 훈련
history = model.fit(X_train, y_train, epochs=15, validation_split=0.2, batch_size=3)

# 모델 평가
loss, accuracy = model.evaluate(X_test, y_test)
print(f'테스트 정확도: {accuracy}')
print('----------------------------------------------')

# Confusion Matrix 계산
y_test_labels = np.argmax(y_test, axis=1)  # 원-핫 인코딩을 정수 라벨로 변환
y_pred_prob = model.predict(X_test)
y_pred_labels = np.argmax(y_pred_prob, axis=1)  # 예측 확률을 정수 라벨로 변환
 
print(f'y_test_labels: {y_test_labels}')
print(f'y_pred_labels: {y_pred_labels}')
conf_matrix = confusion_matrix(y_test_labels, y_pred_labels)
print(f'conf_matrix: {conf_matrix}')

# Confusion Matrix 출력
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['N', 'AN'], yticklabels=['N', 'AN'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

precision = precision_score(y_test_labels, y_pred_labels)
recall = recall_score(y_test_labels, y_pred_labels)
f1 = f1_score(y_test_labels, y_pred_labels)

print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1 Score: {f1}')
print('----------------------------------------------')