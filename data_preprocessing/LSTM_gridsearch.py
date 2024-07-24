'''
gridsearch CV를 통해 최적의 파라미터 도출하기 위한 코드
'''

import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, f1_score

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


# 라벨 인코딩
y = data['F1'].map({'N': 0, 'AN': 1}).values  # 비정상 경로가 'AN'인 경우
#y = to_categorical(y)                         # 원-핫 인코딩으로 모델이 이해할 수 있는 숫자 형태로 변환한다.(이진 벡터로 변환)



# 훈련/테스트 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 모델 생성 함수
def create_model(embedding_dim=64, lstm_units=128, optimizer='adam'):
    model = Sequential([
        Embedding(input_dim=len(np.unique(X)) + 1, output_dim=embedding_dim, input_length=X.shape[1]),    
        LSTM(lstm_units, return_sequences=False),
        Dense(1, activation='sigmoid'),
    ])
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

# KerasClassifier 래퍼
model = KerasClassifier(build_fn=create_model, verbose=0)

# 하이퍼파라미터 그리드 설정
param_grid = {
    'embedding_dim': [128],
    'lstm_units': [64],
    'optimizer': ['adam'],
    'epochs': [20],
    'batch_size': [2,3]
}
# F1 스코어를 위한 스코어러 설정
f1_scorer = make_scorer(f1_score, average='binary')

# GridSearchCV 설정
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3, verbose=2, scoring=f1_scorer)

# 모델 훈련
grid_result = grid.fit(X_train, y_train)

# 결과 출력
print(f"최적 F1 스코어: {grid_result.best_score_}")
print(f"최적 파라미터: {grid_result.best_params_}")

# 테스트 데이터로 평가
best_model = grid_result.best_estimator_
y_pred = best_model.predict(X_test)
f1 = f1_score(y_test, y_pred, average='binary')
print(f"테스트 데이터 F1 스코어: {f1}")