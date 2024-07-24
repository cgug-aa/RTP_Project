import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt
import folium
import webbrowser
import os

# 레이블 데이터 
label_files_directory = os.getcwd()+'/Label'
csv_files=[]

for filename in os.listdir(label_files_directory):
    if filename.endswith('.csv'):
        file_path = os.path.join(label_files_directory, filename)
        # CSV 파일을 읽어옵니다.
        df = pd.read_csv(file_path)
        # 읽어온 데이터프레임을 리스트에 추가합니다.
        csv_files.append(df)

combined_df = pd.concat(csv_files, ignore_index=True)
print(combined_df)

# 시간은 제외하고 그리드 레이블 정보만 사용 
combined_df_label=combined_df['grid_label']
'''
# 시퀀스 생성 함수
def create_sequences(data, seq_length):
    sequences = []
    targets = []
    for i in range(len(data) - seq_length):
        sequences.append(data[i:i+seq_length])
        targets.append(data[i+seq_length])
    return np.array(sequences), np.array(targets)

# 시퀀스 길이 설정
seq_length = 15

# 시퀀스와 타겟 데이터 생성
seq_1, tgt_1 = create_sequences(location_data1, seq_length)

# 시퀀스 형태 확인
print(f"X shape: {seq_1.shape}")
print(f"y shape: {tgt_1.shape}")

# LSTM에 입력할 수 있도록 데이터 형태 변경 (samples, time steps, features)
input_1 = seq_1.reshape((seq_1.shape[0], seq_1.shape[1], 2))
print(f"Reshaped X shape: {input_1.shape}")

# 데이터 분할 (학습용과 테스트용)
split_ratio = 0.8
split_index = int(len(seq_1) * split_ratio)

seq_1_train, seq_1_test = seq_1[:split_index], seq_1[split_index:]
tgt_1_train, tgt_1_test = tgt_1[:split_index], tgt_1[split_index:]

# 모델 생성
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(seq_length, 2)))
model.add(Dense(2))  # 예측할 위치 정보의 feature 수 (위도와 경도)

# 모델 컴파일
model.compile(optimizer='adam', loss='mse')

# 모델 요약
model.summary()

# 모델 학습
history = model.fit(seq_1_train, tgt_1_train, epochs=20, validation_split=0.2, batch_size=32)

# 모델 평가
loss = model.evaluate(seq_1_test, tgt_1_test)
print(f"Test Loss: {loss}")

# 테스트 데이터 예측
predictions = model.predict(seq_1_test)
print(predictions)

# 예측 결과 시각화 (테스트 데이터에 대한 예측)
map=folium.Map(location=[seq_1[0][0][0], seq_1[0][0][1]], zoom_start=12)
for i in range(len(seq_1_train)):
    marker=folium.CircleMarker([seq_1_train[i][0][0], seq_1_train[i][0][1]], radius=1, color='blue', fill_color='blue')
    marker.add_to(map)
for i in range(len(seq_1_test)):
    marker=folium.CircleMarker([tgt_1_test[i][0], tgt_1_test[i][1]], radius=1, color='red', fill_color='red')
    marker.add_to(map)
for i in range(len(tgt_1_test)):
    marker=folium.CircleMarker([predictions[i][0], predictions[i][1]], radius=1, color='yellow', fill_color='yellow')
    marker.add_to(map)
map_path='map.html'
map.save(map_path)
webbrowser.open(map_path)
'''