import numpy as np
import pandas as pd
import folium
import webbrowser
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam

# 예제 데이터를 생성합니다. 실제 데이터 프레임을 여기에 대체하십시오.
data1=pd.read_csv("../RTP/data/vehicle/1be2e43d69994758973f6185bdd973d0/20230503095924.csv")

# 시간은 제외하고 위치 정보만 사용 
location_data1=data1[['lat','lng']].values

# 데이터 전처리
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(location_data1)

# Autoencoder 모델 정의
input_dim = scaled_data.shape[1]
encoding_dim = 2  # 잠재 공간의 크기

input_layer = Input(shape=(input_dim,))
encoder = Dense(encoding_dim, activation='relu')(input_layer)
decoder = Dense(input_dim, activation='sigmoid')(encoder)

autoencoder = Model(inputs=input_layer, outputs=decoder)
autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

# 모델 학습
autoencoder.fit(scaled_data, scaled_data, epochs=50, batch_size=32, shuffle=True, validation_split=0.2)

# 예측 (복원된 데이터)
predicted = autoencoder.predict(scaled_data)

# 복원된 데이터 역변환
restored_data = scaler.inverse_transform(predicted)

# 결과 출력

#restored_df = pd.DataFrame(restored_data, columns=['lat', 'lng', 'time'])
print(restored_data)
map=folium.Map(location=[restored_data[0][0], restored_data[0][1]], zoom_start=12)
for i in range(int(len(location_data1)*0.8)):
    marker=folium.CircleMarker([location_data1[i][0], location_data1[i][1]], radius=1, color='blue', fill_color='blue')
    marker.add_to(map)
for i in range(int(len(location_data1)*0.2)):
    marker=folium.CircleMarker([location_data1[-1-i][0], location_data1[-1-i][1]], radius=1, color='red', fill_color='red')
    marker.add_to(map)
for i in range(len(restored_data)):
    marker=folium.CircleMarker([restored_data[i][0], restored_data[i][1]], radius=1, color='yellow', fill_color='yellow')
    marker.add_to(map)
map_path='map.html'
map.save(map_path)
webbrowser.open(map_path)