import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.cluster import DBSCAN
import numpy as np
from geopy.distance import great_circle
import matplotlib.pyplot as plt
import folium
import webbrowser

# 샘플 데이터

# 예제 데이터를 생성합니다. 실제 데이터 프레임을 여기에 대체하십시오.
data1=pd.read_csv("../RTP/data/vehicle/1be2e43d69994758973f6185bdd973d0/20230503095924.csv")

# 시간은 제외하고 위치 정보만 사용 
location_data1=data1[['lat','lng']]

# 데이터를 훈련 세트와 테스트 세트로 분할 (80% 훈련, 20% 테스트)
train_df, test_df = train_test_split(location_data1, test_size=0.2, random_state=42)

# 위도와 경도를 라디안으로 변환
train_df['latitude_rad'] = np.radians(train_df['lat'])
train_df['longitude_rad'] = np.radians(train_df['lng'])
test_df['latitude_rad'] = np.radians(test_df['lat'])
test_df['longitude_rad'] = np.radians(test_df['lng'])

# 훈련 데이터 좌표
train_coords = train_df[['latitude_rad', 'longitude_rad']].to_numpy()

# 테스트 데이터 좌표
test_coords = test_df[['latitude_rad', 'longitude_rad']].to_numpy()

# 파라미터 설정
epsilon = 20 / 6371000  # 20미터를 라디안으로 변환 (지구 반지름 = 6371000 미터)
min_samples = 2

# 훈련 데이터에 DBSCAN 적용
db = DBSCAN(eps=epsilon, min_samples=min_samples, algorithm='ball_tree', metric='haversine').fit(train_coords)

# 훈련 데이터에 클러스터 라벨 추가
train_df['cluster'] = db.labels_

# 각 클러스터의 중심점을 계산
cluster_centers = train_df.groupby('cluster')[['latitude_rad', 'longitude_rad']].mean().reset_index()

def find_nearest_cluster(test_point, cluster_centers):
    distances = cluster_centers.apply(
        lambda center: great_circle((test_point[0], test_point[1]), (center['latitude_rad'], center['longitude_rad'])).meters,
        axis=1
    )
    return cluster_centers.loc[distances.idxmin()]['cluster']

# 테스트 데이터에 클러스터 할당
test_df['predicted_cluster'] = [find_nearest_cluster(point, cluster_centers) for point in test_coords]
print(test_df)

# 예측 결과 시각화 (테스트 데이터에 대한 예측)
map=folium.Map(location=[train_df['lat'][0], train_df['lng'][0]], zoom_start=12)
for i in range(len(train_df)):
    marker=folium.CircleMarker([train_df['lat'].iloc[i], train_df['lng'].iloc[i]], radius=1, color='blue', fill_color='blue')
    marker.add_to(map)
for i in range(len(test_df)):
    marker=folium.CircleMarker([test_df['lat'].iloc[-1-i], test_df['lng'].iloc[-1-i]], radius=1, color='red', fill_color='red')
    marker.add_to(map)
for i in range(len(test_df)):
    if test_df['predicted_cluster'].iloc[i]>2:
        marker=folium.CircleMarker([test_df['lat'].iloc[i], test_df['lng'].iloc[i]], radius=1, color='yellow', fill_color='yellow')
        marker.add_to(map)
map_path='map.html'
map.save(map_path)
webbrowser.open(map_path)
