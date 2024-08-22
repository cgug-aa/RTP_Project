import os
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#데이터 경로
directory=os.getcwd()+'/data/Label/위드라이브/1be2e43d69994758973f6185bdd973d0'

#데이터 불러오기 (위도, 경도, 시간레이블)
def extract_lat_lng_TL_from_csv(directory):
    all_lat_lng_TL_lists = []
    for filename in os.listdir(directory):
        if filename.endswith(".csv"):
            filepath = os.path.join(directory, filename)
            df = pd.read_csv(filepath)
            lat_lng_list = [[row['lat'], row['lng'], row['time_label']] for _, row in df.iterrows()]
            all_lat_lng_TL_lists.append(lat_lng_list)

    #결측치 제거
    all_lat_lng_TL_lists.pop(-1)
    print(all_lat_lng_TL_lists)
    return all_lat_lng_TL_lists

#경로 데이터 보간
def interpolate_path(path, num_points=10):
    latitudes = [point[0] for point in path]
    longitudes = [point[1] for point in path]
    time_labels= [point[2] for point in path]
    distances = np.linspace(0, 1, len(path))
    interp_lat = interp1d(distances, latitudes, kind='linear')
    interp_lon = interp1d(distances, longitudes, kind='linear')
    interp_TL = interp1d(distances, time_labels, kind='linear')
    new_distances = np.linspace(0, 1, num_points)
    new_latitudes = interp_lat(new_distances)
    new_longitudes = interp_lon(new_distances)
    new_TL = interp_TL(new_distances)
    return np.column_stack((new_latitudes, new_longitudes, new_TL)).flatten()

# 경로 및 시간(레이블값) 데이터
lat_lng_TL_values = extract_lat_lng_TL_from_csv(directory)

# 보간된 경로 벡터들
path_vectors = np.array([interpolate_path(path) for path in lat_lng_TL_values])

#DBSCAN 모델링
dbscan=DBSCAN(eps=0.5, min_samples=5)
labels = dbscan.fit_predict(path_vectors)

# 시각화
fig = plt.figure( figsize=(6,6))
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
ax.scatter(r['Sepal length'],r['Sepal width'],r['Petal length'],c=r['predict'],alpha=0.5)
ax.set_xlabel('Sepal lenth')
ax.set_ylabel('Sepal width')
ax.set_zlabel('Petal length')
plt.show()

# 경로 벡터에서 각 축에 해당하는 데이터 추출
latitudes = path_vectors[:, 0]  # 위도 (Latitude)
longitudes = path_vectors[:, 1]  # 경도 (Longitude)
time_labels = path_vectors[:, 2]  # 타임레이블 (Time Labels)

# 3D 플롯 생성
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')  # Axes3D 객체 생성

# 경로 시각화
ax.plot(latitudes, longitudes, time_labels, label='Interpolated Path', marker='o')

# 축 레이블 설정
ax.set_xlabel('Latitude')
ax.set_ylabel('Longitude')
ax.set_zlabel('Time Label')

# 그래프 제목 설정
ax.set_title('3D Visualization of Interpolated Path')

# 범례 추가
ax.legend()

# 그래프 표시
plt.show()
