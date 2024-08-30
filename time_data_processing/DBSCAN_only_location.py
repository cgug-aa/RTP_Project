import os
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

#데이터 경로
directory=os.getcwd()+'/data/Label/위드라이브/1be2e43d69994758973f6185bdd973d0'

#데이터 불러오기 (위도, 경도, 시간레이블)
def extract_lat_lng_TL_from_csv(directory):
    all_lat_lng_TL_lists = []
    for filename in os.listdir(directory):
        if filename.endswith(".csv"):
            filepath = os.path.join(directory, filename)
            df = pd.read_csv(filepath)
            lat_lng_list = [[row['lat'], row['lng']] for _, row in df.iterrows()]
            all_lat_lng_TL_lists.append(lat_lng_list)

    #결측치 제거
    all_lat_lng_TL_lists.pop(-1)

    return all_lat_lng_TL_lists

#경로 데이터 보간
def interpolate_path(path, num_points=10):
    latitudes = [point[0] for point in path]
    longitudes = [point[1] for point in path]
    distances = np.linspace(0, 1, len(path))
    interp_lat = interp1d(distances, latitudes, kind='linear')
    interp_lon = interp1d(distances, longitudes, kind='linear')
    new_distances = np.linspace(0, 1, num_points)
    new_latitudes = interp_lat(new_distances)
    new_longitudes = interp_lon(new_distances)
    return np.column_stack((new_latitudes, new_longitudes)).flatten()

# 경로 및 시간(레이블값) 데이터
lat_lng_values = extract_lat_lng_TL_from_csv(directory)

# 보간된 경로 벡터들
path_vectors = np.array([interpolate_path(path) for path in lat_lng_values])

#DBSCAN 모델링
dbscan=DBSCAN(eps=1.5, min_samples=15)
labels = dbscan.fit_predict(path_vectors)

plt.figure(figsize=(10, 6))

normal_count=0
abnormal_count=0

for segment, label in zip(path_vectors, labels):
    lat=segment[0::2]
    lng=segment[1::2]
    if label == -1:
        color = 'red' 
        abnormal_count+=1
    else:
        color = 'blue'
        normal_count+=1
    plt.plot(lng, lat, marker='o', c = color, alpha=0.7)
    

plt.title('Interpolated Path Vectors')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.grid(True)
plt.legend()
plt.show()

# 정상 경로/ 이상치 카운트
print(f'정상 개수: {normal_count}, 이상치 개수: {abnormal_count}')