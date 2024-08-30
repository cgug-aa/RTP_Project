import os
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import normalize


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

# 경로 및 시간(레이블값) 데이터
lat_lng_TL_values = extract_lat_lng_TL_from_csv(directory)

# 고정된 크기의 벡터로 변환하기 위한 함수
def vectorize_path(path, vector_size=20):
    # 패딩 또는 트리밍을 통해 길이를 고정합니다.
    if len(path) < vector_size:
        path += [(0, 0)] * (vector_size - len(path))  # 패딩
    else:
        step = len(path) / vector_size
        path = [path[int(i * step)] for i in range(vector_size)]  # 샘플링
    
    # 벡터화하여 평탄화합니다.
    vector = np.array(path).flatten()
    return vector

# 모든 경로 데이터를 벡터화합니다.
vectorized_paths = np.array([vectorize_path(path) for path in lat_lng_TL_values])
# 정규화(옵션, 선택사항)
vectorized_paths = normalize(vectorized_paths)

# DBSCAN 모델을 생성합니다.
dbscan = DBSCAN(eps=0.1, min_samples=45)
labels = dbscan.fit_predict(vectorized_paths)

'''
# 3D 시각화
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for segment, label, vec in zip(lat_lng_TL_values, labels, vectorized_paths):
    for i in range(60):
        if vec[i]!=0:
            lat=segment[i][0]
            lng=segment[i][1]
            TL=segment[i][2]
            color = 'red' if label == -1 else 'blue'
            ax.scatter(lat, lng, TL, c=color, marker='o')
            count+=1
    if label==-1:
        abnormal_count+=1
    else:
        normal_count+=1

ax.set_xlabel('Latitude')
ax.set_ylabel('Longitude')
ax.set_zlabel('Time Label')
plt.title('lat-lng-TL 3D graph')
plt.show()
'''

# 정상/이상치 카운트
normal_count=0
abnormal_count=0


#DBSCAN 2차원 시각화
plt.figure(figsize=(10, 6))

for i, lat_lng_list in enumerate(lat_lng_TL_values):
    lats, lngs = zip(*lat_lng_list)
    if labels[i]==-1:
        col = 'r'
    else:
        col = 'b'
    plt.plot(lngs, lats, marker='o',color = col, alpha=0.7)
    

plt.title('Interpolated Path Vectors')
plt.xlabel('Latitude')
plt.ylabel('Longitude')
plt.grid(True)
plt.legend()
plt.show()

# 정상 경로/ 이상치 카운트
print(f'정상 개수: {normal_count}, 이상치 개수: {abnormal_count}')