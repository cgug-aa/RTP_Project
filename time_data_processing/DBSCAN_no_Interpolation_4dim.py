import os
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


#데이터 경로
directory=os.getcwd()+'/data/Label/위드라이브/1be2e43d69994758973f6185bdd973d0'

#파일에 따른 start, end TimeLabel 리스트
start_end_list=[]

#데이터 불러오기 (위도, 경도)
def extract_lat_lng_TL_from_csv(directory):
    all_lat_lng_TL_lists = []
    for filename in os.listdir(directory):
        if filename.endswith(".csv"):
            filepath = os.path.join(directory, filename)
            df = pd.read_csv(filepath)
            lat_lng_list=[]
            for index, row in df.iterrows():
                lat_lng_list.append([row['lat'], row['lng']])
                if index==0:
                    s=row['time_label']
                elif index==(df.shape[0]-1):
                    e=row['time_label']
            start_end_list.append([s, e])
            all_lat_lng_TL_lists.append(lat_lng_list)   

    return all_lat_lng_TL_lists

# 경로 및 시간(레이블값) 데이터
lat_lng_TL_values = extract_lat_lng_TL_from_csv(directory)

# 경로에 start, end 레이블 추가하기
for location, time in zip(lat_lng_TL_values, start_end_list):
    for point in location:
        point.extend(time)

# 고정된 크기의 벡터로 변환하기 위한 함수
def vectorize_path(path, vector_size=20):
    # 패딩 또는 트리밍을 통해 길이를 고정합니다.
    if len(path) < vector_size:
        path += [(0, 0, 0, 0)] * (vector_size - len(path))  # 패딩
    else:
        step = len(path) / vector_size
        path = [path[int(i * step)] for i in range(vector_size)]  # 샘플링
    
    # 벡터화하여 평탄화합니다.
    vector = np.array(path).flatten()
    return vector

# 모든 경로 데이터를 벡터화합니다.
vectorized_paths = np.array([vectorize_path(path) for path in lat_lng_TL_values])

'''
#최적 파라미터 탐색
from sklearn.neighbors import NearestNeighbors
import statistics

k = 6

# NearestNeighbors 모델을 사용하여 이웃 계산
neigh = NearestNeighbors(n_neighbors=k)
neigh.fit(vectorized_paths)

# 각 데이터 포인트의 k번째 이웃까지의 거리 계산
distances, indices = neigh.kneighbors(vectorized_paths)

# k번째 이웃까지의 거리만 추출 (k번째 이웃의 거리를 사용)
k_distances = distances[:, -1]

# k-거리를 오름차순으로 정렬
k_distances = np.sort(k_distances)

# k-거리의 차이 계산 (기울기)
gradients = np.diff(k_distances)

# 기울기의 변화량 계산
gradients_diff = np.diff(gradients)

# 급격한 변화가 발생하는 인덱스 찾기 (최초의 큰 변화 찾기)
threshold = np.percentile(gradients_diff, 95)  # 기울기 변화 중 상위 5%를 임계값으로 설정
print('threshold',threshold)
elbow_start = np.argmax(gradients_diff > threshold)  # 첫 번째 급격한 변화 지점
elbow_end = len(gradients_diff) - np.argmax(np.flip(gradients_diff) > threshold) - 1  # 마지막 급격한 변화 지점

# y-좌표 값 출력 (k-거리)
y_values = k_distances[elbow_start:elbow_end + 2]
print(y_values)
eps_opt=statistics.median(y_values) # 중앙값으로 결정
print(f"eps_opt:{eps_opt}")


# k-거리 그래프 시각화
plt.plot(k_distances, label='k-distances')
plt.axvline(x=elbow_start, color='r', linestyle='--', label='Elbow start')
plt.axvline(x=elbow_end, color='g', linestyle='--', label='Elbow end')
plt.xlabel("Data Points sorted by distance")
plt.ylabel(f"Distance to {k}th nearest neighbor")
plt.title(f"k-distance Graph (k={k})")
plt.legend()
plt.show()

from sklearn.metrics import silhouette_score

def evaluate_dbscan(data, eps_values, min_samples_values):
    best_score = -1
    best_eps = None
    best_min_samples = None

    for eps in eps_values:
        for min_samples in min_samples_values:
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            labels = dbscan.fit_predict(data)
            if len(set(labels)) > 1:  # 최소 두 개 이상의 클러스터가 필요
                score = silhouette_score(data, labels, metric='euclidean')
                print(f'eps={eps}, min_samples={min_samples}, silhouette_score={score}')
                if score > best_score:
                    best_score = score
                    best_eps = eps
                    best_min_samples = min_samples

    return best_eps, best_min_samples

# 예시 사용
eps_values = np.arange(1, 2, 0.2)
min_samples_values = range(5, 20, 5)
data = np.concatenate(lat_lng_TL_values)  # 모든 데이터 포인트를 하나로 결합
best_eps, best_min_samples = evaluate_dbscan(data, eps_values, min_samples_values)

print(f'Best eps: {best_eps}, Best min_samples: {best_min_samples}')
'''
# DBSCAN 모델을 생성합니다.
dbscan = DBSCAN(eps=1.6, min_samples=15)
labels = dbscan.fit_predict(vectorized_paths)

# 3D 시각화
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')


for segment, label in zip(lat_lng_TL_values, labels):
        for i in range(0, len(segment), int(len(segment)*0.1)):
            lat=segment[i][0]
            lng=segment[i][1]
            TL=(4*segment[i][2]+segment[i][3])/4

            color = 'red' if label == -1 else 'blue'
            ax.scatter(lat, lng, TL, c=color, marker='o')



ax.set_xlabel('Latitude')
ax.set_ylabel('Longitude')
ax.set_zlabel('Time Label')
plt.title('lat-lng-TL 3D graph')
plt.show()


abnormal_count=0
normal_count=0

for label in labels:
    if label==-1:
        abnormal_count+=1
    else:
        normal_count+=1

# 정상 경로/ 이상치 카운트
print(f'정상 개수: {normal_count}, 이상치 개수: {abnormal_count}')