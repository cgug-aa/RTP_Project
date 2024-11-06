import os
import pandas as pd
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import dask.dataframe as dd

#데이터 경로
directory=os.getcwd()+'/data/Label/위드라이브/1be2e43d69994758973f6185bdd973d0'
#(어디쉐어)
#directory=os.getcwd()+'\\data\\어디쉐어전처리데이터\\___ecfd1086a6934ae08b555b3ae880d31e'

'''
# 병합할 빈 DataFrame 생성
merged_df = pd.DataFrame()

# 폴더 내 모든 CSV 파일에 대해 반복
for filename in os.listdir(directory):
    if filename.endswith('.csv'):
        file_path = os.path.join(directory, filename)
        
        # CSV 파일을 DataFrame으로 읽기
        df = pd.read_csv(file_path)
        
        # 필요한 열만 선택 (lat, lng, output)
        df = df[['lat', 'lng', 'time_label']]
        
        # 병합된 DataFrame에 추가
        merged_df = pd.concat([merged_df, df], ignore_index=True)

# 결과를 하나의 CSV 파일로 저장
merged_df.to_csv('./combined_위드라이브.csv', index=False)

print("CSV 파일이 성공적으로 병합되었습니다.")
'''
# CSV 파일 읽기 (파일 경로를 적절히 변경하세요) Dask를 사용해 대용량 CSV 파일을 처리
file_path = "./combined_위드라이브.csv"

# 데이터프레임 읽기
df = pd.read_csv(file_path)

#전체 데이터 경로. 필요한 열만 선택
all_lat_lng_TL_lists = [[row['lat'], row['lng'], row['time_label']] for _, row in df.iterrows()]

#최적 파라미터 탐색
from sklearn.neighbors import NearestNeighbors
import statistics
import numpy as np

k = 4

# NearestNeighbors 모델을 사용하여 이웃 계산
neigh = NearestNeighbors(n_neighbors=k)
neigh.fit(all_lat_lng_TL_lists)

# 각 데이터 포인트의 k번째 이웃까지의 거리 계산
distances, indices = neigh.kneighbors(all_lat_lng_TL_lists)

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


# DBSCAN 모델 생성
dbscan = DBSCAN(eps=0.001, min_samples=15)
labels = dbscan.fit_predict(all_lat_lng_TL_lists)

# 결과 시각화
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 정상 데이터는 파란색, 이상치는 빨간색으로 표시
for i in range(0,len(labels),500):
        lat=all_lat_lng_TL_lists[i][0]
        lng=all_lat_lng_TL_lists[i][1]
        TL=all_lat_lng_TL_lists[i][2]
        color = 'red' if labels[i] == -1 else 'blue'
        ax.scatter(lat, lng, TL, c=color, marker='o')


ax.set_xlabel('Latitude')
ax.set_ylabel('Longitude')
ax.set_zlabel('Time Label')
plt.title('DBSCAN Clustering of Geographical Data')
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