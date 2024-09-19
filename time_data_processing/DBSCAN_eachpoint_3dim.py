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
merged_df.to_csv('./combined.csv', index=False)

print("CSV 파일이 성공적으로 병합되었습니다.")
'''

# CSV 파일 읽기 (파일 경로를 적절히 변경하세요) Dask를 사용해 대용량 CSV 파일을 처리
file_path = "./combined.csv"

# Dask로 데이터프레임 읽기
ddf = dd.read_csv(file_path)

# 필요한 열만 선택
coordinates = ddf[['lat', 'lng', 'time_label']]

# Dask DataFrame을 Pandas DataFrame으로 변환 (메모리 문제를 피하기 위해 샘플링 가능)
coordinates_sample = coordinates.sample(frac=0.2).compute()

# DBSCAN 클러스터링 수행
db = DBSCAN(eps=0.005, min_samples=20).fit(coordinates_sample)

# 클러스터 라벨 추가
labels = db.labels_

# 이상치 (라벨 -1)
outliers = coordinates_sample[labels == -1]
core_samples = coordinates_sample[labels != -1]

# 결과 시각화
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 정상 데이터는 파란색, 이상치는 빨간색으로 표시
ax.scatter(core_samples['lat'], core_samples['lng'], core_samples['time_label'], c='blue', marker='o', label='Core Points')
ax.scatter(outliers['lat'], outliers['lng'], outliers['time_label'], c='red', marker='o', label='Outliers')

ax.set_xlabel('Latitude')
ax.set_ylabel('Longitude')
ax.set_zlabel('Time Label')
plt.title('DBSCAN Clustering of Geographical Data')
plt.legend()
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