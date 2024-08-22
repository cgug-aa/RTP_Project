import pandas as pd
import os
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler

#데이터 경로
directory=os.getcwd()+'/data/Label/위드라이브/1be2e43d69994758973f6185bdd973d0'

# 한 사람의 모든 데이터 통합
# 각 폴더 내의 CSV 파일들을 하나로 합쳐서 저장
dataframes = []
count=0
for folder in os.listdir(directory):
    # 데이터프레임 리스트 초기화
    csv_files=os.path.join(directory, folder)
    # 각 파일을 읽어서 데이터프레임으로 변환하고 리스트에 추가
    df = pd.read_csv(csv_files)
    count+=len(df)
    dataframes.append(df)
    
# 모든 데이터프레임을 하나로 합치기
combined_df = pd.concat(dataframes, ignore_index=True)

# 합쳐진 데이터프레임 저장
combined_csv_path = os.path.join(os.getcwd(), 'combined.csv')
combined_df.to_csv(combined_csv_path, index=False)

# 데이터 불러오기 및 결측값 제거
data = pd.read_csv(os.getcwd()+'/combined.csv')
data = data.dropna(subset=['lat', 'lng', 'time_label'])

# lat, lng, time_label 열만 사용
X = data[['lat', 'lng', 'time_label']]

# train-test split
X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)

# Isolation Forest 모델 학습
iso_forest = IsolationForest(contamination=0.2, random_state=42)
iso_forest.fit(X_train)

# 예측
predictions = iso_forest.predict(X_test)

# 예측 결과를 데이터프레임에 추가
X_test['anomaly'] = predictions

# 3D 시각화
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')

# 정상 데이터 포인트
ax.scatter(X_test[X_test['anomaly'] == 1]['lat'], 
           X_test[X_test['anomaly'] == 1]['lng'], 
           X_test[X_test['anomaly'] == 1]['time_label'], 
           color='blue', s=5, label='Normal')

# 이상 데이터 포인트
ax.scatter(X_test[X_test['anomaly'] == -1]['lat'], 
           X_test[X_test['anomaly'] == -1]['lng'], 
           X_test[X_test['anomaly'] == -1]['time_label'], 
           color='red', s=5, label='Anomaly')

# 그래프 제목 및 축 라벨
ax.set_title('Isolation Forest: Anomaly Detection')
ax.set_xlabel('Latitude')
ax.set_ylabel('Longitude')
ax.set_zlabel('Time_Label')

# 범례 추가
ax.legend()

# 그래프 보여주기
plt.show()