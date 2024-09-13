import os
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#데이터 경로
directory=os.getcwd()+'/data/Label/위드라이브/1be2e43d69994758973f6185bdd973d0'

#파일에 따른 start, end TimeLabel 리스트
start_end_list=[]

#데이터 불러오기 (위도, 경도, 시간레이블)
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

    #결측치 제거
    all_lat_lng_TL_lists.pop(-1)
    start_end_list.pop(-1)

    return all_lat_lng_TL_lists

#경로 데이터 보간
def interpolate_path(path, num_points=10):
    latitudes = [point[0] for point in path]
    longitudes = [point[1] for point in path]
    start_time_labels= [point[2] for point in path]
    end_time_labels= [point[3] for point in path]
    distances = np.linspace(0, 1, len(path))
    interp_lat = interp1d(distances, latitudes, kind='linear')
    interp_lon = interp1d(distances, longitudes, kind='linear')
    interp_start_TL = interp1d(distances, start_time_labels, kind='linear')
    interp_end_TL = interp1d(distances, end_time_labels, kind='linear')
    new_distances = np.linspace(0, 1, num_points)
    new_latitudes = interp_lat(new_distances)
    new_longitudes = interp_lon(new_distances)
    new_start_TL = interp_start_TL(new_distances)
    new_end_TL=interp_end_TL(new_distances)
    return np.column_stack((new_latitudes, new_longitudes, new_start_TL, new_end_TL)).flatten()

# 경로 및 시간(레이블값) 데이터
lat_lng_TL_values = extract_lat_lng_TL_from_csv(directory)

# 경로에 start, end 레이블 추가하기
for location, time in zip(lat_lng_TL_values, start_end_list):
    for point in location:
        point.extend(time)

# 보간된 경로 벡터들
path_vectors = np.array([interpolate_path(path) for path in lat_lng_TL_values])

#DBSCAN 모델링
dbscan=DBSCAN(eps=1.2, min_samples=15)
labels = dbscan.fit_predict(path_vectors)


normal_count=0
abnormal_count=0

# 이상치 여부를 'dbscan output' 열에 기록하는 함수
def record_outliers(directory, labels, filenames):
    for label, filename in zip(labels, filenames):
        filepath = os.path.join(directory, filename)
        global abnormal_count
        global normal_count
        df = pd.read_csv(filepath)
        if 'dbscan output' not in df.columns:
            df['dbscan output'] = 1  # 기본값 1로 설정
        if label == -1:
            df['dbscan output'] = -1  # 이상치일 경우 -1로 설정
            abnormal_count+=1
        else:
            normal_count+=1
            
        df.to_csv(filepath, index=False)

# 이상치 여부를 CSV 파일에 기록
record_outliers(directory, labels, os.listdir(directory))

# 정상 경로/ 이상치 카운트
print(f'정상 개수: {normal_count}, 이상치 개수: {abnormal_count}')