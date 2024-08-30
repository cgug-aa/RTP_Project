from sklearn.metrics import silhouette_score
import pandas as pd
from sklearn.cluster import DBSCAN
import os
from scipy.interpolate import interp1d
import numpy as np

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
eps_values = np.arange(1, 2, 0.5)
min_samples_values = range(20, 50, 10)
data = np.concatenate(lat_lng_TL_values)  # 모든 데이터 포인트를 하나로 결합
best_eps, best_min_samples = evaluate_dbscan(data, eps_values, min_samples_values)

print(f'Best eps: {best_eps}, Best min_samples: {best_min_samples}')
