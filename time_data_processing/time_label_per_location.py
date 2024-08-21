import pandas as pd
import os
from datetime import datetime


#데이터 경로 
directory=os.getcwd()+'\\data\\위드라이브\\1be2e43d69994758973f6185bdd973d0'
labeled_directory=os.getcwd()+'\\data\\Label\\위드라이브\\1be2e43d69994758973f6185bdd973d0'


# 타임스탬프 변환
def to_datetime(data_csv):
    timestamp_list=data_csv['time'].tolist()
    transfer_time=[]
    for time in timestamp_list:
        transfer_time.append(int(str(datetime.fromtimestamp(time//1000))[11:13]))         #datetime에서 시간에 해당하는 값만 추출해 int형으로 변환
    
    time_label_list=time_labeling(transfer_time)
    
    new_data=data_csv.copy()
    new_data['time_label']=time_label_list

    return new_data

    
#데이터의 시간을 레이블링한다. (T1=0, T2=1, T3=2, T4=3)
def time_labeling(input_list):
    time_label=[]
    label1= list(range(7, 11))                                    # 7 ~ 10시                 0
    label2= list(range(11, 15))                                   # 11 ~ 14시                1
    label3= list(range(15, 23))                                   # 15시 ~ 22시              2
    label4= list(range(23, 24))+list(range(0,7))                  # 23시 ~ 0시 및 0시 ~ 6시   3

    for time in input_list:
        if time in label1: time_label.append(0)
        elif time in label2: time_label.append(1)
        elif time in label3: time_label.append(2)
        elif time in label4: time_label.append(3)
    
    return time_label


#데이터 전처리 (원본 데이터에 추가 칼럼(start_TL, end_TL) 추가)
for file_name in os.listdir(directory):
    if file_name.endswith('.csv'):
        file_path=os.path.join(directory, file_name)
        raw_data=pd.read_csv(file_path)
        labeled_data=to_datetime(raw_data)
        labeled_dataFrame=pd.DataFrame(labeled_data)

        labeled_data_file=os.path.join(labeled_directory, file_name)
        labeled_dataFrame.to_csv(labeled_data_file)