import os
import pandas as pd
from datetime import datetime

# 1) 기존 이동 데이터 csv 파일에 start, end, duration 타임스템프의 시간(hour)을 기록하여 칼럼으로 추가한다.

# 1st: 차량 이동 경로 불러오기
def data_processing(data_directory):
    data_path=data_directory
    usr_name=data_path.split('\\')[-1]
    date=[]
    start_time=[]
    end_time=[]
    for file_name in os.listdir(data_path):
        if file_name.endswith('.csv'):
            raw_data=pd.read_csv(os.path.join(data_path, file_name))
            start, end=to_datetime(raw_data)
            start_time.append(start)
            end_time.append(end)
            date.append(file_name[:-4])
    make_new_csv(usr_name, date, start_time, end_time)


# 2nd: 시간스탬프 변환하기
def to_datetime(data_csv):
    timestamp_list=data_csv['time'].tolist()
    datetime_start=int(str(datetime.fromtimestamp(timestamp_list[0]//1000))[11:13])         #datetime에서 시간에 해당하는 값만 추출해 int형으로 변환
    datetime_end=int(str(datetime.fromtimestamp(timestamp_list[-1]//1000))[11:13])
    return (datetime_start, datetime_end)
    
    
# 3rd: id의 이동 시작 시간 및 종료 시간을 기록한 csv 파일 생성
def make_new_csv(usr_name, list1, list2, list3):
    dataframe=zip(list1, list2, list3)
    new_dataframe=pd.DataFrame(dataframe, columns=['date', 'start', 'end'])
    new_file_Directory=os.getcwd()+'\\data\\위드라이브\\preprocessing_time\\'+usr_name
    print(new_file_Directory)
    new_dataframe.to_csv(new_file_Directory)
        


    
    #만들어지는 파일 다시 확인하기




#테스트 데이터 위드라이브(1be2e43d69994758973f6185bdd973d0)

directory=os.getcwd()+'\\data\\위드라이브\\1be2e43d69994758973f6185bdd973d0'
data_processing(directory)