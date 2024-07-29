import os
import pandas as pd
from datetime import datetime
from collections import defaultdict

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
    new_dataframe['(start, end)'] = new_dataframe.apply(lambda row: (row['start'], row['end']), axis=1)
    
    #start와 end 칼럼, (start, end) 칼럼을 추가한 csv파일 생성
    new_file_path=os.getcwd()+'\\data\\위드라이브\\preprocessing_time\\'+usr_name
    new_file=os.getcwd()+'\\data\\위드라이브\\preprocessing_time\\'+usr_name+'\\'+usr_name+'.csv'
    
    if not os.path.exists(new_file_path):  
        os.makedirs(new_file_path)
    new_dataframe.to_csv(new_file)
    
    #start와 end의 튜플쌍을 카운팅한 csv파일 생성
    count_file=os.getcwd()+'\\data\\위드라이브\\preprocessing_time\\'+usr_name+'\\count_'+usr_name+'.csv'
    data_dict=count_tuple(new_dataframe)
    data_dict_dataframe=pd.DataFrame(columns=['(start, end)', 'count'])
    rows = [{'(start, end)': key, 'count': value} for key, value in data_dict.items()]
    data_dict_dataframe=pd.concat([data_dict_dataframe, pd.DataFrame(rows)], ignore_index=True)
    data_dict_dataframe.to_csv(count_file)


    #count_id.csv파일에 start와 end시간의 TL매칭하기
    # 3단계 시간 분할 매칭 파일생성   
    TL_matching_DataFrame_by3=TL_matching_csv(3, new_dataframe)
    TL_matching_by3_path=os.getcwd()+'\\data\\위드라이브\\preprocessing_time\\'+usr_name+'\\TL_matching_by3_'+usr_name+'.csv'
    TL_matching_DataFrame_by3.to_csv(TL_matching_by3_path)

    # 4단계 시간 분할 매칭 파일생성
    TL_matching_DataFrame_by4=TL_matching_csv(4, new_dataframe)
    TL_matching_by4_path=os.getcwd()+'\\data\\위드라이브\\preprocessing_time\\'+usr_name+'\\TL_matching_by4_'+usr_name+'.csv'
    TL_matching_DataFrame_by4.to_csv(TL_matching_by4_path)

    #TL4, 3으로 분할한 csv 파일 만들기
    label_count(usr_name, data_dict_dataframe)


    

# timelabel의 값을 기준으로 count하는 함수
def count_tuple(dataframe):
    TL_dict = defaultdict(int)  # defaultdict를 사용하여 초기값을 0으로 설정
    for value in dataframe['(start, end)']:
        TL_dict[value] += 1  # 'TL' 값의 빈도를 증가시킴
    return dict(TL_dict)  # defaultdict를 일반 딕셔너리로 변환하여 반환

#TL을 네단계로 나눈 값의 개수와 세단계로 나눈 값의 개수를 카운트해 csv파일로 생성하는 함수
def label_count(id, csv_dataframe):

    # 4분할 시간 범위 정의
    time_ranges_by4 = {
        'label1': range(5, 11),   # 5 ~ 10시
        'label2': range(11, 16),  # 11 ~ 15시
        'label3': range(16, 20),  # 16시 ~ 19시
        'label4': list(range(20, 24)) + list(range(0, 5))  # 20시 ~ 24시 및 0시 ~ 4시
    }
    
    # 4분할 TL 카운트
    count_dict_by4 = defaultdict(int)
    for value, count in zip(csv_dataframe['(start, end)'], csv_dataframe['count']):
        value_count=count
        start_time, end_time = value
        for label, hours in time_ranges_by4.items():
            if start_time in hours and end_time in hours:
                count_dict_by4[label] += value_count
                break

    #4분할 카운트 csv 파일 생성
    label_count_by4_csv=os.getcwd()+'\\data\\위드라이브\\preprocessing_time\\'+id+'\\count_by4_'+id+'.csv'
    label4_frame=pd.DataFrame(columns=['TL', 'count'])
    label4_rows = [{'TL': key, 'count': value} for key, value in count_dict_by4.items()]
    label4_dataframe=pd.concat([label4_frame, pd.DataFrame(label4_rows)], ignore_index=True)
    label4_dataframe.to_csv(label_count_by4_csv)

    #---------------------------------------------------------------------
    #3분할 시간 범위 정의
    time_ranges_by3 = {
        'label1': range(6, 11),   # 6 ~ 11시
        'label2': range(12, 18),  # 12 ~ 18시
        'label3': list(range(19, 24))+ list(range(0, 5))  # 19시 ~ 5시
    }

    #3분할 시간 카운트
    count_dict_by3 = defaultdict(int)
    for value, count in zip(csv_dataframe['(start, end)'], csv_dataframe['count']):
        value_count=count
        start_time, end_time = value
        for label, hours in time_ranges_by3.items():
            if start_time in hours and end_time in hours:
                count_dict_by3[label] += value_count
                break

    #3분할 카운트 csv 파일 생성
    label_count_by3_csv=os.getcwd()+'\\data\\위드라이브\\preprocessing_time\\'+id+'\\count_by3_'+id+'.csv'
    label3_frame=pd.DataFrame(columns=['TL', 'count'])
    label3_rows = [{'TL': key, 'count': value} for key, value in count_dict_by3.items()]
    label3_dataframe=pd.concat([label3_frame, pd.DataFrame(label3_rows)], ignore_index=True)
    label3_dataframe.to_csv(label_count_by3_csv)

#count_id.csv파일에 start와 end시간의 TL매칭하기 
def TL_matching_csv(label_num, dataframe):
    new_dataframe=dataframe.copy()
    if label_num==3:
        T1_by3=range(0, 7)
        T2_by3=range(7, 17)
        T3_by3=range(17, 24)

        start_TL=[]
        end_TL=[]
        for start, end in dataframe['(start, end)']:
            if start in T1_by3: start_TL.append('T1')
            elif start in T2_by3: start_TL.append('T2')
            elif start in T3_by3: start_TL.append('T3')
            else : raise TypeError
            if end in T1_by3: end_TL.append('T1')
            elif end in T2_by3: end_TL.append('T2')
            elif end in T3_by3 : end_TL.append('T3')
            else: raise TypeError
        new_dataframe['TL_start_by3']=start_TL
        new_dataframe['TL_end_by3']=end_TL
        return new_dataframe
    else:
        T1_by4=range(0, 7)
        T2_by4=range(7, 13)
        T3_by4=range(13, 17)
        T4_by4=range(17, 24)
        start_TL=[]
        end_TL=[]
        for start, end in dataframe['(start, end)']:
            if start in T1_by4: start_TL.append('T1')
            elif start in T2_by4: start_TL.append('T2')
            elif start in T3_by4: start_TL.append('T3')
            elif start in T4_by4: start_TL.append('T4')
            else : raise TypeError
            if end in T1_by4: end_TL.append('T1')
            elif end in T2_by4: end_TL.append('T2')
            elif end in T3_by4 : end_TL.append('T3')
            elif end in T4_by4 : end_TL.append('T4')
            else: raise TypeError
        new_dataframe['TL_start_by4']=start_TL
        new_dataframe['TL_end_by4']=end_TL
        return new_dataframe

# 원본 데이터에서 시간스탬프를 시간시간과 끝시간을 추출하여 파일로 저장
# 테스트 데이터 위드라이브(1be2e43d69994758973f6185bdd973d0)
directory=os.getcwd()+'\\data\\위드라이브\\1be2e43d69994758973f6185bdd973d0'
data_processing(directory)