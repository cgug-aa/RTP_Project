import os
import pandas as pd
from datetime import datetime
from collections import defaultdict

# 1) 기존 이동 데이터 csv 파일에 start, end, duration 타임스템프의 시간(hour)을 기록하여 칼럼으로 추가한다.

# 1st: 차량 이동 경로 불러오기
def data_processing(data_directory):
    total_start_time=[]
    total_end_time=[]
    for directory_name in os.listdir(data_directory):                                           #17562개 데이터
        dir_path=os.path.join(data_directory, directory_name)
        if (os.path.isdir(dir_path) == True) and (directory_name != 'preprocessing_time'):
            date=[]
            start_time=[]
            end_time=[]
            for file_name in os.listdir(dir_path):
                if file_name.endswith('.csv'):
                    raw_data=pd.read_csv(os.path.join(dir_path, file_name))
                    start, end=to_datetime(raw_data)
                    start_time.append(start)
                    end_time.append(end)
                    total_start_time.extend(start_time)
                    total_end_time.extend(end_time)
                    date.append(file_name[:-4])
            # csv 파일 생성
            make_new_csv(directory_name, date, start_time, end_time)

    # start~end 사이에 포함되는 모든 시간을 카운팅한 csv 파일 생성
    count_time_label(start_time, end_time)

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

    #new_dataframe['(start, end)'] = new_dataframe.apply(lambda row: (row['start'], row['end']), axis=1)
    
    #start와 end 칼럼, (start, end) 칼럼을 추가한 csv파일 생성
    new_file_path=os.getcwd()+'\\data\\위드라이브\\preprocessing_time\\'+usr_name
    new_file=os.getcwd()+'\\data\\위드라이브\\preprocessing_time\\'+usr_name+'\\'+usr_name+'.csv'
    
    if not os.path.exists(new_file_path):  
        os.makedirs(new_file_path)
    new_dataframe.to_csv(new_file)

    '''
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
    #label_count(usr_name, data_dict_dataframe)

    '''

    #new_dataframe에 TL에 따른 매칭 값 칼럼 추가
    new_dataframe=TL_matching_csv(3, new_dataframe)
    new_dataframe=TL_matching_csv(4, new_dataframe)
    TL_matching_path=os.getcwd()+'\\data\\위드라이브\\preprocessing_time\\'+usr_name+'\\TL_matching_'+usr_name+'.csv'
    new_dataframe.to_csv(TL_matching_path)

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
        'label1': range(7, 11),   # 7 ~ 10시
        'label2': range(11, 15),  # 11 ~ 14시
        'label3': range(15, 23),  # 15시 ~ 22시
        'label4': list(range(23, 24)) + list(range(0, 7))  # 23시 ~ 0시 및 0시 ~ 6시
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
        'label1': range(11, 15),   # 11 ~ 14시
        'label2': range(15, 23),  # 15 ~ 23시
        'label3': list(range(23, 24))+ list(range(0, 11))  # 23시 ~ 10시
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
        T1_by3=list(range(23, 24)) + list(range(0, 11))
        T2_by3=range(11, 15)
        T3_by3=range(15, 23)
        start_TL=[]
        end_TL=[]

        for row in new_dataframe.itertuples(index=False):
            start = row.start
            end = row.end
            if start in T1_by3: start_TL.append('T1')
            elif start in T2_by3: start_TL.append('T2')
            elif start in T3_by3: start_TL.append('T3')
            else : raise TypeError
            if end in T1_by3: end_TL.append('T1')
            elif end in T2_by3: end_TL.append('T2')
            elif end in T3_by3 : end_TL.append('T3')
            else: raise TypeError
        new_dataframe['TL_3_start']=start_TL
        new_dataframe['TL_3_end']=end_TL
        return new_dataframe
    else:
        T1_by4=list(range(23, 24)) + list(range(0, 7))
        T2_by4=range(7, 11)
        T3_by4=range(11, 15)
        T4_by4=range(15, 23)
        start_TL=[]
        end_TL=[]
        for row in new_dataframe.itertuples(index=False):
            start = row.start
            end = row.end
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
        new_dataframe['TL_4_start']=start_TL
        new_dataframe['TL_4_end']=end_TL
        return new_dataframe


#start 시간과 end 시간 내 존재하는 모든 시간 카운트하기
def count_time_label(start_time, end_time):
    count_time_dict={   '0' : 0,
                        '1' : 0,
                        '2' : 0,
                        '3' : 0,
                        '4' : 0,
                        '5' : 0,
                        '6' : 0,
                        '7' : 0,
                        '8' : 0,
                        '9' : 0,
                        '10' : 0,
                        '11' : 0,
                        '12' : 0,
                        '13' : 0,
                        '14' : 0,
                        '15' : 0,
                        '16' : 0,
                        '17' : 0,
                        '18' : 0,
                        '19' : 0,
                        '20' : 0,
                        '21' : 0,
                        '22' : 0,
                        '23' : 0    }
    for s, e in zip(start_time, end_time):
        if s <= e:
            time=[x for x in range(s, e+1)]
            for i in time:
                count_time_dict[str(i)]+=1
        else:
            time=[x for x in range(s, 24)]
            time2=[x for x in range(e+1)]
            time.extend(time2)
            for i in time:
                count_time_dict[str(i)]+=1

    #csv 파일 경로
    count_time_dict_csv=os.getcwd()+'\\data\\위드라이브\\preprocessing_time\\count_all_time.csv'

    #csv 파일 생성
    new_dataframe=pd.DataFrame(columns=['time', 'count'])
    rows = [{'time': key, 'count': value} for key, value in count_time_dict.items()]
    new_dataframe=pd.concat([new_dataframe, pd.DataFrame(rows)], ignore_index=True)
    new_dataframe.to_csv(count_time_dict_csv)


# 원본 데이터에서 시간스탬프를 시간시간과 끝시간을 추출하여 파일로 저장
# 테스트 데이터 위드라이브(1be2e43d69994758973f6185bdd973d0)

directory=os.getcwd()+'\\data\\위드라이브'
data_processing(directory)