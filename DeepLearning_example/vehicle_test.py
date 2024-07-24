# data 확인용 파일

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import folium
import webbrowser


#data 불러오기
data1=pd.read_csv("../RTP/data/vehicle/1be2e43d69994758973f6185bdd973d0/20230503095924.csv")
data2=pd.read_csv('../RTP/data/vehicle/1be2e43d69994758973f6185bdd973d0/20230503111021.csv')
data3=pd.read_csv('../RTP/data/vehicle/1be2e43d69994758973f6185bdd973d0/20230503133135.csv')
data4=pd.read_csv('../RTP/data/vehicle/1be2e43d69994758973f6185bdd973d0/20230724175624.csv')
data5=pd.read_csv('../RTP/data/vehicle/1be2e43d69994758973f6185bdd973d0/20230519164051.csv')
data6=pd.read_csv('../RTP/data/vehicle/1be2e43d69994758973f6185bdd973d0/20230716163658.csv')

location_data1=data1[['lat','lng']]
location_data2=data2[['lat', 'lng']]
location_data3=data3[['lat', 'lng']]
location_data4=data4[['lat', 'lng']]
location_data5=data5[['lat', 'lng']]
location_data6=data6[['lat', 'lng']]

latitude=location_data1['lat'].iloc[0]
longtitude=location_data1['lng'].iloc[0]



map=folium.Map(location=[latitude, longtitude], zoom_start=12)
for i in data1.index:
    marker=folium.CircleMarker([location_data1['lat'][i], location_data1['lng'][i]], radius=1, color='blue', fill_color='blue')
    marker.add_to(map)
for i in data2.index:
    marker=folium.CircleMarker([location_data2['lat'][i], location_data2['lng'][i]], radius=1, color='red', fill_color='red')
    marker.add_to(map)
for i in data3.index:
    marker=folium.CircleMarker([location_data3['lat'][i], location_data3['lng'][i]], radius=1, color='yellow', fill_color='yellow')
    marker.add_to(map)
for i in data4.index:
    marker=folium.CircleMarker([location_data4['lat'][i], location_data4['lng'][i]], radius=1, color='purple', fill_color='purple')
    marker.add_to(map)
for i in data5.index:
    marker=folium.CircleMarker([location_data5['lat'][i], location_data5['lng'][i]], radius=1, color='green', fill_color='green')
    marker.add_to(map)
for i in data6.index:
    marker=folium.CircleMarker([location_data6['lat'][i], location_data6['lng'][i]], radius=1, color='orange', fill_color='orange')
    marker.add_to(map)

map_path='map.html'
map.save(map_path)
webbrowser.open(map_path)