# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 22:26:13 2023

@author: Asistan
"""
from math import radians, sin, cos, sqrt, atan2
import os
import scipy.io as sio
import numpy as np
from scipy.stats import pearsonr

def haversine(lat1, lon1, lat2, lon2):
    # Dünya yarıçapı (km)
    R = 6371.0

    # Radyan cinsinden enlem ve boylam değerleri
    lat1 = radians(lat1)
    lon1 = radians(lon1)
    lat2 = radians(lat2)
    lon2 = radians(lon2)

    # Farkları hesaplama
    dlon = lon2 - lon1
    dlat = lat2 - lat1

    # Haversine formülü
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    # Mesafeyi hesaplama
    distance = R * c
    return distance
distance = []
PS = []
fname = r"D:\Baris\Interface\matfiles\saved"
for b in os.listdir(fname)[0:len(os.listdir(fname))]: 
    dictname = 'afad'
    dataset=sio.loadmat('{}/{}'.format(fname, b))
    
    stationLat = dataset['anEQ']['statco'][0][0][0][0]
    stationLon = dataset['anEQ']['statco'][0][0][0][1]
    
    epicenterLat = dataset['anEQ']['epicenter'][0][0][0][0]
    epicenterLon = dataset['anEQ']['epicenter'][0][0][0][1]
    
    
    P_arrival = dataset['anEQ']['Ptime'][0][0][0][0]
    S_arrival = dataset['anEQ']['Stime'][0][0][0][0]
    
    if P_arrival==-1 or S_arrival == -1:
        continue
    
    distance.append(haversine(epicenterLat, epicenterLon,stationLat,stationLon))
    
    PS.append(S_arrival-P_arrival)
    
korelasyon_katsayisi, _ = pearsonr(PS, distance)
print("Pearson Korelasyon Katsayısı:", korelasyon_katsayisi)