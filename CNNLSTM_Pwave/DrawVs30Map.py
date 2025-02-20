# -*- coding: utf-8 -*-

"""

Created on Mon Dec 12 00:24:56 2022



@author: khas

"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
import scipy.io


#load the mat files
MatData = scipy.io.loadmat(r'D:\Baris\codes\baris-git06\exps\EXP2023_08_31_08_04_Freq_True_duration_30_Ep_300_lr_1e-05_dropout_0.5_dropout_ResNet\heatmap.mat')
Vs30MapData = MatData['yeni_veri_seti']

ErrorVs30ANN =[]

# creating the DataFrame
Vs30MapData = pd.DataFrame(Vs30MapData) 
# adding column name to the respective columns
Vs30MapData.columns =['Latitude','Longitude','Pred-Vs30','Meas-Vs30']
ErrorVs30ANNt = (np.abs(Vs30MapData['Pred-Vs30'] - Vs30MapData['Meas-Vs30'])/Vs30MapData['Meas-Vs30'])*100
Vs30MapData['ErrorVs30ANN'] = ErrorVs30ANNt

ZA = []
ZB = []
ZC = []
ZD = []
ZE = []

for sayı,err in zip(Vs30MapData['Meas-Vs30'],Vs30MapData['ErrorVs30ANN']):
    if sayı < 180:
        ZE.append(err)
    elif sayı < 360 and sayı > 180:
        ZD.append(err)
    elif sayı > 360 and sayı < 732:
        ZC.append(err)
    elif sayı > 731 and sayı < 1400:
        ZB.append(err)    
    elif sayı > 1500:
        ZA.append(err)

print(len(ZA),np.mean(ZA))
print(len(ZB),np.mean(ZB))
print(len(ZC),np.mean(ZC))
print(len(ZD),np.mean(ZD))
print(len(ZE),np.mean(ZE))

Vs30MapData.plot(x="Longitude", y="Latitude", kind="scatter", c="ErrorVs30ANN", colormap="YlOrRd",vmax=np.mean(ErrorVs30ANNt)+np.std(ErrorVs30ANNt))

plt.show()



countries = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))

countries.head()



countries.plot(color="lightgrey")



countries[countries["name"] == "Turkey"].plot(color="lightgrey")





#Combine scatter plot with Geopandas

# initialize an axis

fig, ax = plt.subplots(figsize=(8,6))



# plot map on axis

countries = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))

countries[countries["name"] == "Turkey"].plot(color="lightgrey", ax=ax)



# # parse dates for plot's title

# first_month = df["acq_date"].min().strftime("%b %Y")

# last_month = df["acq_date"].max().strftime("%b %Y")



# plot points

Vs30MapData.plot(x="Longitude", y="Latitude", kind="scatter", c="ErrorVs30ANN", colormap="YlOrRd", 

        title="Vs30 Percentage Error for ANN",vmax=np.mean(ErrorVs30ANNt)+np.std(ErrorVs30ANNt), ax=ax)



# add grid

ax.grid(b=True, alpha=0.5)

plt.text(28.0, 35.5, f'Std: {np.std(ErrorVs30ANNt):.2f}', fontsize=12, color='black')
plt.text(28.0, 35.0, f'Mean: {np.mean(ErrorVs30ANNt):.2f}', fontsize=12, color='black')

plt.text(40.0, 36.75, f'ZA: {len(ZA):.1f}', fontsize=8, color='black')
plt.text(43.0, 36.75, f'Mean: {np.mean(ZA):.1f}', fontsize=8, color='black')
plt.text(40.0, 36.25, f'ZB: {len(ZB):.1f}', fontsize=8, color='black')
plt.text(43.0, 36.25, f'Mean: {np.mean(ZB):.1f}', fontsize=8, color='black')
plt.text(40.0, 35.75, f'ZC: {len(ZC):.1f}', fontsize=8, color='black')
plt.text(43.0, 35.75, f'Mean: {np.mean(ZC):.1f}', fontsize=8, color='black')
plt.text(40.0, 35.25, f'ZD: {len(ZD):.1f}', fontsize=8, color='black')
plt.text(43.0, 35.25, f'Mean: {np.mean(ZD):.1f}', fontsize=8, color='black')
plt.text(40.0, 34.75, f'ZE: {len(ZE):.1f}', fontsize=8, color='black')
plt.text(43.0, 34.75, f'Mean: {np.mean(ZE):.1f}', fontsize=8, color='black')

plt.show()


# ---------------------------------------------------------------------------------------------------------------

#load the mat files
# MatData = scipy.io.loadmat(r'D:\Baris\codes\baris-git06\exps\EXP2023_08_30_14_00_Freq_True_duration_30_Ep_300_lr_1e-05_dropout_0.1_dropout_ResNet\heatmap.mat')
Vs30MapData = MatData['yeni_veri_seti']

ErrorVs30ANN =[]

# creating the DataFrame
Vs30MapData = pd.DataFrame(Vs30MapData) 
# adding column name to the respective columns
Vs30MapData.columns =['Latitude','Longitude','Pred-Vs30','Meas-Vs30']
ErrorVs30ANNt = (np.abs(Vs30MapData['Pred-Vs30'] - Vs30MapData['Meas-Vs30'])/Vs30MapData['Meas-Vs30'])*100
Vs30MapData['ErrorVs30ANN'] = ErrorVs30ANNt





Vs30MapData.plot(x="Longitude", y="Latitude", kind="scatter", c="ErrorVs30ANN", colormap="YlOrRd",vmax=np.mean(ErrorVs30ANNt)+(np.std(ErrorVs30ANNt)*2))

plt.show()



countries = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))

countries.head()



countries.plot(color="lightgrey")



countries[countries["name"] == "Turkey"].plot(color="lightgrey")





#Combine scatter plot with Geopandas

# initialize an axis

fig, ax = plt.subplots(figsize=(8,6))



# plot map on axis

countries = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))

countries[countries["name"] == "Turkey"].plot(color="lightgrey", ax=ax)



# # parse dates for plot's title

# first_month = df["acq_date"].min().strftime("%b %Y")

# last_month = df["acq_date"].max().strftime("%b %Y")



# plot points

Vs30MapData.plot(x="Longitude", y="Latitude", kind="scatter", c="ErrorVs30ANN", colormap="YlOrRd", 

        title="Vs30 Percentage Error for ANN",vmax=np.mean(ErrorVs30ANNt)+(np.std(ErrorVs30ANNt)*2), ax=ax)



# add grid

ax.grid(b=True, alpha=0.5)

plt.text(28.0, 35.5, f'Std2: {np.std(ErrorVs30ANNt)*2:.2f}', fontsize=12, color='black')
plt.text(28.0, 35.0, f'Mean: {np.mean(ErrorVs30ANNt):.2f}', fontsize=12, color='black')

plt.text(40.0, 36.75, f'ZA: {len(ZA):.1f}', fontsize=8, color='black')
plt.text(43.0, 36.75, f'Mean: {np.mean(ZA):.1f}', fontsize=8, color='black')
plt.text(40.0, 36.25, f'ZB: {len(ZB):.1f}', fontsize=8, color='black')
plt.text(43.0, 36.25, f'Mean: {np.mean(ZB):.1f}', fontsize=8, color='black')
plt.text(40.0, 35.75, f'ZC: {len(ZC):.1f}', fontsize=8, color='black')
plt.text(43.0, 35.75, f'Mean: {np.mean(ZC):.1f}', fontsize=8, color='black')
plt.text(40.0, 35.25, f'ZD: {len(ZD):.1f}', fontsize=8, color='black')
plt.text(43.0, 35.25, f'Mean: {np.mean(ZD):.1f}', fontsize=8, color='black')
plt.text(40.0, 34.75, f'ZE: {len(ZE):.1f}', fontsize=8, color='black')
plt.text(43.0, 34.75, f'Mean: {np.mean(ZE):.1f}', fontsize=8, color='black')

plt.show()
