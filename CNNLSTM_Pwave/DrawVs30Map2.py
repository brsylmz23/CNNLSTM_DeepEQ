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
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


#load the mat files
MatData = scipy.io.loadmat(r'D:\Baris\codes\baris-git06\exps\EXP2023_08_31_02_24_Freq_True_duration_60_Ep_300_lr_1e-06_dropout_0.3_dropout_ResNet\heatmap.mat')
Vs30MapData = MatData['veri']

ErrorVs30ANN =[]
ErrorVs30ANNtabs =[]

# creating the DataFrame
Vs30MapData = pd.DataFrame(Vs30MapData) 
# adding column name to the respective columns
Vs30MapData.columns =['Latitude','Longitude','Pred-Vs30','Meas-Vs30']
ErrorVs30ANNt = (Vs30MapData['Pred-Vs30'] - Vs30MapData['Meas-Vs30'])/Vs30MapData['Meas-Vs30']*100
ErrorVs30ANNtabs = (np.abs(Vs30MapData['Pred-Vs30'] - Vs30MapData['Meas-Vs30'])/Vs30MapData['Meas-Vs30'])*100
Vs30MapData['ErrorVs30ANN'] = ErrorVs30ANNt
Vs30MapData['ErrorVs30ANNabs'] = ErrorVs30ANNtabs

ZA = []
ZB = []
ZC = []
ZD = []
ZE = []

for sayı,err in zip(Vs30MapData['Meas-Vs30'],Vs30MapData['ErrorVs30ANNabs']):
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

# Renk paletini özelleştirin
colors = ['#FF0000', '#FFA500', '#FFFF00', '#FFA500', '#FF0000']  # Kırmızıdan başlayarak sarıya, turuncuya ve yeniden kırmızıya dönüş yapar
cmap_custom = mcolors.LinearSegmentedColormap.from_list("custom_colormap", colors, N=256)

fig = plt.figure(figsize=(8, 6), dpi=800)

Vs30MapData.plot(x="Longitude", y="Latitude", kind="scatter", c="ErrorVs30ANN", colormap=cmap_custom,vmin=np.abs(np.std(ErrorVs30ANNt))*-1,vmax=np.abs(np.std(ErrorVs30ANNt)))


plt.show()



countries = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))

countries.head()



countries.plot(color="lightgrey")



countries[countries["name"] == "Turkey"].plot(color="lightgrey")





#Combine scatter plot with Geopandas

# initialize an axis

fig, ax = plt.subplots(figsize=(8,6),dpi=1200)



# plot map on axis

countries = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))

countries[countries["name"] == "Turkey"].plot(color="lightgrey", ax=ax)



# # parse dates for plot's title

# first_month = df["acq_date"].min().strftime("%b %Y")

# last_month = df["acq_date"].max().strftime("%b %Y")


# plot points

Vs30MapData.plot(x="Longitude", y="Latitude", kind="scatter", c="ErrorVs30ANN", colormap=cmap_custom, 

        title="Vs30 Percentage Error",vmin=np.abs(np.std(ErrorVs30ANNt))*-1,vmax=np.abs(np.std(ErrorVs30ANNt)), ax=ax)



# add grid

ax.grid(b=True, alpha=0.5)


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
ErrorVs30ANNt = (Vs30MapData['Pred-Vs30'] - Vs30MapData['Meas-Vs30'])/Vs30MapData['Meas-Vs30']*100
ErrorVs30ANNtabs = (np.abs(Vs30MapData['Pred-Vs30'] - Vs30MapData['Meas-Vs30'])/Vs30MapData['Meas-Vs30'])*100
Vs30MapData['ErrorVs30ANN'] = ErrorVs30ANNt
Vs30MapData['ErrorVs30ANNabs'] = ErrorVs30ANNtabs





Vs30MapData.plot(x="Longitude", y="Latitude", kind="scatter", c="ErrorVs30ANN", colormap=cmap_custom,vmin=np.abs(np.std(ErrorVs30ANNt))*-1,vmax=np.abs(np.std(ErrorVs30ANNt)))

plt.show()



countries = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))

countries.head()



countries.plot(color="lightgrey")



countries[countries["name"] == "Turkey"].plot(color="lightgrey")





#Combine scatter plot with Geopandas

# initialize an axis

fig, ax = plt.subplots(figsize=(8,6),dpi=1200)



# plot map on axis

countries = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))

countries[countries["name"] == "Turkey"].plot(color="lightgrey", ax=ax)



# # parse dates for plot's title

# first_month = df["acq_date"].min().strftime("%b %Y")

# last_month = df["acq_date"].max().strftime("%b %Y")



# plot points

Vs30MapData.plot(x="Longitude", y="Latitude", kind="scatter", c="ErrorVs30ANN", colormap=cmap_custom, 

        title="Vs30 Percentage Error (%)",vmin=np.abs(np.std(ErrorVs30ANNt))*-2,vmax=np.abs(np.std(ErrorVs30ANNt))*2, ax=ax)



# add grid

ax.grid(b=True, alpha=0.5)


plt.show()


# Boş bir figür oluşturun ve koordinat ekseni olmasın
fig, ax = plt.subplots(figsize=(8,6),dpi=1200)
ax.axis('off')

# Verileri ve metinleri hazırlayın
data = [
    {"Site": "Site A", "Number": len(ZA), "Absolute Mean Error": np.mean(ZA)},
    {"Site": "Site B", "Number": len(ZB), "Absolute Mean Error": np.mean(ZB)},
    {"Site": "Site C", "Number": len(ZC), "Absolute Mean Error": np.mean(ZC)},
    {"Site": "Site D", "Number": len(ZD), "Absolute Mean Error": np.mean(ZD)},
    {"Site": "Site E", "Number": len(ZE), "Absolute Mean Error": np.mean(ZE)},
    {"Site": "Std2", "Number": np.std(ErrorVs30ANNt) * 2, "Absolute Mean Error": np.mean(ErrorVs30ANNtabs)},
]

# Tabloyu çizdirin
table_data = []
for row in data:
    absolute_mean_error = f"{row['Absolute Mean Error']:.1f}%"  # % işareti ekleyin
    table_data.append([row["Site"], f"{row['Number']:.1f}", absolute_mean_error])

table = ax.table(cellText=table_data, colLabels=["Site", "Number", "Absolute Mean Error"], cellLoc='center', loc='center')
table.auto_set_font_size(False)
table.set_fontsize(8)
table.scale(1, 1.2)  # Tablo boyutunu ayarlayın

plt.show()

