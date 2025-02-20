# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 15:20:34 2023

@author: Asistan
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
MatData2 = scipy.io.loadmat(r'D:\Baris\codes\baris-git06\exps\EXP2023_08_31_02_24_Freq_True_duration_60_Ep_300_lr_1e-06_dropout_0.3_dropout_ResNet\heatmap2.mat')
MatData3 = scipy.io.loadmat(r'D:\Baris\codes\baris-git06\exps\EXP2023_08_31_02_24_Freq_True_duration_60_Ep_300_lr_1e-06_dropout_0.3_dropout_ResNet\heatmap3.mat')

MatData['veri'] = np.vstack((MatData2['veri'], MatData['veri']))
MatData['veri'] = np.vstack((MatData3['veri'], MatData['veri']))
Vs30MapData = MatData['veri']


# creating the DataFrame
Vs30MapData = pd.DataFrame(Vs30MapData) 
# adding column name to the respective columns
Vs30MapData.columns =['Latitude','Longitude','Pred-Vs30','Meas-Vs30']
ErrorVs30ANNt = (Vs30MapData['Pred-Vs30'] - Vs30MapData['Meas-Vs30'])/Vs30MapData['Meas-Vs30']*100
ErrorVs30ANNtabs = (np.abs(Vs30MapData['Pred-Vs30'] - Vs30MapData['Meas-Vs30'])/Vs30MapData['Meas-Vs30'])*100
Vs30MapData['ErrorVs30ANN'] = ErrorVs30ANNt
Vs30MapData['ErrorVs30ANNabs'] = ErrorVs30ANNtabs

# Verileri kategorilere göre ayırın
ZA_data = Vs30MapData[Vs30MapData['Meas-Vs30'] > 1500]
ZB_data = Vs30MapData[(Vs30MapData['Meas-Vs30'] > 760) & (Vs30MapData['Meas-Vs30'] < 1500)]
ZC_data = Vs30MapData[(Vs30MapData['Meas-Vs30'] > 360) & (Vs30MapData['Meas-Vs30'] < 760)]
ZD_data = Vs30MapData[(Vs30MapData['Meas-Vs30'] > 180) & (Vs30MapData['Meas-Vs30'] < 360)]
# ZE_data = Vs30MapData[Vs30MapData['Meas-Vs30'] < 180]

# Kategorilere göre renkler
renkler = {
    'A': 'yellow',
    'B': 'green',
    'C': 'purple',
    'D': 'blue'
}

# Renk paletini özelleştirin
cmap_custom = mcolors.LinearSegmentedColormap.from_list("custom_colormap", list(renkler.values()), N=256)
norm = plt.Normalize(180, 1500)

# Scatter plot'u oluşturun ve her kategori için ayrı renkte çizin
fig, ax = plt.subplots(figsize=(8, 6), dpi=1200)

for kategori, data in zip(renkler.keys(), [ZA_data, ZB_data, ZC_data, ZD_data]):
    c = data['Meas-Vs30']
    ax.scatter(data["Longitude"], data["Latitude"], c=c, cmap=cmap_custom, label=kategori, norm=norm, zorder=2)

# Renk çubuğunu ekleyin
cbar = plt.colorbar(ax.collections[0], ax=ax, orientation='vertical')

# Renk çubuğu etiketlerini ve renk aralığını özelleştirin
cbar.set_label('Vs' + str(30).lower(), fontsize=12)
cbar.set_ticks([180, 360, 760, 1500])  # Özel aralıkları burada belirtin
cbar.set_ticklabels(['180', '360', '760', '1500'])  # Etiketleri burada belirtin

# Eksen etiketleri ve başlık ekleyin
plt.xlabel('Longitude')
plt.ylabel('Latitude')

# Türkiye haritasını yükleyin
world = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
turkey = world[world["name"] == "Turkey"]

# Türkiye haritasını çizin
turkey.plot(ax=ax, color="lightgrey", zorder=1)

# Legend'i sağ alt köşeye taşıyın ve isimleri ayarlayın
legend = ax.legend(loc='lower right', bbox_to_anchor=(1, 0), labels=['A', 'B', 'C', 'D'], title='Site classes', fontsize='small')

# Legend'in boyutunu küçültmek için
legend.get_title().set_fontsize('small')

plt.show()

