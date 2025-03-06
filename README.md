# 📡 Seismic Signal Analysis with CNN-LSTM

![Python](https://img.shields.io/badge/Python-3.8%2B-blue) ![Deep Learning](https://img.shields.io/badge/Deep%20Learning-TensorFlow-orange) ![Seismic Data](https://img.shields.io/badge/Seismic%20Data-Processing-green)

## 📌 Overview  
Python code applying deep learning techniques to strong motion records for estimating Vs30, a parameter representing the average shear-wave velocity in the top 30 meters of soil. This study explores whether strong motion records contain useful information for Vs30 estimation and whether DL-based methods can effectively utilize them. The paper introduces a large-scale strong motion record collection, AFAD-1218, which contains over 36,000 strong motion records from Türkiye.

## Dataset

This study uses the AFAD-1218 dataset, a comprehensive collection of over 36,000 strong motion records from Turkey. These records were obtained from Turkey's national strong-motion network operated by the Disaster and Emergency Management Authority (AFAD). The dataset spans a wide range of seismic events and regions, providing a rich resource for deep learning applications.

Characteristics of the AFAD-1218 dataset:
Number of Records: 36,418
Sampling Rate: All sampled at 100 Hz
Duration: Varying event durations from 5 to 300 seconds
SNR: Ranging from a few dBs to 100 dB. Signals with SNR values lower than 25 dB were eliminated during the training process.
Geographic Coverage: Nation-wide strong ground motion stations across Turkey
Features: Includes ground acceleration time series and metadata, such as event magnitude, epicenter location, and station coordinates.

## 📂 Experiment Folders  

### 🟢 `CNNLSTM_Pwave/`  
- This experiment adjusts seismic signals **around the P-wave arrival time**.  

### 🔵 `CNNLSTM_SP_EQT/`  
- Uses **P-wave arrival time** as additional input information.  
- **EQT dataset** is used for **both training and testing**.   
  
# 📜 Code Explanation

## 🔹 runme.py
Main script for evaluating the model.

## 🔹 functions.py
Includes functions for data processing and evaluation setup.

## 🔹 CNNLSTM.py
Defines the CNN-LSTM model architecture.

## 🔹 evaluate.py
Evaluates sample data.

## 📜 License
This project is licensed under the MIT License.
