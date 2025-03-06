# 📡 Seismic Signal Analysis with CNN-LSTM

![Python](https://img.shields.io/badge/Python-3.8%2B-blue) ![Deep Learning](https://img.shields.io/badge/Deep%20Learning-TensorFlow-orange) ![Seismic Data](https://img.shields.io/badge/Seismic%20Data-Processing-green)

## 📌 Overview  
Python code applying deep learning techniques to strong motion records for estimating Vs30, a parameter representing the average shear-wave velocity in the top 30 meters of soil. This study explores whether strong motion records contain useful information for Vs30 estimation and whether DL-based methods can effectively utilize them. The paper introduces a large-scale strong motion record collection, AFAD-1218, which contains over 36,000 strong motion records from Türkiye.

## 📂 Experiment Folders  

### 🟢 `CNNLSTM_Pwave/`  
- This experiment adjusts seismic signals **around the P-wave arrival**.  

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
