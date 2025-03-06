# 📡 Seismic Signal Analysis with CNN-LSTM

![Python](https://img.shields.io/badge/Python-3.8%2B-blue) ![Deep Learning](https://img.shields.io/badge/Deep%20Learning-TensorFlow-orange) ![Seismic Data](https://img.shields.io/badge/Seismic%20Data-Processing-green)

## 📌 Overview  
This repository contains an implementation of a CNN-LSTM model for Vs30 estimation using sample data.  

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
