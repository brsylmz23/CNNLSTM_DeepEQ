# 📡 Seismic Signal Analysis with CNN-LSTM

![Python](https://img.shields.io/badge/Python-3.8%2B-blue) ![Deep Learning](https://img.shields.io/badge/Deep%20Learning-TensorFlow-orange) ![Seismic Data](https://img.shields.io/badge/Seismic%20Data-Processing-green)

## 📌 Overview  
This repository contains an implementation of a **CNN-LSTM model** designed for seismic signal processing. It includes scripts for **data preprocessing, model training, cross-validation, and signal processing**.  

## 📂 Experiment Folders  

### 🟢 `CNNLSTM_Pwave/`  
- This experiment adjusts seismic signals **around the P-wave arrival**.  

### 🔵 `CNNLSTM_SP_EQT/`  
- Uses **P-wave arrival time** as additional input information.  
- **EQT dataset** is used for **both training and testing**.   

### 🟠 `CNNLSTM_SP_stTEST/`  
- Similar to `CNNLSTM_SP_EQT/`, but uses **manually labeled** seismic data as the **test set**.  

### 🔴 `CNNLSTM_TRANSFER/`  
- Implements **transfer learning** by using **pretrained weights** from previous experiments.  
  
📜 Code Explanation

## 🔹 experiment.py
Main script for training and evaluating the model.

## 🔹 datasetLoader.py
Loads and preprocesses seismic datasets.
Handles signal normalization, augmentation, and batching.

## 🔹 CNNLSTM.py
Defines the CNN-LSTM model architecture.

## 🔹 utils.py
Implements cross-validation and signal processing functions.
Handles feature extraction, normalization, and other utilities.

## 📊 Results & Logs
Training results and evaluation metrics are saved inside the exps/ folder.
Logs can be found in the exps/ directory for further analysis.

## 📜 License
This project is licensed under the MIT License.
