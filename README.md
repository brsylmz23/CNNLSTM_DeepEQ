# 📡 Seismic Signal Analysis with CNN-LSTM

![Python](https://img.shields.io/badge/Python-3.8%2B-blue) ![Deep Learning](https://img.shields.io/badge/Deep%20Learning-TensorFlow-orange) ![Seismic Data](https://img.shields.io/badge/Seismic%20Data-Processing-green)

## 📌 Overview  
This repository contains an implementation of a **CNN-LSTM model** designed for seismic signal processing. It includes scripts for **data preprocessing, model training, cross-validation, and signal processing**.  

## 📁 Project Structure  

📜 Code Explanation
🔹 experiment.py
Main script for training and evaluating the model.
Calls necessary functions from CNNLSTM.py, datasetLoader.py, and utils.py.
🔹 datasetLoader.py
Loads and preprocesses seismic datasets.
Handles signal normalization, augmentation, and batching.
🔹 CNNLSTM.py
Defines the CNN-LSTM model architecture.
Uses convolutional layers to extract spatial features, followed by LSTMs for temporal dependencies.
🔹 utils.py
Implements cross-validation and signal processing functions.
Handles feature extraction, normalization, and other utilities.
📊 Results & Logs
Training results and evaluation metrics are saved inside the results/ folder.
Logs can be found in the logs/ directory for further analysis.
🏗 Future Work
Improve generalization of the CNN-LSTM model.
Experiment with alternative architectures such as Transformers for time series analysis.
🤝 Contributions
Feel free to open issues or pull requests for improvements and discussions! 🚀

📜 License
This project is licensed under the MIT License.
