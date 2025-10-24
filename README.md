# CNN-BiLSTM Forecasting Tuner (Research Implementation)

## Project Overview

This repository features a **Hybrid Deep Learning tool** designed for **Time Series Demand Forecasting**. The application provides a simple Graphical User Interface (GUI) built with **Tkinter** to load data, train the model, and evaluate performance metrics.

The core architecture is a direct Python/TensorFlow implementation of the **Hybrid CNN-BiLSTM Deep Learning Model** proposed in the paper: **"Hybrid CNN-BiLSTM Deep Learning Model for Daily Demand Forecasting of Sales Orders."**

| Feature | Description |
| :--- | :--- |
| **Model** | **Hybrid 1D CNN + Bidirectional LSTM** for complex sequence pattern extraction. |
| **Interface** | **Tkinter GUI** for setting data split ratios and executing training runs. |
| **Metrics** | Calculates and compares **MAE** (Mean Absolute Error) and **RMSE** (Root Mean Squared Error). |

---

## ⚙️ How to Run the Current Version

### 1. Prerequisites

You'll need Python 3 and the following libraries. The key machine learning libraries are `tensorflow` (for Keras), `scikit-learn`, and `pandas`.

```bash
pip install pandas numpy matplotlib scikit-learn tensorflow

### 2. Execution

1. Run the script from your terminal
2. In the GUI click "Load Data" and select attached CSV file. (The version requires a semicolon delimeter.
3. Set your desired Train (%) and Testing (%) ratios, then click "Train Model"

### 3. Issues



