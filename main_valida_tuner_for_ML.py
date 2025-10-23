import tkinter as tk
from tkinter import *
from tkinter import PhotoImage
from tkinter import filedialog
from tkinter import messagebox

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
#from PIL import Image
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Reshape, Bidirectional, LSTM
from tensorflow.keras.models import Sequential


# Values
validation_results = [] #this is for later use when we compare our results of different models in one plot
testing_results = []

def load_data():
    global df, file_path
    file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    if file_path:
        try:
            df = pd.read_csv(file_path, delimiter=';')
            lightInd.configure(text="Loaded", bg="green")
            messagebox.showinfo("Success", "Data loaded successfully!")
            # return df
            trainButton.configure(state=NORMAL)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load data: {str(e)}")



def train_model():
    global df, X, y, validation_results, testing_results
    # df = load_data()
    if df is not None:

        validPercentage = validRatio.get()
        testPercentage = testRatio.get()

        # checker - balancer
        if validPercentage + testPercentage > 90:
            messagebox.showerror("Error", "The sum of Validation and Testing percentages should not be greater than 90%. Please, revise your options")
            return

        imputer = SimpleImputer(strategy='mean')
        df_missing = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
        scaler = StandardScaler()
        df_standa = pd.DataFrame(scaler.fit_transform(df_missing), columns=df_missing.columns)

        # Feature and focus om Total orders
        X = df_standa.drop('Target (Total orders)', axis=1).values # features
        y = df_standa['Target (Total orders)'].values #target

        # Spliting and making space for Testing dataset /Validation and Modeling
        #X_train_valid, X_test, y_train_valid, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        # X-feature, Y-target
        #X_train, X_valid, y_train, y_valid = train_test_split(X_train_valid, y_train_valid, test_size=0.25, random_state=42)

        validPercentage = validRatio.get() / 100
        testPercentage = testRatio.get() / 100

        # Calculate training/modelling percentage
        trainingPercentage = 1 - validPercentage - testPercentage

        X_train_valid, X_test, y_train_valid, y_test = train_test_split(X, y, test_size=testPercentage, random_state=42)
        X_train, X_valid, y_train, y_valid = train_test_split(X_train_valid, y_train_valid, test_size=validPercentage / (validPercentage + trainingPercentage), random_state=42)

        # Reshape data for needed format 2 -> 3 for CNN
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        X_valid = X_valid.reshape((X_valid.shape[0], X_valid.shape[1], 1))
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

        model = Sequential()

        ## CNN Extract ##
        model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(X_train.shape[1], 1)))
        # can play
        model.add(MaxPooling1D(pool_size=2))

        # reshape to FEED it into LSTMI
        conv_shape = model.layers[-1].output_shape
        new_elements = conv_shape[1] * conv_shape[2]
        new_timesteps = conv_shape[1] #timesteps for LSTMI
        new_features = new_elements // new_timesteps #number of features for each time step

        # Reshape output for BiLSTM input
        model.add(Reshape((new_timesteps, new_features)))

        # BiLSTM Layer for time series forecasting
        model.add(Bidirectional(LSTM(50, activation='relu')))
        model.add(Dense(1))

        # Compile the model
        model.compile(optimizer='adam', loss='mse')

        model.fit(X_train, y_train, epochs=100, verbose=1) # train on training data # can fix values

        y_pred_val = model.predict(X_valid)
        y_pred_test = model.predict(X_test)

        # calculate MAE and RMSE for validation and test sets
        mae_val = mean_absolute_error(y_valid, y_pred_val)
        rmse_val = np.sqrt(mean_squared_error(y_valid, y_pred_val))
        mae_test = mean_absolute_error(y_test, y_pred_test)
        rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))

        #validation_results.append((validRatio.get(), rmse_val))
        #testing_results.append((testRatio.get(), rmse_test))

        validation_results.append((validPercentage, mae_val, rmse_val))
        testing_results.append((testPercentage, mae_test, rmse_test))

        compareButton.configure(state=NORMAL)

        # Plot here
        plt.figure(figsize=(10, 6))
        plt.plot(y_test, label='Actual')
        plt.plot(y_pred_test.flatten(), label='Predicted', alpha=0.7)
        plt.title('Test Set Forecast vs Actual')
        plt.xlabel('Time value')
        plt.ylabel('Total Orders')
        plt.legend()
        plt.show()

        messagebox.showinfo("Results",f"Validation Set MAE: {mae_val}\nValidation Set RMSE: {rmse_val}\nTest Set MAE: {mae_test}\nTest Set RMSE: {rmse_test}")


def comparo():
    plt.figure(figsize=(8, 6))
    colors = ['blue', 'blue', 'red', 'red', 'purple', 'purple', 'yellow', 'yellow']  # bad design! but it works :P
    color_index = 0  # initer

    for (val, mae_val, rmse_val), (tes, mae_tes, rmse_tes) in zip(validation_results, testing_results):
        plt.scatter(mae_val, rmse_val, color=colors[color_index], label='Validation')
        plt.text(mae_val, rmse_val, f'Val: {val}%', fontsize=8, color='black')  # Apercentage
        color_index = (color_index + 1) % len(colors)  # part of bad design +2

        plt.scatter(mae_tes, rmse_tes, color=colors[color_index], label='Testing')
        plt.text(mae_tes, rmse_tes, f'Test: {tes}%', fontsize=8, color='black')  # percentage for testing
        color_index = (color_index + 1) % len(colors)

    plt.xlabel('MAE')
    plt.ylabel('RMSE')
    plt.title('Comparison of MAE and RMSE values based on split to Validation and Testing %')
    plt.legend()
    plt.grid(True)
    plt.show()

# GUI here
root = tk.Tk()
root.title("Time Series Forecasting")
root.geometry("450x300")

validRatio = tk.IntVar(value=20)
testRatio = tk.IntVar(value=25)

loadButton = tk.Button(root, text="Load Data", command=load_data, state=NORMAL)
loadButton.pack(pady=10)

trainButton = tk.Button(root, text="Train Model", command=train_model, state=DISABLED)
trainButton.pack(pady=10)

compareButton = tk.Button(root, text="Compare Results", command=comparo, state=DISABLED)
compareButton.pack(pady=10)

lightInd = tk.Label(root, text="Not Loaded", bg="red", width=10)
lightInd.place(y=12, x=90)

picture = PhotoImage(file="Untitled.png")
imageLogo = tk.Label(root, image=picture)
imageLogo.place(y=150)


validLabel = tk.Label(root, text="Validation (%)")
validLabel.place(x=280, y=10)
validEntry = tk.Entry(root, textvariable=validRatio, width=5)
validEntry.place(x=370, y=10)

testLabel = tk.Label(root, text="Testing (%)")
testLabel.place(x=280, y=50)
testEntry = tk.Entry(root, textvariable=testRatio, width=5)
testEntry.place(x=370, y=50)



root.mainloop()
