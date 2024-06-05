import numpy as np

import matplotlib.pyplot as plt
from matplotlib.pylab import rcParams

import pandas as pd
import pandas_datareader as web
import datetime as dt

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM

#SET PLOT SIZE
rcParams['figure.figsize'] = 15,6
print("HMMM")
#Dataset for AI LSTM
data = pd.read_csv('PM_DATABASE.csv')
data.head()

def convert_df_price_type(data):
  data["avg_price"] = data["avg_price"].astype(str).str.replace(',', '.')
  return data

data = convert_df_price_type(data)

def separate_training_and_test_data(data):
  data_to_train = data[:175]
  data_to_test = data[175:]
  data_to_train.to_csv('train_data.csv')
  data_to_test.to_csv('test_data.csv')

separate_training_and_test_data(data)

#Datasets for training and testing
train_data = pd.read_csv('train_data.csv')
test_data = pd.read_csv('test_data.csv')

#Prepare Data. transformar preços em valores de 0 a 1
def reshape_df(df):
  scaler = MinMaxScaler(feature_range=(0,1))
  scaled_data = scaler.fit_transform(df['avg_price'].values.reshape(-1,1))
  return scaled_data,scaler

scaled_data,scaler = reshape_df(test_data)

#how many days foward to predict
prediction_days = 1

def get_data_and_value_list(scaled_data,prediction_days):
  x_train=[]
  y_train=[]
  for x in range(prediction_days, len(scaled_data)):
      x_train.append(scaled_data[x-prediction_days:x, 0])
      y_train.append(scaled_data[x, 0])
  return x_train,y_train

x_train, y_train = get_data_and_value_list(scaled_data,prediction_days)

#reshape x_train
x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

#Build the Model 4 LAYERS
def create_model_LSTM(x_train):
  model = Sequential()
  model.add(LSTM(units=75, return_sequences=True, input_shape=(x_train.shape[1], 1)))
  model.add(Dropout(0.2))
  model.add(LSTM(units=75, return_sequences=True))
  model.add(Dropout(0.2))
  model.add(LSTM(units=75, return_sequences=True))
  model.add(Dropout(0.2))
  model.add(LSTM(units=75))
  model.add(Dropout(0.2))
  model.add(Dense(units=1))
  model.summary()
  model.compile(optimizer='adam', loss='mean_squared_error')
  model.fit(x_train, y_train, epochs=200, batch_size=32)
  return model

model = create_model_LSTM(x_train)

def get_model_inputs(train_data,test_data):
  total_dataset = pd.concat((train_data['avg_price'], test_data['avg_price']), axis=0)
  model_inputs = total_dataset[len(total_dataset)-len(test_data)-prediction_days:].values
  model_inputs = model_inputs.reshape(-1, 1)
  model_inputs = scaler.transform(model_inputs)
  return model_inputs

model_inputs = get_model_inputs(train_data,test_data)
actual_prices = test_data['avg_price'].values

# Make Predictions on Test Data
def make_predictions(prediction_days,model_inputs):
  x_test=[]
  for x in range(prediction_days, len(model_inputs)):
      x_test.append(model_inputs[x-prediction_days:x, 0])
  x_test=np.array(x_test)
  x_test=np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
  predicted_prices=model.predict(x_test)
  predicted_prices=scaler.inverse_transform(predicted_prices)
  return predicted_prices,x_test

predicted_prices,x_test = make_predictions(prediction_days,model_inputs)

# Plot the test predictions
plt.plot(actual_prices, color = "blue", label=f"Actual SOJA Price")
plt.plot(predicted_prices, color="green", label=f"Predicted SOJA Price")
plt.title(f"SOJA Share Price")
plt.xlabel("Time")
plt.ylabel(f"SOJA Share Price")
plt.legend()
plt.show()

def predict_tomorrow(real_data):
  prediction = model.predict(real_data)
  #transformar valores de 0 a 1 para os valores reais
  prediction = scaler.inverse_transform(prediction)
  return prediction

#Predict Next Day
real_data = [model_inputs[len(model_inputs)-prediction_days:len(model_inputs+prediction_days), 0]]
real_data = np.array(real_data)
real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1],1))

prediction = predict_tomorrow(real_data)

print(f"Prediction: {prediction}")
print(f"Date prediction: ",data["date"].iloc[-1])
print(f"Last price",scaler.inverse_transform(x_test[-1]))
print(f"Date last price: ",data["date"].iloc[-2])