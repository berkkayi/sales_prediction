import pandas as pd
import numpy as np
import matplotlib.pylab as plt 
import matplotlib.patches as mpatches
from keras.layers.core import Dense,Activation,Dropout,Flatten,Reshape
from keras.layers import LSTM
from keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler , StandardScaler
from sklearn.metrics import mean_squared_error 
import math
from sklearn.preprocessing import PowerTransformer


ad = pd.read_csv("VERİ revize_yeni.csv")
df = ad.copy()
df.set_index("date",inplace=True)
df = df.T
df.index = pd.DatetimeIndex(df.index)
tt = df.iloc[:,0:1]
Q1 = tt.quantile(0.25)
Q3 = tt.quantile(0.75)
IQR = Q3 - Q1
over = Q3 +IQR
aykiri_tf = tt > (over)
#tt = np.array(tt)
#tt[aykiri_tf] = over
#tt = pd.DataFrame(tt)
tt.index = df.index
tt1 = tt.values
tt1 = tt1.reshape(len(tt1),1)

scaler = MinMaxScaler(feature_range=(0,1))
ss = PowerTransformer(method="box-cox",standardize = False)
ts = ss.fit_transform(tt1)
#ts = scaler.fit_transform(tt1)
          
seed = 7
np.random.seed(seed)
n_features =1
n_input =3
timestep = 3
X=[]
Y=[]

data = ts

for i in range(len(data)-timestep):
    X.append(data[i:i+timestep])
    Y.append(data[i+timestep])

X = np.asanyarray(X)
Y = np.asanyarray(Y)

X = X.reshape((X.shape[0],X.shape[1],1))


k = 43

Xtrain = X[:k,:,:]
Ytrain = Y[:k]

Xtest = X[k:,:,:]
Ytest = Y[k:]

model = Sequential()
model.add(LSTM(48,activation="relu",recurrent_activation="relu",batch_input_shape=(None,timestep,n_features),return_sequences=True))
model.add(Dropout(0.1))
model.add(Dense(10))
model.add(LSTM(96,activation="relu",return_sequences=True,recurrent_activation="relu"))
model.add(Dropout(0.1))
model.add(Dense(120))
model.add(Dense(10))
model.add(LSTM(48,activation="relu"))
model.add(Dropout(0.1))
model.add(Dense(3))
model.add(Dense(1))


model.compile(loss='mse',optimizer='ADAgrad')
history = model.fit(Xtrain,Ytrain,batch_size=41 ,epochs=10000,validation_data=(Xtest,Ytest))

Ypred = model.predict(X)
Ypred = ss.inverse_transform(Ypred)
#Ypred = scaler.inverse_transform(Ypred)
Ypred = np.reshape(Ypred,len(Ypred))
Ypred = pd.Series(Ypred)
Yreel = tt1[timestep:]
#test predict
Ypred_test = model.predict(Xtest)
Ypred_test = Ypred_test.reshape(len(Ypred_test),1)
Ypred_test = ss.inverse_transform(Ypred_test)
Ypred_test = np.reshape(Ypred_test,len(Ypred_test))
Ypred_test = pd.Series(Ypred_test)
Yreel_test = tt1[46:]
#train predict
Ypred_train = model.predict(Xtrain)
Ypred_train = ss.inverse_transform(Ypred_train)
Ypred_train = np.reshape(Ypred_train,len(Ypred_train))
Ypred_train = pd.Series(Ypred_train)
Yreel_train = tt1[timestep:k+timestep]


#interval confidence and plotting

step = int(Ypred.mean()/Ypred.std())
time_series_df = pd.DataFrame(Ypred)
smooth_path    = time_series_df.rolling(2).mean()
path_deviation = 1.96 * time_series_df.rolling(2).std()
under_line     = (smooth_path-path_deviation)[0]
over_line      = (smooth_path+path_deviation)[0]

fig , ax = plt.subplots(figsize=(20,10))
ax.plot(Yreel,label='Yreel',color='blue',lw=3)
ax.plot(Ypred,label='Ypred',color='red')
ax.fill_between(Ypred.index ,under_line, over_line, color='b', alpha=0.2);

fig , ax = plt.subplots(figsize=(12,6))
ax.plot(Yreel_test,label='Yreel_test',color='blue',lw=3)
ax.plot(Ypred_test,label='Ypred_test',color='red');

fig , ax = plt.subplots(figsize=(12,6))
ax.plot(Yreel_train,label='Yreel_train',color='blue',lw=3)
ax.plot(Ypred_train,label='Ypred_train',color='red');


#error score data
mse1 = mean_squared_error(Yreel,Ypred)
rmse1 = np.sqrt(mse1)


#loss fonksiyonu
fig , ax = plt.subplots(figsize=(12,6))
ax.plot(history.history['loss'], label='Train Loss')
ax.plot(history.history['val_loss'], label='Test Loss')
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epochs')
plt.legend(loc='upper right')
plt.show();

#test error
mse_test = mean_squared_error(Yreel_test,Ypred_test)
rmse_test = np.sqrt(mse_test)

#train error
mse_train = mean_squared_error(Yreel_train,Ypred_train)
rmse_train = np.sqrt(mse_train)

#predicter models accuracy
print("RMSE:"+str(rmse1))
print("RMSE_train:"+str(rmse_train))
print("RMSE_test:"+str(rmse_test))
print("test-train differences:",np.abs(rmse_train - rmse_test))

#################
model1 = Sequential()
model1.add(LSTM(48,activation="relu",recurrent_activation="relu",batch_input_shape=(None,timestep,n_features),return_sequences=True))
model1.add(Dropout(0.1))
model1.add(Dense(10))
model1.add(LSTM(96,activation="relu",return_sequences=True,recurrent_activation="relu"))
model1.add(Dropout(0.1))
model1.add(Dense(120))
model1.add(Dense(10))
model1.add(LSTM(48,activation="relu"))
model1.add(Dropout(0.1))
model1.add(Dense(3))
model1.add(Dense(1))


model1.compile(loss='mse',optimizer='Adagrad')
model1.fit(Xtrain,Ytrain,batch_size=45 ,epochs=6000)


forecast = model1.predict(X)
#forecast = scaler.inverse_transform(forecast)
#forecast = np.reshape(forecast,len(forecast))
#Yreel = tt1[timestep:]
tt = ss.fit_transform(tt1)
prediction = []
first = tt[-timestep:]
current = first.reshape((1,3,1))

for i in range(10):
    current_pred = model.predict(current)[0]
    prediction.append(current_pred)
    current = np.append(current[:,1:,:],[[current_pred]],axis=1)

prediction = np.array(prediction)
prediction = ss.inverse_transform(prediction)
prediction = np.reshape(prediction,len(prediction))

prediction = pd.Series(prediction)
prediction.shape
prediction.plot()
Yhat = Ypred.append(prediction)
Ypred.shape

date = pd.date_range('2016-01', periods=55, freq='M')



Yreel = np.reshape(Yreel,len(Yreel))
Yreel = pd.Series(Yreel)
Yhat = pd.DataFrame(Yhat)
Yhat["Yreel"] = Yreel

Yhat.reset_index(inplace=True)
Yhat.iloc[45:,2:3] = None
del Yhat["index"]
Yhat.columns = ["Ypred","Yreel"]
Yhat.index = date

step = int(Yhat["Ypred"].values.mean()/Yhat["Ypred"].values.std())
time_series_df = pd.DataFrame(Yhat["Ypred"].values)
time_series_df.index = date
smooth_path    = time_series_df.rolling(2).mean()
path_deviation = 1.96 * time_series_df.rolling(2).std()
under_line     = (smooth_path-path_deviation)[0]
over_line      = (smooth_path+path_deviation)[0]

fig , ax = plt.subplots(figsize=(12,6))
ax.plot(Yhat["Yreel"],label='gerçekleşen',color='blue',lw=3)
ax.plot(Yhat["Ypred"],label='öngörülen',color='red')
ax.fill_between(Yhat.index ,under_line, over_line, color='b', alpha=0.2)
plt.title('SMART 12 12 KADEME SMART RÖLE satış tahmin')
plt.ylabel('satışlar')
plt.xlabel('tarih')
plt.legend(loc='upper right')
plt.show();

yıl1 = tt1[:12]
yıl1 = yıl1.reshape(len(yıl1))
yıl1 = pd.Series(yıl1)
yıl2 = tt1[12:25]
yıl2 = yıl2.reshape(len(yıl2))
yıl2 = pd.Series(yıl2)
yıl3 = tt1[24:37]
yıl3 = yıl3.reshape(len(yıl3))
yıl3 = pd.Series(yıl3)
yıl4 = tt1[36:]
yıl4 = yıl4.reshape(len(yıl4))
yıl4 = pd.Series(yıl4)
yıl1.plot()
yıl2.plot()
yıl3.plot()
yıl4.plot();




tt.plot.hist()
