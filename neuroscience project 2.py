


# In[2]:


import numpy as np 
import pandas as pd 
import os
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
from plotly.offline import init_notebook_mode, iplot, plot
init_notebook_mode(connected=True)
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.regularizers import l2, l1
from keras.metrics import BinaryAccuracy
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import mean_squared_error
from keras.layers import Dense, SimpleRNN
from keras.layers import GRU, LSTM


# In[3]:


datatest = pd.read_csv("datatest.txt")
datatest2 = pd.read_csv("datatest2.txt")
datatraining = pd.read_csv("datatraining.txt")


# In[4]:


print("Veri seti boyutu:", datatest.shape)
print("Her bir veri noktasının özellik sayısı:", datatest.shape[1])  # İkinci boyut özellik sayısını verir
print("Zaman adımlarının sayısı:", datatest.shape[0])  # İlk boyut zaman adımlarının sayısını verir


# In[5]:


print(datatest.info())
datatest.head()


# In[6]:


print(datatest2.info())
datatest2.head()


# In[7]:


print(datatraining.info())
datatraining.head()


# In[8]:


datatest['date'] = pd.to_datetime(datatest['date'])
datatest2['date'] = pd.to_datetime(datatest2['date'])
datatraining['date'] = pd.to_datetime(datatraining['date'])
datatest.reset_index(drop=True, inplace=True)
datatest2.reset_index(drop=True, inplace=True)
datatraining.reset_index(drop=True, inplace=True)


# In[9]:


datatraining.describe()


# In[10]:


scaler = MinMaxScaler()
columns = ['Temperature', 'Humidity', 'Light', 'CO2', 'HumidityRatio']
scaler.fit(np.array(datatraining[columns]))
datatest[columns] = scaler.transform(np.array(datatest[columns]))
datatest2[columns] = scaler.transform(np.array(datatest2[columns]))
datatraining[columns] = scaler.transform(np.array(datatraining[columns]))


# In[11]:


plt.figure(figsize=(10,10))
plt.title('Box Plot for Features', fontdict={'fontsize':18})
ax = sns.boxplot(data=datatraining.drop(['date', 'Occupancy'],axis=1), orient="h", palette="Set2")
print(datatraining.drop(['date', 'Occupancy'],axis=1).describe())


# In[12]:


plt.figure(figsize=(10,8))
plt.title('Correlation Table for Features', fontdict={'fontsize':18})
ax = sns.heatmap(datatraining.corr(), annot=True, linewidths=.2)


# In[13]:


data = datatraining.copy()
data.Occupancy = data.Occupancy.astype(str)
fig = px.scatter_3d(data, x='Temperature', y='Humidity', z='CO2', size='Light', color='Occupancy', color_discrete_map={'1':'red', '0':'blue'})
fig.update_layout(scene_zaxis_type="log", title={'text': "Features and Occupancy",
                                                'y':0.9,
                                                'x':0.5,
                                                'xanchor': 'center',
                                                'yanchor': 'top'})
iplot(fig)


# In[14]:


sns.set(style="darkgrid")
plt.title("Occupancy Distribution", fontdict={'fontsize':18})
ax = sns.countplot(x="Occupancy", data=datatraining)


# In[15]:


hours_1 = []
hours_0 = []
for date in datatraining[datatraining['Occupancy'] == 1]['date']:
    hours_1.append(date.hour)
for date in datatraining[datatraining['Occupancy'] == 0]['date']:
    hours_0.append(date.hour)


# In[16]:


plt.figure(figsize=(8,8))
ax = sns.displot(hours_1)
ax = sns.displot(hours_0)


# In[17]:


datatest['period_of_day'] = [1 if (i.hour >= 7 and i.hour <= 17) else 0 for i in datatest['date']]
datatest2['period_of_day'] = [1 if (i.hour >= 7 and i.hour <= 17) else 0 for i in datatest2['date']]
datatraining['period_of_day'] = [1 if (i.hour >= 7 and i.hour <= 17) else 0 for i in datatraining['date']]
datatraining.sample(10)


# In[18]:


X_train = datatraining.drop(columns=['date', 'Occupancy'], axis=1)
y_train = datatraining['Occupancy']
X_validation = datatest.drop(columns=['date', 'Occupancy'], axis=1)
y_validation = datatest['Occupancy']
X_test = datatest2.drop(columns=['date', 'Occupancy'], axis=1)
y_test = datatest2['Occupancy']


# In[19]:


n_neighbors_list = [7,15,45,135]
weights_list = ['uniform', 'distance']
metric_list = ['euclidean', 'manhattan']
accuracies = {}
for n in n_neighbors_list:
    for weight in weights_list:
        for metric in metric_list:
            knn_model = KNeighborsClassifier(n_neighbors=n, weights=weight, metric=metric)
            knn_model.fit(X_train, y_train)
            accuracy = knn_model.score(X_validation, y_validation)
            accuracies[str(n)+"/"+weight+"/"+metric] = accuracy


# In[20]:


plotdata = pd.DataFrame()
plotdata['Parameters'] = accuracies.keys()
plotdata['Accuracy'] = accuracies.values()
fig = px.line(plotdata, x="Parameters", y="Accuracy")
fig.update_layout(title={'text': "Accuracies for Different Hyper-Parameters",
                                                'x':0.5,
                                                'xanchor': 'center',
                                                'yanchor': 'top'})
iplot(fig)


# In[21]:


knn_model = KNeighborsClassifier(n_neighbors=135)
knn_model.fit(X_train, y_train)
y_pred = knn_model.predict(X_validation)
plt.title("KNN Confusion Matrix for Validation Data", fontdict={'fontsize':18})
ax = sns.heatmap(confusion_matrix(y_validation, y_pred), annot=True, fmt="d")


# In[22]:


svm_model = SVC()
svm_model.fit(X_train, y_train)
print("Accuracy for SVM on validation data: {}%".format(round((svm_model.score(X_validation, y_validation)*100),2)))


# In[23]:


y_pred = svm_model.predict(X_validation)
plt.title("SVM Confusion Matrix for Validation Data", fontdict={'fontsize':18})
ax = sns.heatmap(confusion_matrix(y_validation, y_pred), annot=True, fmt="d")


# In[24]:


data = [[i for i in range(100)]]
data = np.array(data, dtype=float)
target = [[i for i in range(1, 101)]]
target = np.array(target, dtype=float)


# In[25]:


data = data.reshape((1, 1, 100))
target = target.reshape((1, 1, 100))


# In[26]:


model = Sequential()
model.add(LSTM(50, input_shape=(1, 100), return_sequences=True))
model.add(Dense(100))
model.compile(loss='mean_absolute_error', optimizer='adam')


# In[27]:


model.fit(data, target, epochs=10000, batch_size=1, verbose=1)


# In[28]:


test_data = np.array([[i for i in range(100, 200)]], dtype=float)
test_data = test_data.reshape((1, 1, 100))
prediction = model.predict(test_data)
print(prediction)


# In[29]:


data = [[i for i in range(100)]]
data = np.array(data, dtype=float)
target = [[i for i in range(1, 101)]]
target = np.array(target, dtype=float)


# In[30]:


data = data.reshape((1, 1, 100))
target = target.reshape((1, 1, 100))


# In[31]:


model = Sequential()
model.add(GRU(50, input_shape=(1, 100), return_sequences=True))
model.add(Dense(100))
model.compile(loss='mean_absolute_error', optimizer='adam')


# In[32]:


model.fit(data, target, epochs=10000, batch_size=1, verbose=1)


# In[33]:


test_data = np.array([[i for i in range(100, 200)]], dtype=float)
test_data = test_data.reshape((1, 1, 100))
prediction = model.predict(test_data)
print(prediction)

