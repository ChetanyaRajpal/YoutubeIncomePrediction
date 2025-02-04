import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import tensorflow as tf
import seaborn as sns 
from AutoClean import AutoClean
sns.set()

raw_data = pd.read_csv('Global Youtube Statistics.csv', encoding = 'unicode_escape')
pd.options.display.max_columns = 9999
print(raw_data.head())

print(raw_data.columns)

preprocesseddata0 = raw_data.copy()
preprocesseddata0 = preprocesseddata0.drop(['rank' , 'Youtuber' , 'Title' , 'Country' , 'Abbreviation' , 'category' , 'video_views_rank','country_rank','channel_type_rank','lowest_monthly_earnings','highest_monthly_earnings','created_month','created_date','Gross tertiary education enrollment (%)','Population','Unemployment rate','Urban_population', 'Latitude', 'Longitude'],axis = 1)

print(preprocesseddata0)

preprocesseddata1 = preprocesseddata0.copy()
print(preprocesseddata1.isna().sum())

# preprocesseddata1['subscribers_for_last_30_days'] = preprocesseddata1['subscribers_for_last_30_days'].map({'nan' : 0})
# print(preprocesseddata1.isna().sum())

# print(preprocesseddata1)
preprocesseddata1['subscribers_for_last_30_days'] = np.where(np.isnan(preprocesseddata1['subscribers_for_last_30_days']), 0 , preprocesseddata1['subscribers_for_last_30_days'])
preprocesseddata1['video_views_for_the_last_30_days'] = np.where(np.isnan(preprocesseddata1['video_views_for_the_last_30_days']), 0 , preprocesseddata1['video_views_for_the_last_30_days'])
preprocesseddata1 = preprocesseddata1.dropna()
print(preprocesseddata1)
print(preprocesseddata1.isna().sum())

preprocesseddata2 = preprocesseddata1.copy()

from statsmodels.stats.outliers_influence import variance_inflation_factor
variables = preprocesseddata2.drop(['channel_type','lowest_yearly_earnings','highest_yearly_earnings'],axis=1)
vif = pd.DataFrame()
vif["VIF"] = [variance_inflation_factor(variables.values, i) for i in range(variables.shape[1])]
vif["features"] = variables.columns
print(vif)

preprocesseddata3 = pd.get_dummies(preprocesseddata2,drop_first=True)
print(preprocesseddata3)


preprocesseddata3['channel_type_Autos'] = preprocesseddata3['channel_type_Autos'].map({False : 0, True : 1})
preprocesseddata3['channel_type_Comedy'] = preprocesseddata3['channel_type_Comedy'].map({False : 0, True : 1})
preprocesseddata3['channel_type_Education'] = preprocesseddata3['channel_type_Education'].map({False : 0, True : 1})
preprocesseddata3['channel_type_Entertainment'] = preprocesseddata3['channel_type_Entertainment'].map({False : 0, True : 1})
preprocesseddata3['channel_type_Film'] = preprocesseddata3['channel_type_Film'].map({False : 0, True : 1})
preprocesseddata3['channel_type_Games'] = preprocesseddata3['channel_type_Games'].map({False : 0, True : 1})
preprocesseddata3['channel_type_Howto'] = preprocesseddata3['channel_type_Howto'].map({False : 0, True : 1})
preprocesseddata3['channel_type_Music'] = preprocesseddata3['channel_type_Music'].map({False : 0, True : 1})
preprocesseddata3['channel_type_News'] = preprocesseddata3['channel_type_News'].map({False : 0, True : 1})
preprocesseddata3['channel_type_Nonprofit'] = preprocesseddata3['channel_type_Nonprofit'].map({False : 0, True : 1})
preprocesseddata3['channel_type_People'] = preprocesseddata3['channel_type_People'].map({False : 0, True : 1})
preprocesseddata3['channel_type_Sports'] = preprocesseddata3['channel_type_Sports'].map({False : 0, True : 1})
preprocesseddata3['channel_type_Tech'] = preprocesseddata3['channel_type_Tech'].map({False : 0, True : 1})
print(preprocesseddata3)

print(preprocesseddata3.isna().sum())

variables = preprocesseddata3.drop(['lowest_yearly_earnings','highest_yearly_earnings','created_year'],axis=1)
vif = pd.DataFrame()
vif["VIF"] = [variance_inflation_factor(variables.values, i) for i in range(variables.shape[1])]
vif["features"] = variables.columns
print(vif)

preprocesseddata4 = preprocesseddata3.drop(['created_year'],axis = 1)
preprocesseddata4 = preprocesseddata4.drop(['video_views_for_the_last_30_days'], axis=1)
preprocesseddata4 = preprocesseddata4.drop(['subscribers_for_last_30_days'],axis = 1)
print(preprocesseddata4.head())

preprocesseddata5 = preprocesseddata4.copy()
scaler = StandardScaler()
scaler.fit(preprocesseddata5)

preprocesseddata6 = scaler.transform(preprocesseddata5)
print(preprocesseddata6)

print(preprocesseddata6.shape)

preprocesseddata6 = pd.DataFrame(preprocesseddata6)
preprocesseddata6.reset_index()
print(preprocesseddata6)

preprocesseddata = preprocesseddata6.copy()

train_data,test_data = train_test_split (preprocesseddata, test_size = 0.2,random_state=365)
print(train_data)
print(test_data)

test_data1,validation_data = train_test_split(test_data, test_size= 0.5 , random_state=365)
print(test_data1)
print(validation_data)

test_data = test_data1
print(train_data.shape)
print(validation_data.shape)
print(test_data.shape)

train_data_output = train_data.loc[:, 4:5]
print(train_data_output)
print(train_data)
train_data_input = train_data.drop([4],axis = 1)
train_data_input = train_data_input.drop([5],axis = 1)
print(train_data_input)

test_data_output = test_data.loc[:,4:5]
test_data_input = test_data.drop([4], axis = 1)
test_data_input = test_data_input.drop([5], axis = 1)
validation_data_output = validation_data.loc[:,4:5]
validation_data_input = validation_data.drop([4], axis = 1)
validation_data_input = validation_data_input.drop([5], axis = 1)
print("Final data sets : ")
print(test_data_input)
print(test_data_output)
print(validation_data_input)
print(validation_data_output)
print(train_data_input)
print(train_data_output)

np.savez('Youtube_data_train', inputs = train_data_input , targets = train_data_output)
np.savez('Youtube_data_test' , inputs = test_data_input, targets = test_data_output)
np.savez('Youtube_data_validation' , inputs = validation_data_input, targets = validation_data_output)

npz = np.load('Youtube_data_train.npz')
train_inputs = npz['inputs'].astype(np.float64)
train_targets = npz['targets'].astype(np.float64)

npz = np.load('Youtube_data_validation.npz')
validation_inputs = npz['inputs'].astype(np.float64)
validation_targets = npz['targets'].astype(np.float64)

npz = np.load('Youtube_data_test.npz')
test_inputs = npz['inputs'].astype(np.float64)
test_targets = npz['targets'].astype(np.float64)

input_size = 16
output_size = 2
hidden_layer_size = 34

model = tf.keras.Sequential([
    tf.keras.layers.Dense(units= hidden_layer_size , activation='linear'),
    tf.keras.layers.Dense(hidden_layer_size, activation='relu'),
    tf.keras.layers.Dense(hidden_layer_size, activation='relu'),
    tf.keras.layers.Dense(hidden_layer_size, activation='relu'),
    tf.keras.layers.Dense(hidden_layer_size, activation='relu'),
    tf.keras.layers.Dense(hidden_layer_size, activation='relu'),
    tf.keras.layers.Dense(hidden_layer_size, activation='relu'),
    tf.keras.layers.Dense(hidden_layer_size, activation='relu'),
    tf.keras.layers.Dense(hidden_layer_size, activation='relu'),
    tf.keras.layers.Dense(hidden_layer_size, activation='relu'),
    tf.keras.layers.Dense(hidden_layer_size, activation='relu'),
    tf.keras.layers.Dense(hidden_layer_size, activation='relu'),
    tf.keras.layers.Dense(hidden_layer_size, activation='relu'),
    tf.keras.layers.Dense(hidden_layer_size, activation='linear'),
    tf.keras.layers.Dense(hidden_layer_size, activation='linear'),
    tf.keras.layers.Dense(hidden_layer_size, activation='linear'),
    tf.keras.layers.Dense(output_size, activation='linear')
    
])

model.compile(optimizer='adam', loss = 'mean_absolute_error', metrics=['accuracy'])
num_epochs = 100
batch_size = 100

early_stopping = tf.keras.callbacks.EarlyStopping(patience=2)

model.fit(train_inputs,
          train_targets,
          batch_size=batch_size, 
          epochs = num_epochs,
          callbacks = (early_stopping),
          validation_data=(validation_inputs,validation_targets),
          verbose = 2)

test_loss,test_accuracy = model.evaluate(test_inputs, test_targets)
print(test_loss)
print(test_accuracy)
