from numpy.random import seed
seed(1)
from tensorflow.keras.utils import set_random_seed
set_random_seed(2)
import os
import matplotlib
matplotlib.use('qt5agg')
import matplotlib.pyplot as plt
import scipy
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, LSTM, TimeDistributed, Dropout
from scikeras.wrappers import KerasRegressor

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
import shap
from matplotlib.dates import DateFormatter


# Single step dataset preparation
def singleStepSampler(df, window):
    xRes = []
    yRes = []
    for i in range(0, len(df) - window):
        res = []
        for j in range(0, window):
            r = []
            for col in df.columns:
                r.append(df[col][i + j])
            res.append(r)
        xRes.append(res)
        yRes.append(df[['SPM (g m-3)']].iloc[i + window].values)
    return np.array(xRes), np.array(yRes)

###########################################
##### load SPM at river mouth (y) #####
###########################################
DW_insitu = scipy.io.loadmat(f"./imported_datasets/insitu/SPM/ElwhaGF20132014.mat")
time_insitu = np.concatenate(DW_insitu['aqd']['td'][0][0]).astype(float)
SPM_insitu = np.concatenate(DW_insitu['aqd']['obs'][0][0]*1000).astype(float) #convert to g m-3
time_insitu_list = []
time_insitu_date = []
for i in np.arange(0, len(time_insitu), 1):
    tmp1 = timedelta(days=time_insitu[i]) + datetime(1,1,1) - relativedelta(years=1)
    time_insitu_list.append(tmp1)
time_insitu_array = np.array(time_insitu_list)
DW_SPM = pd.DataFrame(
    {'Time (UTC)': time_insitu_array,
     'SPM (g m-3)': SPM_insitu,
    })
DW_SPM['Time (UTC)'] = pd.to_datetime(DW_SPM['Time (UTC)'])
DW_SPM = DW_SPM.set_index('Time (UTC)')
DW_SPM_daily = DW_SPM.resample('D').mean()
DW_SPM_daily.drop(DW_SPM_daily.loc[DW_SPM_daily.index > datetime(2014,3,7)].index, inplace=True)

###########################################
######## load discharge data upstream ########
###########################################
discharge = pd.read_csv(f"./imported_datasets/insitu/discharge/USGS_12045500_discharge.txt", sep='\t', skiprows = 26, low_memory=False)
discharge = discharge.iloc[1:]
discharge = discharge[['datetime','150691_00060']]
discharge = discharge.rename(columns={"150691_00060" : "discharge"})
discharge['datetime'] = pd.to_datetime(discharge['datetime'])
discharge = discharge.set_index('datetime')
discharge = discharge.astype(np.float64)
discharge_daily = discharge.resample('D').mean()

###########################################
######## load turbidity data upstream ########
###########################################
turbidity = pd.read_csv(f"./imported_datasets/insitu/SPM/USGS_12046260_turbidity.txt", sep='\t', skiprows = 27, low_memory=False)
turbidity = turbidity.iloc[1:]
turbidity = turbidity[['datetime','227925_63680']]
turbidity = turbidity.rename(columns={"227925_63680" : "turbidity"})
turbidity['datetime'] = pd.to_datetime(turbidity['datetime'])
turbidity = turbidity.set_index('datetime')
turbidity = turbidity.astype(np.float64)
turbidity_daily = turbidity.resample('D').mean()

###########################################
######## load meteo data airport ########
###########################################
meteo = pd.read_csv(f"./imported_datasets/insitu/meteo/WA_ASOS_CLM_PortAngelesAirport.csv", low_memory=False)
meteo['valid'] = pd.to_datetime(meteo['valid'])
meteo = meteo.drop(columns=['station',
                            'skyc1',
                            'skyc2',
                            'skyc3',
                            'skyc4',
                            'metar',
                            'wxcodes',
                            'peak_wind_time',
                            'p01i',
                            'ice_accretion_1hr',
                            'ice_accretion_3hr',
                            'ice_accretion_6hr',
                            ], axis=1, inplace=False)
meteo[meteo=='nan'] = np.nan
meteo = meteo.set_index('valid')
meteo.index = pd.to_datetime(meteo.index)
meteo_daily = meteo.resample('D').mean()

###########################################
######## load wave data buoy ########
###########################################
waves = pd.read_csv(f"./imported_datasets/insitu/waves/46087-Generic_Export-20250611T12T12_35.csv", low_memory=False)
waves['time'] = pd.to_datetime(waves['time'])
waves = waves.set_index('time')
waves_daily = waves.resample('D').mean()

###########################################
########   load water level data   ########
###########################################
directory = './insitu/water_level/'
wl = pd.DataFrame()
for file in os.listdir(directory):
    filename = os.fsdecode(file)
    tmp = pd.read_csv(directory+filename, sep='\t', skiprows = 9, low_memory=False)
    tmp.index = pd.to_datetime(tmp.index)
    tmp = tmp.rename(columns={"// datetime [ISO8601], waterlevel_unassessed [m]": "water_level"})
    wl = wl._append(tmp)
wl = wl.sort_index()
wl.index = pd.to_datetime(wl.index)
wl.index = wl.index.tz_localize(None)
wl_daily = wl.resample('D').mean()

###########################################
######## load insitu data upstream ########
###########################################
UP_SPM_daily = pd.read_csv(f"./imported_datasets/insitu/SPM/Elwha_DailySedimentLoads_2011to2016.csv")
UP_SPM_daily['Day'] = pd.to_datetime(UP_SPM_daily['Day']).dt.date
UP_SPM_daily = UP_SPM_daily.set_index('Day')
UP_SPM_daily.index = pd.to_datetime(UP_SPM_daily.index)

###########################################
######## interpolate on SPM timestamps ########
###########################################
df = DW_SPM_daily._append([waves_daily,
                     discharge_daily,
                     turbidity_daily,
                     meteo_daily,
                     wl_daily,
                     UP_SPM_daily], sort=True)
df = df.sort_index()
df_interpolate = df.interpolate(method = 'time', limit_direction='backward', inplace=False)
# retieve values for original DW_SPM dates
df_final = df_interpolate.loc[DW_SPM_daily.index]
#check for missing value
print(df_final.isnull().sum())
df_final = df_final.drop(columns=['Project year',
                                  'Release period',
                                  'Remarks',
                                  'latitude',
                                  'longitude',
                                  'snowdepth',
                                  'skyl4',
                                  'waveSensorOutput',
                                  'waveSysSensor',
                                  'latitude',
                                  'longitude',
                                  'waveFrequency_47',
                                  'depth',
                                  'Daily Total gauged > 2-mm bedload (tonnes)',
                                  'Daily gauged bedload for 2-16mm particles (tonnes )',
                                  'Daily gauged bedload for >16mm particles (tonnes )',
                                  'Estimated daily ungauged bedload (tonnes)'], axis=1, inplace=False)
df = df_final.dropna()
print(df.isnull().sum())
df = df[~df.index.duplicated(keep='first')]  # keep='first' or 'last' or False

#check correlation matrix
correlation_matrix = df.corr()
print(correlation_matrix['SPM (g m-3)'])
df = df.drop(columns=['Ave fraction fines (based on two turbidimeters)',
                      'gust',
                      'relh',
                      'skyl1',
                      'skyl2',
                      'skyl3',
                      'waveHs',
                      'waveTp'], axis=1, inplace=False)

imputer = SimpleImputer(missing_values=np.nan)  # Handling missing values
dataFrame = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
dataFrame = dataFrame.reset_index(drop=True)
# Applying feature scaling
scaler = MinMaxScaler(feature_range=(0, 1))
df_scaled = scaler.fit_transform(dataFrame.to_numpy())
df_scaled = pd.DataFrame(df_scaled, columns=list(dataFrame.columns))
target_scaler = MinMaxScaler(feature_range=(0, 1))
df_scaled[['SPM (g m-3)']] = target_scaler.fit_transform(dataFrame[['SPM (g m-3)']].to_numpy())
df_scaled = df_scaled.astype(float)

#select features
selected_features = df_scaled.columns.values
selected_features = np.delete(selected_features, 6)
X = df_scaled[selected_features]
y = df_scaled['SPM (g m-3)']

# Dataset splitting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # 0.2

# # create model
def create_model():
    model = Sequential()  # Define model architecture
    model.add(Dense(128, input_shape=(30,), kernel_initializer='normal', activation='relu'))  # 128
    model.add(Dropout(0.2))
    model.add(Dense(1, kernel_initializer='normal', activation='linear'))
    model.compile(loss='mean_absolute_error', metrics=['r2_score','mae'])
    return model
model = KerasRegressor(model=create_model, verbose=2)
# Define hyperparameter distributions
param_dist = {
    'optimizer': ['adam', 'sgd', 'rmsprop', 'nadam'],
    'epochs': [30, 60, 90, 120, 150, 180, 210, 240],
    'batch_size': [10, 20, 40, 60, 80, 100]
}

# Perform GridSearchCV
grid = GridSearchCV(estimator=model, param_grid=param_dist, cv=3)
grid_result = grid.fit(X_train, y_train)

result = grid_result
# summarize results
print("Best: %f using %s" % (result.best_score_, result.best_params_))
means = result.cv_results_['mean_test_score']
stds = result.cv_results_['std_test_score']
params = result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
#Best: 0.223838 using {'batch_size': 10, 'epochs': 150, 'optimizer': 'adam'}

model = Sequential()  # Define model architecture
model.add(Dense(128, input_shape=(30,), kernel_initializer='normal', activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1, kernel_initializer='normal', activation='linear'))
model.compile(loss='mean_absolute_error', metrics=['r2_score', 'mae'],optimizer='adam')
history = model.fit(X_train, y_train, epochs=150, verbose = 0, batch_size=10)

# Forecast Plot with Dates on X-axis
predicted_values = model.predict(X_test, verbose = 0)
# invert predictions
testPredict = target_scaler.inverse_transform(predicted_values)
testY = target_scaler.inverse_transform([y_test])

d = {
    'Predicted_SPM': testPredict[:, 0],
    'Actual_SPM': testY[0, :],
}

# Create a SHAP explainer
explainer = shap.KernelExplainer(model.predict, X_train)
# Get SHAP values for a sample of your data
shap_values = explainer.shap_values(X_train)
# Average absolute SHAP values across samples to get feature importances
importances = np.mean(np.abs(shap_values), axis=0)
# Pair feature names with their importances
feature_importance = dict(zip(selected_features, importances))
# Sort by importance
feature_importance = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))
print(feature_importance)

d = pd.DataFrame(d)
d.index = df.index[-len(y_test):]  # Assigning the correct date index

# Model Evaluation
def eval(model):
    return {
        'MSE': mean_squared_error(d[f'Actual_{model.split("_")[1]}'].to_numpy(), d[model].to_numpy()),
        'MAE': mean_absolute_error(d[f'Actual_{model.split("_")[1]}'].to_numpy(), d[model].to_numpy()),
        'R2': r2_score(d[f'Actual_{model.split("_")[1]}'].to_numpy(), d[model].to_numpy())
    }

result = dict()

for item in ['Predicted_SPM']:
    result[item] = eval(item)

print(result)

# create X for predictions
new_df = df_interpolate[~df_interpolate.index.duplicated(keep='first')]  # keep='first' or 'last' or False
new_df = new_df.drop(columns=['Project year',
                                  'Release period',
                                  'Remarks',
                                  'latitude',
                                  'longitude',
                                  'snowdepth',
                                  'skyl4',
                                  'waveSensorOutput',
                                  'waveSysSensor',
                                  'latitude',
                                  'longitude',
                                  'waveFrequency_47',
                                  'depth',
                                  'Daily Total gauged > 2-mm bedload (tonnes)',
                                  'Daily gauged bedload for 2-16mm particles (tonnes )',
                                  'Daily gauged bedload for >16mm particles (tonnes )',
                                  'Ave fraction fines (based on two turbidimeters)',
                                  'Estimated daily ungauged bedload (tonnes)',
                                  'gust',
                                  'relh',
                                  'skyl1',
                                  'skyl2',
                                  'skyl3',
                                  'waveHs',
                                  'waveTp'
                                  ], axis=1, inplace=False)
new_df_final = new_df[~new_df.index.duplicated(keep='first')]  # keep='first' or 'last' or False

# Applying feature scaling
scaler = MinMaxScaler(feature_range=(0, 1))
df_scaled = scaler.fit_transform(new_df_final.to_numpy())
df_scaled = pd.DataFrame(df_scaled, columns=list(new_df_final.columns))
X2 = df_scaled[selected_features]

# Forecast Plot with Dates on X-axis
predicted_values = model.predict(X2, verbose = 0)
# invert predictions
finalPredict = target_scaler.inverse_transform(predicted_values)
finalPredict = pd.DataFrame(finalPredict)
finalPredict.index = pd.to_datetime(new_df_final.index)
finalPredict[finalPredict < 0] = 0

idx = df.index
finalPredict_sub = finalPredict.loc[idx]

# ==================================================
# CALCULATE RMSE and R²
# ==================================================
rmse = np.sqrt(mean_squared_error(df['SPM (g m-3)'], finalPredict_sub))
r2 = r2_score(df['SPM (g m-3)'], finalPredict_sub)
# ==================================================
# START PLOTTING
# ==================================================
fig, axes = plt.subplots(2, 2, figsize=(12, 9))

my_date_format = DateFormatter("%m-%Y")
# ---- 1) Time Series ----
axes[0,0].axvspan(df.index[0], df.index[len(df)-1], facecolor='lightgreen', alpha=0.5, label='Training-Test',zorder = 0)
axes[0,0].plot(df['SPM (g m-3)'], label='Observed', color='steelblue', alpha=0.7,zorder=2)
axes[0,0].plot(finalPredict, label='Hindcast', color='darkorange', alpha=0.7,zorder=1)
axes[0,0].set_xlabel('Time',font='Arial',
               size=12)
axes[0,0].set_ylabel('Suspended Sediment (g m$^{-3}$)',font='Arial',
               size=12)
axes[0,0].set_title('Observed vs. Hindcasted Time Series',font='Arial',
               size=16)
axes[0,0].legend()
axes[0,0].axis(xmin=pd.to_datetime('2011-09-15 00:00:00'),xmax=pd.to_datetime('2016-09-20 00:00:00'),)
axes[0,0].xaxis.set_major_formatter(my_date_format)
axes[0,0].grid(True)

# ---- 1) Time Series ----
my_date_format = DateFormatter("%d-%m")
axes[1,0].axvspan(df.index[0], df.index[len(df)-1], facecolor='lightgreen', alpha=0.5,zorder = 0)
axes[1,0].plot(df['SPM (g m-3)'], label='Observed', color='steelblue', alpha=0.7)
axes[1,0].plot(finalPredict, label='Hindcast', color='darkorange', alpha=0.7)
axes[1,0].set_xlabel('Time',font='Arial',
               size=12)
axes[1,0].set_ylabel('Suspended Sediment (g m$^{-3}$)',font='Arial',
               size=12)
axes[1,0].set_title('Observed vs. Hindcasted Time Series',font='Arial',
               size=16)
axes[1,0].legend()
axes[1,0].axis(xmin=df.index[0],xmax=df.index[len(df)-1])
axes[1,0].xaxis.set_major_formatter(my_date_format)
axes[1,0].grid(True)

# ---- 2) Scatter Plot ----
axes[0,1].scatter(df['SPM (g m-3)'], finalPredict_sub, color='green', alpha=0.5, edgecolors='black')
min_val = min(min(df['SPM (g m-3)']), min(finalPredict))
max_val = max(max(df['SPM (g m-3)']), max(finalPredict))
axes[0,1].plot([min_val, max_val], [min_val, max_val], 'k--')
axes[0,1].set_xlabel('Observed',font='Arial',
               size=12)
axes[0,1].set_ylabel('Hindcast',font='Arial',
               size=12)
axes[0,1].set_title('Observed vs. Hindcast',font='Arial',
               size=16)
axes[0,1].legend()
axes[0,1].grid(True)

# Add RMSE and R² annotation
axes[0,1].text(0.05, 0.9,
             f'RMSE: {rmse:.2f}\nR²: {r2:.2f}',
             transform=axes[0,1].transAxes,
             fontsize=10,
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))

# ---- 3) Feature Importance ----
labels = list(feature_importance.keys())
values = list(feature_importance.values())
values = np.concatenate(values)
axes[1,1].barh(labels, values, color='teal', alpha=0.8)
axes[1,1].set_xlabel('Relative Importance',font='Arial',
               size=12)
axes[1,1].set_title('Feature Importance',font='Arial',
               size=16)
axes[1,1].invert_yaxis()
axes[1,1].grid(True, axis='x')
axes[1,1].yaxis.set_label_position("right")       # Labels on the right
axes[1,1].yaxis.tick_right()

# ==================================================
# FINAL TOUCHES
# ==================================================
plt.tight_layout()
plt.savefig('FULL_analysis2.png')
plt.show()