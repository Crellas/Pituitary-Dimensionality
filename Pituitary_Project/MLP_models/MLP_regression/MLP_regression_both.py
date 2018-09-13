import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pylab as plt
import matplotlib.patches as patches
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import numpy as np

## Import data to a pandas dataframe
display_label = "GSK"
id_labels = {"GK": 0,
             "GCAL": 1,
             "GSK": 2}
label_id = id_labels[display_label]

data_file = '/Users/jamesashford/Documents/NatSci Exeter/2017-18 (Third Year)/Summer Work/Pituitary Dimensionality Project - J. Tabak/Neural Network/SavedTXTs/10100_iterations_3var_4labels.txt'
#Read data from file
data = pd.read_csv(data_file, header=0)
#Read headers from file
column_headers = data.columns.values
#Remove silent data
data = data[data["TYPE"] != 0]

# Get the feature and label data
features = data.drop(["GK", "GCAL", "GSK"], axis=1)
labels = data[["GK", "GCAL", "GSK"]]
# Split into training and testing data
features_train, features_test, labels_train, labels_test = train_test_split(features, labels)

## Train a scaler for the data
scaler = StandardScaler()
scaler.fit(features_train)
# Rescale the feature data
features_train = scaler.transform(features_train)
features_test = scaler.transform(features_test)
features_all = scaler.transform(features)

## Training the model
mlp = MLPRegressor(hidden_layer_sizes=(20, 20), max_iter=2000)
mlp.fit(features_train, labels_train)

## Predictions and evaluation
pred_test = mlp.predict(features_test)
pred_train = mlp.predict(features_train)
pred_full = mlp.predict(features_all)

# Test MSE and Variance
print("Mean Squared Error (Test): {} ".format(mean_squared_error(labels_test, pred_test)))
print("Variance Score (Test): {}".format(r2_score(labels_test, pred_test)))
print('')
print("Mean Squared Error (Train): {} ".format(mean_squared_error(labels_train, pred_train)))
print("Variance Score (Train): {}".format(r2_score(labels_train, pred_train)))
print('')
print("Mean Squared Error (All): {} ".format(mean_squared_error(labels, pred_full)))
print("Variance Score (All): {}".format(r2_score(labels, pred_full)))

# Plot all data against predictions w/ labels
colours = {0: 'b',
           1: 'r',
           2: 'g',
           3: 'y'}
silent_patch = patches.Patch(color='b', label='Silent')
spiking_patch = patches.Patch(color='r', label='Spiking')
one_spike_burst_patch = patches.Patch(color='y', label='One-Spike Bursting')
bursting_patch = patches.Patch(color='g', label='Bursting')

plt.figure()
plt.plot(np.linspace(0, 10, 11), color='black')
col_full = [colours[val] for val in features["TYPE"]]
plt.scatter(labels, pred_full, color=col_full)
plt.legend(handles=[silent_patch, spiking_patch, bursting_patch, one_spike_burst_patch])

plt.title("Value Prediction of {} using MLP Regression trained on All Labels".format(display_label))
plt.xlabel("Actual Label Value")
plt.ylabel("Predicted Label Value")
plt.show()
