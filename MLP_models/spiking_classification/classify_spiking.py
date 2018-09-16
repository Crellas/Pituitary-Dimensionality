import matplotlib, re
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import numpy as np
from io import StringIO
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix

## Import data to a pandas dataframe
data_file = '/Users/jamesashford/Documents/NatSci Exeter/2017-18 (Third Year)/Summer Work/Pituitary Dimensionality Project - J. Tabak/Neural Network/SavedTXTs/10000_iterations_3var_NewMethod.txt'
#Read data and headers from file
data = pd.read_csv(data_file, header=0)
column_headers = data.columns.values
# Get the feature and label data
features = data[["GK", "GCAL", "GSK"]]
labels = data["TYPE"]
# Split into training and testing data
features_train, features_test, labels_train, labels_test = train_test_split(features, labels)

## Pre-process the data by scaling
scaler = StandardScaler()
# Fit to the training feature data only
scaler.fit(features_train)
# Rescale the feature data
features_train = scaler.transform(features_train)
features_test = scaler.transform(features_test)

## Training the model
mlp = MLPClassifier(hidden_layer_sizes=(20, 20), max_iter=2000, learning_rate='adaptive')
mlp.fit(features_train, labels_train)

## Predict for full set
predict_full = mlp.predict(scaler.transform(features))
print(confusion_matrix(labels, predict_full))
report_full = classification_report(labels, predict_full)
print(report_full)

## Predict for test set
predict_test = mlp.predict(features_test)
print(confusion_matrix(labels_test, predict_test))
report_test = classification_report(labels_test, predict_test)
print(report_test)

## Predict for train set
predict_train = mlp.predict(features_train)
print(confusion_matrix(labels_train, predict_train))
report_train = classification_report(labels_train, predict_train)
print(report_train)

## Plot the outputs, one subplot for the original model and one for the classifier outputs
# Generate dictionary and patches for the colour legends
colours = {0: 'b',
           1: 'r',
           2: 'g',
           3: 'y'}
silent_patch = patches.Patch(color='b', label='Silent')
spiking_patch = patches.Patch(color='r', label='Spiking')
one_spike_burst_patch = patches.Patch(color='y', label='One-Spike Bursting')
bursting_patch = patches.Patch(color='g', label='Bursting')
downsample = 2


# Basic 2D Plotting
# fig_2d = plt.figure()
# ax1 = fig_2d.add_subplot(121)
# col_model = [colours[num] for num in labels.values]
# ax1.scatter(features.values[:, 0][::downsample], features.values[:, 1][::downsample], c=col_model[::downsample])
# ax1.set_title("Behaviour in Original Data")
# ax1.set_xlabel("GK"); ax1.set_ylabel("GCAL")
# plt.legend(handles=[silent_patch, spiking_patch, bursting_patch])
#
# ax2 = fig_2d.add_subplot(122)
# col_classifier = [colours[num] for num in predict]
# ax2.scatter(features.values[:, 0][::downsample], features.values[:, 1][::downsample], c=col_classifier[::downsample])
# ax2.set_title("Behaviour in Classifier Model")
# ax2.set_xlabel("GK"); ax2.set_ylabel("GCAL")

# 2D Plotting in a Data Range
# fig_2d_range = plt.figure()
# col_model, col_classifier, near_xvals = [], [], []
# for val in range(len(labels.values)):
#     if features.values[val, 2] < 1:
#         col_model.append(colours[labels.values[val]])
#         col_classifier.append(colours[predict[val]])
#         near_xvals.append(features.values[val, :])
# near_xvals = np.array(near_xvals)
#
# ax1 = fig_2d_range.add_subplot(121)
# ax1.scatter(near_xvals[:, 0][::downsample], near_xvals[:, 1][::downsample], c=col_model[::downsample])
# ax1.set_title("Behaviour in Original Data")
# ax1.set_xlabel("GK"); ax1.set_ylabel("GCAL")
# plt.legend(handles=[silent_patch, spiking_patch, bursting_patch, one_spike_burst_patch])
#
# ax2 = fig_2d_range.add_subplot(122)
# ax2.scatter(near_xvals[:, 0][::downsample], near_xvals[:, 1][::downsample], c=col_classifier[::downsample])
# ax2.set_title("Behaviour in Classifier Model")
# ax2.set_xlabel("GK"); ax2.set_ylabel("GCAL")


# 3D Plotting
fig_3d = plt.figure()
ax1 = fig_3d.add_subplot(121, projection='3d')
col_model = [colours[num] for num in labels.values]
ax1.scatter(features.values[:, 0][::downsample],  features.values[:, 1][::downsample], features.values[:, 2][::downsample], c=col_model[::downsample])
ax1.set_title("Behaviour in Original Data")
ax1.set_xlabel("GK"); ax1.set_ylabel("GCAL"); ax1.set_zlabel("GSK")
# plt.legend(handles=[silent_patch, spiking_patch, bursting_patch, one_spike_burst_patch])

ax2 = fig_3d.add_subplot(122, projection='3d')
col_classifier = [colours[num] for num in predict_full]
ax2.scatter(features.values[:, 0][::downsample], features.values[:, 1][::downsample], features.values[:, 2][::downsample], c=col_classifier[::downsample])
ax2.set_title("Behaviour in Classifier Model")
ax2.set_xlabel("GK"); ax2.set_ylabel("GCAL"); ax2.set_zlabel("GSK")


# Visualise each type individually
fig_each_type = plt.figure()
axes = [fig_each_type.add_subplot(221, projection='3d'), fig_each_type.add_subplot(222,projection='3d'),
        fig_each_type.add_subplot(223, projection='3d'), fig_each_type.add_subplot(224, projection='3d')]

ax_titles = ['Silent', 'Spiking', 'Bursting', 'One-Spike Bursting']
for i in range(4):
    col_model, xvals = [], []
    for val in range(len(labels.values)):
        if labels.values[val] == i:
            col_model.append(colours[i])
            xvals.append(features.values[val, :])
    xvals = np.array(xvals)
    axes[i].scatter(xvals[:, 0][::downsample], xvals[:, 1][::downsample], xvals[:, 2][::downsample], c=col_model[::downsample])
    axes[i].set_title(ax_titles[i])
    axes[i].set_xlabel("GK"); axes[i].set_ylabel("GCAL"); axes[i].set_zlabel("GSK")
    axes[i].set_xlim((0, 10))
    axes[i].set_ylim((0, 5))
    axes[i].set_zlim((0, 5))

def on_move(event):
    all_axes = fig_each_type.get_axes()
    this_axis = event.inaxes
    for axis in all_axes:
        axis.view_init(elev=this_axis.elev, azim=this_axis.azim)
        fig_each_type.canvas.draw_idle()

def on_move_class(event):
    all_axes = fig_3d.get_axes()
    this_axis = event.inaxes
    for axis in all_axes:
        axis.view_init(elev=this_axis.elev, azim=this_axis.azim)
        fig_each_type.canvas.draw_idle()

c1 = fig_each_type.canvas.mpl_connect('motion_notify_event', on_move)
c2 = fig_3d.canvas.mpl_connect('motion_notify_event', on_move_class)


plt.show()

