import itertools
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pylab as plt
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd

def plot_features(point_data, feature_list, colour_list):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(point_data[[feature_list[0]]], point_data[[feature_list[1]]], point_data[[feature_list[2]]],
               c=colour_list)
    ax.set_xlabel(feature_list[0])
    ax.set_ylabel(feature_list[1])
    ax.set_zlabel(feature_list[2])
    ax.legend(handles=[silent_patch, spiking_patch, bursting_patch, one_spike_burst_patch])

def plot_subplot(point_data, feature_list, colour_list, figure, subplot, subplot_shape, project_3d=True):
    if project_3d:
        ax = figure.add_subplot(subplot_shape[0], subplot_shape[1], subplot, projection='3d')
        ax.scatter(point_data[[feature_list[0]]], point_data[[feature_list[1]]], point_data[[feature_list[2]]],
                   c=colour_list)
        ax.set_zlabel(feature_list[2])
    else:
        ax = figure.add_subplot(subplot_shape[0], subplot_shape[1], subplot)
        ax.scatter(point_data[[feature_list[0]]], point_data[[feature_list[1]]], c=colour_list)
    ax.set_xlabel(feature_list[0])
    ax.set_ylabel(feature_list[1])
    # ax.legend(handles=[silent_patch, spiking_patch, bursting_patch])

## Import data to a pandas dataframe
data_file = '/Users/jamesashford/Documents/NatSci Exeter/2017-18 (Third Year)/Summer Work/Pituitary Dimensionality Project - J. Tabak/Modelling/pituitary_model/list_model/SavedTXT/start:0,0,0_end:10,5,5_time:13:44:0.txt'
#Read data from file
data = pd.read_csv(data_file, header=0)
#Read data from file
labels = data[["GK", "GCAL", "GSK"]]
features = data.drop(["GK", "GCAL", "GSK", "TYPE", "EVENT_AMPLITUDE", "EVENT_BURSTPEAKS"], axis=1)
feature_list = np.array(features.columns.values)
print(feature_list)

# Convert to numpy arrays
feat_array = np.array(features.values)

# So how are we going to plot all these features??

# Generate colour data that represents which type each point is
colours = {0: 'b',
           1: 'r',
           2: 'g',
           3: 'y'}
silent_patch = patches.Patch(color='b', label='Silent')
spiking_patch = patches.Patch(color='r', label='Spiking')
one_spike_burst_patch = patches.Patch(color='y', label='One-Spike Bursting')
bursting_patch = patches.Patch(color='g', label='Bursting')
colour_data = [colours[val] for val in data["TYPE"].values]

# Plot the distribution of labels in a 3d figure, good start point
plot_features(data, ["GK", "GCAL", "GSK"], colour_data)

# For a number of features, n, there are nC3 different possible 3D plots that can be drawn: for us it is 10
# fig = plt.figure()
# comb = itertools.combinations(range(0, len(feature_list)), 3)
# splot, splot_shape = 1, [2, 5]
# for com in comb:
#     comb_values = list(com)
#     these_features = feature_list[comb_values]
#     plot_subplot(features, these_features, colour_data, fig, splot, splot_shape)
#     splot += 1

# For a number of features, n, there are nC2 different possible 2D plots that can be drawn: for us it is 10
fig = plt.figure()
comb = itertools.combinations(range(0, len(feature_list)), 2)
splot, splot_shape = 1, [2, 5]
for com in comb:
    comb_values = list(com)
    these_features = feature_list[comb_values]
    plot_subplot(features, these_features, colour_data, fig, splot, splot_shape, False)
    splot += 1

# Plot each label type in a sub-figure
# fig1 = plt.figure()
# splot1, splot_shape1 = 1, [2, 2]
# for col in colours:
#     this_data = data.loc[data["TYPE"] == col]
#     this_col = [colours[col] for _ in range(len(this_data.values))]
#     plot_subplot(this_data, ["GK", "GCAL", "GSK"], this_col, fig1,  splot1, splot_shape1)
#     splot1 += 1
#
# # Setup a function in this figure to move all axes simultaneously
# def on_move(event):
#     all_axes = fig1.get_axes()
#     this_axis = event.inaxes
#     print(this_axis)
#     for axis in all_axes:
#         axis.view_init(elev=this_axis.elev, azim=this_axis.azim)
#     fig1.canvas.draw_idle()
#
# c1 = fig1.canvas.mpl_connect('motion_notify_event', on_move)
# Show plots
plt.show()
