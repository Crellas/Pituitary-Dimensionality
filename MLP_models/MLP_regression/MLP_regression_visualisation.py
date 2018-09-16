import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pylab as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
import pandas as pd
import numpy as np


def plot_weights(before, after):
    shape = [1, 8]
    fig, axes = plt.subplots(2, 5)
    fig.suptitle("Before Training")
    # use global min / max to ensure all weights are shown on the same scale
    vmin, vmax = after[0].min(), after[0].max()
    for coef, ax in zip(before[0].T, axes.ravel()):
        ax.matshow(np.absolute(coef.reshape(shape[0], shape[1])), cmap=plt.cm.gray, vmin=vmin, vmax=vmax)
        ax.set_xticks(())
        ax.set_yticks(())
    fig, axes = plt.subplots(2, 5)
    fig.suptitle("After Training")
    # use global min / max to ensure all weights are shown on the same scale
    for coef, ax in zip(after[0].T, axes.ravel()):
        ax.matshow(np.absolute(coef.reshape(shape[0], shape[1])), cmap=plt.cm.gray, vmin=vmin, vmax=vmax)
        ax.set_xticks(())
        ax.set_yticks(())
    plt.show()


## Import data to a pandas dataframe
display_label = "GSK"
id_labels = {"GK": 0,
             "GCAL": 1,
             "GSK": 2}
label_id = id_labels[display_label]

data_file = '/Users/jamesashford/Documents/NatSci Exeter/2017-18 (Third Year)/Summer Work/Pituitary Dimensionality Project - J. Tabak/Neural Network/10000_gbk0.5.txt'
#Read data from file
data = pd.read_csv(data_file, header=0)
#Read headers from file
column_headers = data.columns.values
#Remove silent data
data = data[data["TYPE"] != 0]

# Get the feature and label data
X = data.drop(["GK", "GCAL", "GSK"], axis=1)
Y = data[["GK", "GCAL", "GSK"]]
# Split into training and testing data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y)

## Train a scaler for the data
scaler = StandardScaler()
scaler.fit(X_train)
# Rescale the feature data
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

weights = None
repeats = 50
weight_dictionary = {}
for i in range(repeats):
    print(i)
    ## Training the model
    mlp = MLPRegressor(hidden_layer_sizes=(10), max_iter=1000)
    # Train it on one sample so it generates some starting weights
    # mlp.fit(X_train[0:2], Y_train.values[0:2])
    # before_weights = mlp.coefs_
    mlp.fit(X_train, Y_train)
    after_weights = mlp.coefs_

    # Plot weights before and after in matrix format
    # plot_weights(before_weights, after_weights)

    # Do some weight wrangling to investigate the method of the model
    # Round the weight to nearest decimal place
    thresh_array = np.array(np.absolute(after_weights.copy()))
    # thresh_array[0] = np.round(thresh_array[0], 1)
    # thresh_array[1] = np.round(thresh_array[1], 1)
    thresh = 0.3
    for layer in thresh_array:
        # Threshold data with a thresh value, 1 if higher, 0 if lower
        layer[layer > thresh] = 1
        layer[layer < thresh] = 0


    # Store each row of the threshold units in the dictionary, by concatenating the strings and storing
    # Turn each hidden_units array of data into a single string, which can be separated out later
    arrays = [list(thresh_array[0][:, val]) + list(thresh_array[1][val]) for val in range(len(thresh_array[1]))]
    # Test each of these lists and update dictionary
    for input_list in arrays:
        input_list = ','.join(str(int(val)) for val in input_list)
        if input_list in weight_dictionary:
            weight_dictionary[input_list] += 1
        else:
            weight_dictionary[input_list] = 1

    # Sum along axes to find the importance of each node to the other
    thresh_array = [np.sum(thresh_array[0], axis=1), np.sum(thresh_array[1], axis=0)]

    # Add these weights to the previous iterations
    if weights is None:
        weights = thresh_array
    else:
        weights = np.add(weights, thresh_array)

# Sort the weight dictionary
sorted_weights = {}
for val in sorted(weight_dictionary, key=weight_dictionary.get, reverse=True):
    sorted_weights[val] = weight_dictionary[val]
print(sorted_weights)

# Normalise all the values
# minmax = [0, 0]
# for layer in weights:
#     for val in layer:
#         if val < minmax[0]:
#             minmax[0] = val
#         if val > minmax[1]:
#             minmax[1] = val
# weights = [[round((val-minmax[0])/(minmax[1]-minmax[0]), 2) for val in layer] for layer in weights]
# # Display
# for layer in weights:
#     print(layer)

