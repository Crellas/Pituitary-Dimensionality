import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import numpy as np

def gen_data_size_iter(layer_size, iter_values, repeat):
    # Store the MSE at each location then plot on 3D graph with colour designated by MSE?
    MSE_store = [];
    num_counts = len(layer_size) * len(iter_values) * repeat;
    count = 0
    # Do layer_size and iteration
    for size in layer_size:
        model = tuple([size])
        for iterations in iter_values:
            this_MSE = []
            for i in range(repeat):
                # Setup the model from the looped parameters
                mlp = MLPRegressor(hidden_layer_sizes=model, max_iter=iterations)
                mlp.fit(X_train, Y_train)

                # Predict values from test set
                pred_test = mlp.predict(X_test)

                # Calculate MSE
                MSE = mean_squared_error(Y_test, pred_test)
                this_MSE.append(MSE)

                # Percent complete
                print("{}% complete".format(round((count / num_counts) * 100, 4)))
                print(MSE)
                count += 1
            MSE_store.append(np.mean(this_MSE))

    # Plot the data in 3D
    # x data is the layer size
    x_data = list(np.ravel([[n] * len(iter_values) for n in layer_size]))
    # y data is the iter_values
    y_data = list(iter_values) * len(layer_size)
    # z is the MSE values
    z_data = MSE_store
    labels = ["Layer Size", "Iteration Count", "MSE Value"]
    return x_data, y_data, z_data, labels

def gen_data_count_size(layer_count, layer_size, iterations, repeat):
    # Store the MSE at each location then plot on 3D graph with colour designated by MSE?
    MSE_store = [];
    num_counts = len(layer_size) * len(layer_count) * repeat;
    count = 0
    # Do layer_size and iteration
    for count in layer_count:
        for size in layer_size:
            model = tuple([size]*count)
            print(model)
            this_MSE = []
            for i in range(repeat):
                # Setup the model from the looped parameters
                mlp = MLPRegressor(hidden_layer_sizes=model, max_iter=iterations)
                mlp.fit(X_train, Y_train)

                # Predict values from test set
                pred_test = mlp.predict(X_test)

                # Calculate MSE
                MSE = mean_squared_error(Y_test, pred_test)
                this_MSE.append(MSE)

                # Percent complete
                print("{}% complete".format(round((count / num_counts) * 100, 4)))
                print('')
                count += 1
            MSE_store.append(np.mean(this_MSE))

    # Plot the data in 3D
    # x data is the layer count
    x_data = list(np.ravel([[n] * len(layer_size) for n in layer_count]))
    # y data is the layer size
    y_data = list(layer_size) * len(layer_count)
    # z is the MSE values
    z_data = MSE_store
    labels = ["Layer Count", "Layer Size", "MSE Value"]
    return x_data, y_data, z_data, labels

## Import data to a pandas dataframe
data_file = '/Users/jamesashford/Documents/NatSci Exeter/2017-18 (Third Year)/Summer Work/Pituitary Dimensionality Project - J. Tabak/Neural Network/10000_gbk0.txt'
#Read data from file
data = pd.read_csv(data_file, header=0)
column_headers = data.columns.values
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

# ^^ THIS IS ALL DATA LOADING AND SETUP ^^


# We want to investigate the effect of layer size, layer count and iteration count on the MSE (and varience coverage)
layer_count = [1, 2, 3, 4, 5]
layer_size = [3, 4, 5, 8, 10, 20]
iter_values = [200, 500, 1000, 2000, 5000, 10000, 15000, 20000]
repeat = 1

#x_data, y_data, z_data, labels = gen_data_size_iter(layer_size, iter_values, repeat)
x_data, y_data, z_data, labels = gen_data_count_size(layer_count, layer_size, 2000, repeat)

print(x_data)
print(y_data)
print(z_data)

# Scatter Plot
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(x_data, y_data, z_data)
ax.set_title("Investigating Mean Precision related to Layer Count, Layer Size and Model Depth")
ax.set_xlabel(labels[0])
ax.set_ylabel(labels[1])
ax.set_zlabel(labels[2])
plt.show()



