import matplotlib, re
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import numpy as np
from io import StringIO
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix
from scipy.interpolate import griddata
from sklearn.externals import joblib  #For saving, use joblib.dump(model, name); for loading use joblib.load(name)

def report_to_df(report):
    report = re.sub(r" +", " ", report).replace("avg / total", "avg/total").replace("\n ", "\n")
    report_df = pd.read_csv(StringIO("Classes" + report), sep=' ', index_col=0)
    return(report_df)

## Import data to a pandas dataframe
data_file = '/Users/jamesashford/Documents/NatSci Exeter/2017-18 (Third Year)/Summer Work/Pituitary Dimensionality Project - J. Tabak/Neural Network/10000_gbk0.5.txt'
#Read data from file
data = pd.read_csv(data_file, header=0)
#Read headers from file
column_headers = data.columns.values
# Get the feature data
X = data[["GK", "GCAL", "GSK"]]
# Get label data
Y = data["TYPE"]
# Split into training and testing data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y)

## Preprocess the data by scaling
scaler = StandardScaler()
# Fit to the training feature data only
scaler.fit(X_train)
# Rescale the feature data
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
X_all = scaler.transform(X)

# ^^ ALL SETUP


# Gather data about the behaviour of the model
layer_size = [3, 5, 8, 10, 20]
print(layer_size)
layers_depth = [1, 2, 3, 4, 5]
print(layers_depth)
save_data = []
for size in layer_size:
    for depth in layers_depth:
        this_data = []
        model = tuple([size for _ in range(depth)])
        print(model)
        for i in range(20):
            ## Training the model
            mlp = MLPClassifier(hidden_layer_sizes=model, max_iter=10000, learning_rate='adaptive')
            mlp.fit(X_train, Y_train)

            ## Predict for full set
            predict = mlp.predict(X_all)
            print(confusion_matrix(Y, predict))
            report = classification_report(Y, predict)
            report = report_to_df(report)
            this_data.append(report["precision"].values)
        save_data.append(np.mean(this_data, axis=0))

save_data = np.array(save_data)
mean_precision = save_data[:, 2]

# Plot in 3D
x_data = list(np.ravel([[n] * len(layers_depth) for n in layer_size]))
y_data = list(layers_depth)*len(layer_size)
z_data = mean_precision

# Scatter Plot
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(x_data, y_data, z_data)
ax.set_title("Investigating Mean Precision related to Layer Size and Model Depth")
ax.set_xlabel("Layer Size")
ax.set_ylabel("Layer Count")
ax.set_zlabel("Mean Precision")

# Blue Meshgrid
# fig1 = plt.figure()
# ax1 = Axes3D(fig1)
# X, Y = np.meshgrid(x_data, y_data)
# Z = griddata((x_data, y_data), z_data, (X, Y), method='cubic')
# ax1.plot_surface(X, Y, Z)
# ax1.set_title("Investigating Mean Precision related to Layer Size and Model Depth")
# ax1.set_xlabel("Layer Size")
# ax1.set_ylabel("Model Depth")
# ax1.set_zlabel("Mean Precision")
#
# Tri Surf Plot
# fig2 = plt.figure()
# ax2 = Axes3D(fig2)
# ax2.plot_trisurf(x_data, y_data, z_data)
# ax2.set_title("Investigating Mean Precision related to Layer Size and Model Depth")
# ax2.set_xlabel("Layer Size")
# ax2.set_ylabel("Model Depth")
# ax2.set_zlabel("Mean Precision")
plt.show()