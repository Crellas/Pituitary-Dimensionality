import utils
import pyqtgraph as pg
import numpy as np

def mouseMoved(evt):
    global mouse_pos
    mousePoint = objective_plot.vb.mapSceneToView(evt[0])
    mouse_pos = [mousePoint.x(), mousePoint.y()]

mouse_pos = [0,0]
proximity = 0.05

# Open data from file
filename = '/Users/jamesashford/Documents/NatSci Exeter/2017-18 (Third Year)/Summer Work/Pituitary Dimensionality Project - J. Tabak/Modelling/Python Analysis/pituitary_model/random_model/iter:10000_range:0-10_time:03:16:4.txt'
with open(filename) as file:
    data = [[float(digit) for digit in line.split()] for line in file]
data = np.array(data)
iterations, objective_limit = utils.parse_filename(filename)

# Plot the full data set in Window
# x1_list = data[:, 0]
# x_data = [data[:, 0], data[:, 1]]; f_data = [data[:, 2], data[:, 3]]

# Plot the full data set without f1 = 0 in Window
x1_list = []; x2_list = []; f1_list = []; f2_list = []
for set in data:
    if set[2] != 0:
        x1_list.append(set[0])
        x2_list.append(set[1])
        f1_list.append(set[3])
        f2_list.append(set[5])
x_data = [np.array(x1_list), np.array(x2_list)]; f_data = [np.array(f1_list), np.array(f2_list)]

window = pg.GraphicsWindow(title='Plotting Parameter and Feature Dependence of Pituitary Model')

objective_plot, function_plot = utils.plot_function_and_objective(window, x_data, f_data)
proxy = pg.SignalProxy(objective_plot.scene().sigMouseMoved, rateLimit=60, slot=mouseMoved)

# Update layout with new data
while True:
    objective_plot.clear(); function_plot.clear()

    selected = []
    # Selected within a radius of the cursor
    # Check cursor in objective box
    if 0 < mouse_pos[0] < objective_limit and 0 < mouse_pos[1] < objective_limit:
        #  Find distance between cursor and points
        for i in range(len(x_data[0])):
            point_x = x_data[0][i]; point_y = x_data[1][i]
            dist_sq = (point_x-mouse_pos[0])**2 + (point_y-mouse_pos[1])**2
            # Add points that are in range of < closeness * objective_limit to selected (and vice versa)
            if dist_sq < (proximity * objective_limit)**2:
                selected.append(i)

    unselected = np.delete(np.arange(0, len(x1_list)), selected)

    # Plot unselected data in blue
    objective_selected_data = [x_data[0][unselected], x_data[1][unselected]]
    objective_plot.plot(objective_selected_data[0], objective_selected_data[1], pen=None,
                                         symbolBrush=(0, 0, 255), symbolSize=5,
                                         symbolPen=None)
    function_selected_data = [f_data[0][unselected], f_data[1][unselected]]
    function_plot.plot(function_selected_data[0], function_selected_data[1], pen=None,
                                         symbolBrush=(0, 0, 255),
                                         symbolSize=5,
                                         symbolPen=None)

    # Plot selected data in red
    objective_selected_data = [x_data[0][selected], x_data[1][selected]]
    objective_plot.plot(objective_selected_data[0], objective_selected_data[1], pen=None, symbolBrush=(255,0,0), symbolSize=5,
                                         symbolPen=None)
    function_selected_data = [f_data[0][selected], f_data[1][selected]]
    function_plot.plot(function_selected_data[0], function_selected_data[1], pen=None, symbolBrush=(255, 0, 0),
                                         symbolSize=5,
                                         symbolPen=None)

    pg.QtGui.QApplication.processEvents()
