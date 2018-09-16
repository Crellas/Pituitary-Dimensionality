import utils
import pyqtgraph as pg
import numpy as np

def mouseMovedObj(evt):
    global obj_mouse_pos
    mousePoint = objective_plot.vb.mapSceneToView(evt[0])
    obj_mouse_pos = [mousePoint.x(), mousePoint.y()]

def mouseMovedFunc(evt):
    global func_mouse_pos
    mousePoint = function_plot.vb.mapSceneToView(evt[0])
    func_mouse_pos = [mousePoint.x(), mousePoint.y()]

obj_mouse_pos = [0,0]; func_mouse_pos = [0,0]
proximity = 0.03; objective_limit = 10
feature1 = 2; feature2 = 5

# data = utils.open_from_txt(filename)
filename = '/Users/jamesashford/Documents/NatSci Exeter/2017-18 (Third Year)/Summer Work/Pituitary Dimensionality Project - J. Tabak/Modelling/pituitary_model/list_model/start:0,0_end:5,5_time:11:41:2.txt'
with open(filename) as file:
    data = [[float(digit) for digit in line.split()] for line in file[1:]]
data = np.array(data)

# Plot the full data set in Window
# x1_list = data[:, 0]
# x_data = [data[:, 0], data[:, 1]]; f_data = [data[:, feature1], data[:, feature2]]

# Plot the full data set without f1 = 0 in Window
x1_list = []; x2_list = []; f1_list = []; f2_list = []
for set in data:
    if set[2] != 0:
        x1_list.append(set[0])
        x2_list.append(set[1])
        f1_list.append(set[feature1])
        f2_list.append(set[feature2])
x_data = [np.array(x1_list), np.array(x2_list)]; f_data = [np.array(f1_list), np.array(f2_list)]

# Setup the graphics window
window = pg.GraphicsWindow(title='Plotting Parameter and Feature Dependence of Pituitary Model')

# Initialise the plot figures
objective_plot, function_plot = utils.plot_function_and_objective(window, x_data, f_data, objective_limit)

# Setup proxies for mouse positions
obj_proxy = pg.SignalProxy(objective_plot.scene().sigMouseMoved, rateLimit=60, slot=mouseMovedObj)
func_proxy = pg.SignalProxy(function_plot.scene().sigMouseMoved, rateLimit=60, slot=mouseMovedFunc)

# Update layout with new data
while True:
    objective_plot.clear(); function_plot.clear()

    # Initialise selected array
    selected = []

    # Check cursor in objective box
    if 0 < obj_mouse_pos[0] < objective_limit and 0 < obj_mouse_pos[1] < objective_limit:
        #  Find distance between cursor and points
        for i in range(len(x_data[0])):
            point_x = x_data[0][i]; point_y = x_data[1][i]
            dist_sq = (point_x-obj_mouse_pos[0])**2 + (point_y-obj_mouse_pos[1])**2
            # Add points that are in range of < closeness * objective_limit to selected (and vice versa)
            if dist_sq < (proximity * objective_limit)**2:
                selected.append(i)

    # Check cursor in function box
    if np.nanmin(f_data[0]) < func_mouse_pos[0] < np.nanmax(f_data[0]) and np.nanmin(f_data[1]) < func_mouse_pos[1] < np.nanmax(f_data[1]):
        #  Find distance between cursor and points
        for i in range(len(f_data[0])):
            point_x = f_data[0][i]; point_y = f_data[1][i]
            dist_sq = (point_x-func_mouse_pos[0])**2 + (point_y-func_mouse_pos[1])**2
            # Add points that are in range of < closeness * objective_limit to selected (and vice versa)
            if dist_sq < (proximity * (np.nanmax(f_data[0] - np.nanmin(f_data[0]))))**2:
                selected.append(i)

    # Generate unselected array
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
