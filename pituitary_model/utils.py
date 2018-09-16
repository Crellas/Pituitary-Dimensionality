# Near-universal imports here
import numpy as np

# Data Extraction Functions

def parse_filename(filename):
    """ Get information about the file from the stored filename - not used much anymore"""
    name = filename.split("/")[-1]
    subsec = name.split("_")[0:-1]
    iter = subsec[0].split(":")[-1]
    range = subsec[1].split(":")[-1]
    objective_limit = int(range.split("-")[-1])
    return iter, objective_limit


def check_spiking(gradient_data):
    spiking = True
    if np.mean(gradient_data[-100:-1]) == 0.0:
        spiking_value = 0
        spiking = False
    else:
        spiking_value = 1
    return spiking, spiking_value


def all_features(t_data, v_data):
    """Extract features from the t_data and v_data"""
    feature_list = []
    data_type = 0

    #Count maxima to determine the number of events; if 0 then silent
    # Find all maxima times and voltage values, assuming greater than mean / 3
    maxima = [[t_data[pos], v_data[pos], pos] for pos in range(1, len(v_data) - 1) if
              v_data[pos] > v_data[pos - 1] and v_data[pos] > v_data[pos + 1] and v_data[pos] > np.mean(v_data) / 3]

    #Calculate mean separation between events
    mean_sep = 0
    if len(maxima) >= 2:
        # Find mean time separation between adjacent maxima
        maxima_separations = [(maxima[pos][0] - maxima[pos - 1][0]) for pos in range(len(maxima) - 1)]
        valid_seps = [val for val in maxima_separations if val > 0.01]
        if len(valid_seps) > 0:
            mean_sep = np.mean(valid_seps)

    #Define storage arrays for mean values
    event_amplitudes, event_max_dvdts, event_min_dvdts = [], [], []
    event_spikes, event_lengths, event_line_lengths = [], [], []

    if mean_sep != 0:
        # Get "event" data by looking ±mean_sep/2 around each maxima
        sep_count = int(mean_sep / (t_data[1] - t_data[0]))
        for maximum in maxima:
            # Reset event_duration variables
            start_time, end_time = 0, 0
            # Define the data for this event, range of mean separation around each maxima
            lower = int(maximum[2] - (0.75 * sep_count)); upper = int(maximum[2] + (0.75 * sep_count))
            if lower < 0: lower = 0
            event_volt = v_data[lower:upper]
            event_time = t_data[lower:upper]

            # Get event gradient data
            event_dvdt = np.gradient(event_volt)

            ## Find event amplitude
            event_amplitudes.append(np.nanmax(event_volt) - np.nanmin(event_volt))

            ## Find event dvdt max and min
            dvdt_max_pos = np.argmax(event_dvdt); dvdt_min_pos = np.argmin(event_dvdt)
            dvdt_max = event_dvdt[dvdt_max_pos]; dvdt_min = event_dvdt[dvdt_min_pos]
            event_max_dvdts.append(dvdt_max); event_min_dvdts.append(dvdt_min)

            ## Find the number of spikes (maxima) in each event
            event_maxima = [[event_time[pos], event_volt[pos], pos] for pos in range(1, len(event_volt) - 1) if
                      event_volt[pos] > event_volt[pos - 1] and event_volt[pos] > event_volt[pos + 1]]
            spikes = len(event_maxima)
            event_spikes.append(spikes)

            ## Find event length
            # Calculate thresholds
            dvdt_max_threshold = 0.25 * event_dvdt[dvdt_max_pos]
            dvdt_min_threshold = 0.25 * event_dvdt[dvdt_min_pos]
            v_thresh = 0.35 * (np.nanmax(event_volt) - np.nanmin(event_volt))

            # start point; time when > v threshold and first > maximum dvdt threshold
            over_v_indices = np.where(np.array(event_volt) > v_thresh)[0]
            over_dvdt_indices = np.where(np.array(event_dvdt) > dvdt_max_threshold)[0]
            overlap = [ind for ind in over_v_indices if ind in over_dvdt_indices]
            if len(overlap) == 0:
                print('Error! No valid start point found!')
                event_lengths.append(0)
                continue
            else:
                # Of those, find index with lowest dvdt value
                dvdt_values = [event_dvdt[ind] for ind in overlap]
                start_index = overlap[np.argmin(dvdt_values)]
                start_time = event_time[start_index]

            # end point; time when < v threshold and first > lower dvdt threshold
            under_v_indices = np.where(np.array(event_volt) < v_thresh)[0]
            over_dvdt_indices = np.where(np.array(event_dvdt) > dvdt_min_threshold)[0]
            overlap = [ind for ind in under_v_indices if ind in over_dvdt_indices]
            if len(overlap) == 0:
                print('Error! No valid end point found!')
                event_lengths.append(0)
                continue
            else:
                # Of those, find index with lowest dvdt value
                dvdt_values = [event_dvdt[ind] for ind in overlap]
                end_index = overlap[np.argmin(dvdt_values)]
                end_time = event_time[end_index]

            # Event time
            event_length = end_time - start_time
            event_lengths.append(event_length)

            # Calculate length of line between max V and last peak of event; if these are the same return 0
            if spikes == 1:
                line_length = 0
            else:
                line_start_index = np.argmax(event_volt); line_end_index = event_maxima[-1][2]
                # line length is the sum of the distance between all the points in this index range
                # we know separation is always 0.001 seconds, thus dist = root(x^2 + y^2) = root(0.00001 +
                line_length = np.sum([(0.001**2 + (event_volt[i+1]-event_volt[i]))**0.5 for i in range(line_start_index, line_end_index)])
            event_line_lengths.append(line_length)

        # Calc mean event amplitude
        mean_amplitude = float(np.mean(event_amplitudes))
        # Calc mean dvdt max and min
        mean_dvdt_max = float(np.mean(event_max_dvdts))
        mean_dvdt_min = float(np.mean(event_min_dvdts))
        # Calc mean num of spikes in each event
        mean_spikes = int(np.mean(event_spikes))
        # Calc mean event length
        mean_length = float(np.mean(event_lengths))
        # Calc mean line length
        mean_line_length = float(np.mean(event_line_lengths))

        # If event length greater than threshold and/or number of spikes > 1 then bursting, else spiking
        if mean_spikes != 0:
            data_type = 1
            if mean_length > 0.06:
                data_type = 3
            if mean_spikes > 1:
                data_type = 2

        feature_list = [data_type, mean_sep, mean_length, mean_amplitude, mean_dvdt_max, mean_dvdt_min, mean_spikes, mean_line_length]

    else:
        feature_list = [0 for _ in range(8)]

    #Add these all to the feature list
    return feature_list


def extract_all_data(filename):
    """Get the basic data from the .dat files; the labels and v,t data"""
    import csv
    # Open the filename in read mode with temporary storage
    with open(filename, 'r') as myarray:
        data = csv.reader(myarray, delimiter=' ')
        # Extract data into comma separated list
        data = [row for row in data]

    # Get time data at position 5
    t_data = [float(dat[5]) for dat in data]
    # Get Voltage data at position 1
    v_data = [float(dat[1]) for dat in data]
    gk = float(data[0][6])
    gcal = float(data[0][7])
    gsk = float(data[0][8])
    # return this data as two separate arrays and two floats
    return t_data, v_data, gk, gcal, gsk


def extract_key_data(data_folder_path, save_name):
    """Get the key feature data from the files in the given folder and save to a .txt file"""
    import os

    # Get all files in the target folder
    files = os.listdir(data_folder_path)

    # Generate the data from the files
    out_data = []
    file_count = len(files); count = 0
    for file in files:
        # Ensure this is not a hidden file
        if not file.startswith('.'):
            # Print the filename
            print(file)
            print("{}% Complete".format(round((count/file_count)*100, 4)))

            # Extract all the data from the .dat file
            t_data, v_data, gk, gcal, gsk = extract_all_data(data_folder_path+"/"+file)

            # Ignore first two seconds
            ignore = 2; time_step = t_data[1]-t_data[0]
            start = int(ignore/time_step)
            t_data = t_data[start:-1]; v_data = v_data[start:-1]

            # Normalise the v_data
            v_max = max(v_data); v_min = min(v_data)
            normalised_v_data = [(val-v_min)/(v_max-v_min) if v_max != v_min else 0 for val in v_data]

            file_data = [gk, gcal, gsk]

            features = all_features(t_data, normalised_v_data)

            file_data = file_data + features

            out_data.append(file_data)
        count += 1

    headers = ["GK", "GCAL", "GSK", "TYPE", "EVENT_SEPARATION", "EVENT_DURATION", "EVENT_AMPLITUDE", "EVENT_DVDT_MAX",
               "EVENT_DVDT_MIN", "EVENT_BURSTPEAKS", "EVENT_LINELENGTH"]
    np.savetxt(save_name, out_data, delimiter=',', header=','.join(headers), comments='')

    return out_data


# Plotting Functions

def plot_function_and_objective(window, x_data, f_data):
    """Plot Objective and Function space of the original data in one window - not used much anymore"""
    obj_col = (0, 0, 255)
    func_col = (0, 0, 255)

    # Make plot of objective space
    objective_plot = window.addPlot(title='Objective Space')
    objective_plot.setLabel('bottom', "GK")
    objective_plot.setLabel('left', "GCAL")
    objective_data = objective_plot.plot(x_data[0], x_data[1], pen=None, symbolBrush=obj_col, symbolSize=5,
                                       symbolPen=None)


    # Make plot of function space
    function_plot = window.addPlot(title='Function Space')
    function_data = function_plot.plot(f_data[0], f_data[1], pen=None, symbolBrush=func_col, symbolSize=5,
                                       symbolPen=None)

    return objective_plot, function_plot


def plot_two_basic_windows(filename, obj_limit):
    """Plot Function and Objective data in two windows, from file, either using the full data or just
    those without 0 mean separation - not used much anymore """
    import pyqtgraph as pg
    from pyqtgraph.Qt import QtGui, QtCore

    with open(filename) as file:
        data = [[float(digit) for digit in line.split()] for line in file]
    data = np.array(data)

    # Plot the full data set in Window 1
    x1_list = data[:, 0]; x2_list = data[:, 1]
    f1_list = data[:, 2]; f2_list = data[:, 3]
    x_data = [x1_list, x2_list]; f_data = [f1_list, f2_list]

    window1 = pg.GraphicsWindow(title='Plotting Objective and Function Dependence of Pituitary Model')
    plot_function_and_objective(window1, x_data, f_data, obj_limit)

    # Plot a subset of data set, based on removing all 0 mean separations, in Window 2
    x1_list = []; x2_list = []; f1_list = []; f2_list = []
    for set in data:
        if set[2] != 0:
            x1_list.append(set[0])
            x2_list.append(set[1])
            f1_list.append(set[2])
            f2_list.append(set[3])
    x_data = [x1_list, x2_list]; f_data = [f1_list, f2_list]

    window2 = pg.GraphicsWindow(title='Plotting Objective and Function Dependence of Pituitary Model')
    plot_function_and_objective(window2, x_data, f_data, obj_limit)

    # Show the plot in app
    QtGui.QApplication.instance().exec_()

    return window1, window2



