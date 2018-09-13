import os
import time
import utils
import subprocess

# Define the variables for the run
start_point = [0, 0, 0]; end_point = [5, 5, 5]
bash_script = 'ode_runner_list.txt'
samples = 40

# Initiate Run
out = subprocess.run("bash {} {} {} {} {} {} {} {}".format(bash_script, start_point[0], start_point[1], start_point[2], end_point[0], end_point[1], end_point[2], samples), shell=True)

# Collect output from folder and process
filename = 'start:{},{},{}_end:{},{},{}_time:{}.txt'.format(start_point[0], start_point[1], start_point[2], end_point[0], end_point[1], end_point[2], time.ctime()[11:18])
new_data = utils.extract_key_data(os.getcwd()+"/auto_outputs", filename)
