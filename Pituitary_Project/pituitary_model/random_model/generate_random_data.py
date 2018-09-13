import utils, subprocess, os

# Define the variables for the run
ITERATIONS = 100
BASH_SCRIPT = 'ode_runner_random.txt'

# Initiate Run
out = subprocess.run("bash {} {}".format(BASH_SCRIPT, ITERATIONS), shell=True)

# Collect output from folder and process
filename = "{}_iterations_3var.txt".format(ITERATIONS)
new_data = utils.extract_key_data(os.getcwd()+"/auto_outputs", filename)
