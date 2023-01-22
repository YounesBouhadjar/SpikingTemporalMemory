import os
import sys
from joblib import Parallel, delayed

from shtm import helper 

def sim(i):
    os.system("python %s 0 %d 0" % (simulation_script, i))

# get commmand line arguments
try:
    simulation_script = sys.argv[1]
except:
    print("provide simulation script!")

if simulation_script == "training.py":

    import parameters_space as data_pars 
 
    # get parameters 
    PS = data_pars.p
else:
    path_dict = {}
    path_dict['data_root_path'] = "data"
    path_dict['project_name'] = "sequence_learning_performance"
    path_dict['parameterspace_label'] = "stimulus_timing_analysis"

    # get parameters 
    PS, data_path = helper.get_parameter_set(path_dict)
 
# parameters list 
PL = helper.parameter_set_list(PS)

# save parameters.py  
if simulation_script == "training.py":
    helper.copy_scripts(PS['data_path'], "parameters_space.py")

# simulation 
N = len(PL)

os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
Parallel(n_jobs=2)(delayed(sim)(i) for i in range(N))
