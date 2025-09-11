import os
import sys
from joblib import Parallel, delayed

from spikingtemporalmemory import helper 
import parameters_space as data_pars 

def sim(i):
    os.system("python %s 0 %d 0" % (simulation_script, i))

# get commmand line arguments
try:
    simulation_script = sys.argv[1]
except:
    print("provide simulation script!")

# get parameters 
PS = data_pars.p
 
# parameters list 
PL = helper.parameter_set_list(PS)

# save parameters.py  
helper.copy_scripts(PS['data_path'], "parameters_space.py")

# simulation 
N = len(PL)

os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
Parallel(n_jobs=1)(delayed(sim)(i) for i in range(N))
