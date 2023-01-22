import os
import sys
from joblib import Parallel, delayed

from shtm import helper 

def sim(i):
    os.system("python %s 0 %d 0 %s" % (simulation_script, i, pars_space))

# get commmand line arguments
try:
    simulation_script = sys.argv[1]
    pars_space = sys.argv[2]
except:
    print("provide simulation script and task type!")

if pars_space == 'parameters_space_task2.py':
    import parameters_space_task2 as pars 
elif pars_space == 'parameters_space_task3.py':
    import parameters_space_task3 as pars 
elif pars_space == 'parameters_space_task2_stdp.py':
    import parameters_space_task2_stdp as pars 
elif pars_space == 'parameters_space_task4.py':
    import parameters_space_task4 as pars 
else:
    import parameters_space as pars 

# get parameters 
PS = pars.p
 
# parameters list 
PL = helper.parameter_set_list(PS)

# save parameters.py  
helper.copy_scripts(PS['data_path'], pars_space)

# simulation 
N = len(PL)

os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
Parallel(n_jobs=2)(delayed(sim)(i) for i in range(N))
