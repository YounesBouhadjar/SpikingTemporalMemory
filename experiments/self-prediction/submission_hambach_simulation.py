import os
import sys
from docopt import docopt
import numpy as np

from shtm import helper 

# get commmand line arguments
try:
    simulation_script = sys.argv[1]
except:
    print("provide simulation script!")

if simulation_script == 'training.py':

    import parameters_space as data_pars 
 
    # get parameters 
    PS = data_pars.p
    label = "training"
else:

    # data path
    path_dict = {}
    path_dict['data_root_path'] = 'data'
    path_dict['project_name'] = 'sequence_learning_performance'
    path_dict['parameterspace_label'] = "stimulus_timing_analysis"
    
    # get parameters 
    PS, data_path = helper.get_parameter_set(path_dict)
    label = "replay"

PL=helper.parameter_set_list(PS)
params = PL[0]

# save parameters.py  
if simulation_script == "training.py":
    helper.copy_scripts(PS['data_path'], "parameters_space.py")

## write (temporary) submission script
N = len(PL)   ## size of this batch

print("\nNumber of parameter sets: %d\n" % (N))

JOBMAX=1000      ## max number of jobs in one batch    

for batch_id in range(int(np.ceil(1.*N/JOBMAX))):

    batch_end = min((batch_id+1)*JOBMAX, N)   ## id of parameter set corresponding to end of this batch
    batch_size = batch_end - batch_id*JOBMAX   ## size of this batch
    submission_script="%s_%d.sh" % (params['data_path']['parameterspace_label'],batch_id)
    #submission_script="%s.sh" % (params['data_path']['parameterspace_label'])

    file = open(submission_script, 'w')
    file.write('#!/bin/bash\n')
    file.write('#SBATCH --job-name ' + params['data_path']['parameterspace_label'] + '_' + label + '\n')               # set the name of the job
    file.write('#SBATCH --array 0-%d\n' % (batch_size-1))                       # launch an array of jobs
    file.write('#SBATCH --time 24:00:00\n')                                     # specify a time limit
    file.write('#SBATCH --ntasks 1\n')
    if simulation_script != "replay.py":
        file.write('#SBATCH --cpus-per-task %d\n' % params['n_threads'])
    else:    
        file.write('#SBATCH --cpus-per-task 1\n')
    file.write('#SBATCH -o /users/bouhadjar/log/job_%A_%a.o\n')               # redirect stderr and stdout to the same file
    file.write('#SBATCH -e /users/bouhadjar/log/job_%A_%a.e\n')               # redirect stderr and stdout to the same file
    file.write('#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE\n')                  # send email notifications
    file.write('#SBATCH --mail-user=y.bouhadjar@fz-juelich.de\n')
    file.write('#SBATCH --mem=6000\n')                                          # and reserve 4GB of memory
    file.write('source activate nest_sim \n')                                         # activate conda environment
    file.write('srun python %s %d $SLURM_ARRAY_TASK_ID %d \n' % (simulation_script,batch_id,JOBMAX) ) # call simulation script
    file.write('scontrol show jobid ${SLURM_JOBID} -dd # Job summary at exit')
    file.close()
    
    ## execute submission_script
    print("submitting %s" % (submission_script))
    os.system("sbatch ./%s" % submission_script)
    os.system("rm %s" % submission_script)
