import os
import sys
import yaml
import numpy as np

from shtm import helper 
import parameters_space as pars 

# get commmand line arguments
try:
    simulation_script = sys.argv[1]
except:
    print("provide simulation script!")
    raise

assert os.path.isfile('../config.yaml'), "\n>>> ERROR: Create a config file containing a dictionary with your email: config['email']" \
                                         + " and a path to where you want to store the log files config['path'] \n"

with open('../config.yaml', 'r') as cfgfile:
    params_config = yaml.load(cfgfile, Loader=yaml.FullLoader)
    email = params_config['email']
    path = params_config['path']

submission_script_prefix='submission_cluster_analysis'

ps = pars.p

pl=helper.parameter_set_list(ps)
params = pl[0]

# save parameters.py  
helper.copy_scripts(ps['data_path'], "parameters_space.py")

## write (temporary) submission script
N = len(pl)   ## size of this batch

print("\nnumber of parameter sets: %d\n" % (N))

JOBMAX=1000      ## max number of jobs in one batch    

for batch_id in range(int(np.ceil(1.*N/JOBMAX))):

    batch_end = min((batch_id+1)*JOBMAX, N)         # id of parameter set corresponding to end of this batch
    batch_size = batch_end - batch_id*JOBMAX        # size of this batch

    submission_script="%s_%s.sh" % (submission_script_prefix, simulation_script)
    file = open(submission_script, 'w')
    file.write('#!/bin/bash\n')
    file.write('#SBATCH --job-name ' + simulation_script + '\n')                # set the name of the job
    file.write('#SBATCH --array 0-%d\n' % (batch_size-1))                                     # launch an array of jobs
    file.write('#SBATCH --time 01:00:00\n')                                     # specify a time limit
    file.write('#SBATCH --ntasks 1\n')
    file.write('#SBATCH --cpus-per-task 8\n')
    file.write('#SBATCH -o %s' % path + '/log/job_%A_%a.o\n')                     # redirect stderr and stdout to the same file
    file.write('#SBATCH -e %s' % path + '/log/job_%A_%a.o\n')                     # redirect stderr and stdout to the same file
    file.write('#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE\n')                 # send email notifications
    file.write('#SBATCH --mail-user=%s\n' % email)
    file.write('#SBATCH --mem=1000\n')                                         # and reserve 4GB of memory
    file.write('source activate spiking-htm\n')                                # activate conda environment
    file.write('srun python %s %d $SLURM_ARRAY_TASK_ID %d \n' % (simulation_script,batch_id,JOBMAX))                     # call simulation script
    file.close()

    ## execute submission_script
    print("submitting %s" % (submission_script))
    os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
    os.system("sbatch ./%s" % submission_script)
    os.system("rm %s" % submission_script)
