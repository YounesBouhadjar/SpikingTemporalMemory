import os
import sys
import yaml
import wandb
import numpy as np

from spikingtemporalmemory import helper 


#TODO use an argument parser to get command line arguments
try:
    sweep_id_path = sys.argv[1]
    N = int(sys.argv[2])
    params_script = sys.argv[3]
except:
    raise ValueError('provide sweep id path, the number of parallel runs and the parameter script!')

assert os.path.isfile('../config.yaml'), "\n>>> ERROR: Create a config file containing a dictionary with your email: config['email']" \
                                         + " and a path to where you want to store the log files config['path'] \n"

with open('../config.yaml', 'r') as cfgfile:
    params_config = yaml.load(cfgfile, Loader=yaml.FullLoader)
    email = params_config['email']
    path = params_config['path']

PS = __import__(params_script.split(".")[0]).p

PL = helper.parameter_set_list(PS)
params = PL[0]

# save parameters.py  
# helper.copy_scripts(PS['data_path'], params_script)

print("\nNumber of parallel runs: %d\n" % (N))

JOBMAX = 1000      ## max number of jobs in one batch    

for batch_id in range(int(np.ceil(1.*N/JOBMAX))):

    batch_end = min((batch_id+1)*JOBMAX, N)         # id of parameter set corresponding to end of this batch
    batch_size = batch_end - batch_id*JOBMAX        # size of this batch
    submission_script="%s_%d.sh" % (params['data_path']['parameterspace_label'], batch_id)

    file = open(submission_script, 'w')
    file.write('#!/bin/bash\n')
    file.write('#SBATCH --job-name ' + params['data_path']['parameterspace_label'] + '\n')    # set the name of the job
    file.write('#SBATCH --array 0-%d\n' % (batch_size-1))                                     # launch an array of jobs
    file.write('#SBATCH --time 24:00:00\n')                                                   # specify a time limit
    file.write('#SBATCH --ntasks 1\n')
    file.write('#SBATCH --cpus-per-task %d\n' % params['n_threads'])
    file.write('#SBATCH -o %s' % path + '/log/job_%A_%a.o\n')               # redirect stderr and stdout to the same file
    file.write('#SBATCH -e %s' % path + '/log/job_%A_%a.e\n')               # redirect stderr and stdout to the same file
    file.write('#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE\n')              # send email notifications
    file.write('#SBATCH --mail-user=%s\n' % email)
    file.write('#SBATCH --mem-per-cpu=5000\n')                                      # and reserve 6GB of memory
    file.write('wandb agent %s\n' % sweep_id_path)
    file.write('scontrol show jobid ${SLURM_JOBID} -dd # Job summary at exit')
    file.close()
    
    ## execute submission_script
    print("submitting %s" % (submission_script))
    os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
    os.system("sbatch ./%s" % submission_script)
    os.system("rm %s" % submission_script)
