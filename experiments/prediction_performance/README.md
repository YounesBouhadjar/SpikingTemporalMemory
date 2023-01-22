# Running the experiment prediction performance

## prerequisites

to be able to run an experiment, first [prepare your python environment](../../readme.md#prepare-your-python-environment) 
and follow the [instructions for installing nest and nestml](../../readme.md#prepare-your-nest-installation).

## running an experiment

1. On your local machine:

   To reproduce prediction performance for sequence set 2 run:
   ```bash 
   python local_simulation.py training.py parameters_space_task2.py 
   python prediction_performance_analysis.py task2
   ```
  
   To reproduce prediction performance for sequence set 3 and sequence set 2 with STDP as a synaptic model, 
   replace in the code above `parameters_space_task2.py` with `parameters_space_task3.py` or `parameters_space_task2_stdp.py`, 
   and `task2` with `task3` or `task2_stdp`, respectively.
   
2. On the cluster:   
   If you have access to a cluster with SLURM, create `config.yaml` in the experiments' folder 
   containing your email: `config['email']` and the path to where you want to store the log files `config['path']`:
   ```bash
   touch ../config.yaml 
   echo "email : 'youremail'" > ../config.yaml 
   echo "path : 'yourpath'" >> ../config.yaml 
   ```
   and then run the following to reproduce data for sequence set 2: 
   ```bash
   python submission_cluster_simulation.py training.py parameters_space_task2.py
   python submission_cluster_analysis.py prediction_performance_analysis.py task2
   ```
   
To visualize the data look at [Reproduce figures](../../README.md#reproduce-figures). 
The generated data overwrites the accompanied data [here](../../data).
