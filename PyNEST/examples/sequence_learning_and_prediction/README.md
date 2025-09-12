# Running the experiment sequence learning and prediction

## Prerequisites

To be able to run an experiment, first [prepare your python environment](../../README.md) 
and make sure to follow the [instructions for installing NEST](../../README.md#software-requirements).

## Running an experiment #TODO outdated

1. On your local machine:
   ```bash 
   python local_simulation.py training.py 
   #TODO python prediction_performance_analysis.py
   ```

   To run wandb sweep:
   ```bash
   python generate_sweep_id.py wb_configs/xxx_config.yaml         # generate a `sweep_id`
   wandb agent wandb-user_name/project_name/sweep_id
   ```

2. On the cluster:   
   If you have access to a cluster with SLURM, create `config.yaml` in the experiments' folder 
   containing your email: `config['email']` and the path to where you want to store the log files `config['path']`:
   ```bash
   touch ../config.yaml 
   echo "email : 'youremail'" > ../config.yaml 
   echo "path : 'yourpath'" >> ../config.yaml 
   ```
   and then run:
   ```bash
   python submission_cluster_simulation.py training.py parameters_space.py
   #TODO python submission_cluster_analysis.py prediction_performance_analysis.py
   ```

   To run wandb sweep:
   ```bash
   python submission_cluster_simulation.py wandb-user_name/project_name/sweep_id NUMBER_RUNS parameters_space.py 
   ```