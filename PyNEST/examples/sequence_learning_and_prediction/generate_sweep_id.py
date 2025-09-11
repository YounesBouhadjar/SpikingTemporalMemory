import wandb
import sys 
import yaml


# get commmand line arguments
try:
    sweep_config = sys.argv[1]
except:
    print("provide simulation and params script!")
    exit(-1)

with open(sweep_config, 'r') as cfgfile:
    params_config = yaml.load(cfgfile, Loader=yaml.FullLoader)

wandb.login()

sweep_id = wandb.sweep(params_config)

#wandb.agent(sweep_id, generate_reference_data, count=500)
