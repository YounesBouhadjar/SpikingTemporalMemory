from spikingtemporalmemory import default_params
import parameters as para

DELAY = 0.1
MODEL = 'iaf_psc_exp'

p = default_params.p

# data path dict
p['data_path'] = {}
p['data_path']['data_root_path'] = 'data'
p['data_path']['project_name'] = 'sequence_learning_performance'
p['data_path']['parameterspace_label'] = 'sequence_learning_and_prediction_task_complexity_2'

p['seed'] = para.ParameterRange([11, 12, 13])           # seed for NEST

# task parameters
p['task'] = {}
p['task']['vocabulary_size'] = 26                  # vocabulary size
p['task']['S'] = 1                                 # number of sequences
p['task']['C'] = 40                                # sequence length
p['task']['R'] = 0                                 # number of shared subsequences
p['task']['O'] = 0                                 # length of shared subsequences ("order")
p['task']['seed'] = 15                             # seed number

# setup the training loop  
p['learning_episodes'] = 100                       # total number of training episodes
