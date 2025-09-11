from spikingtemporalmemory import default_params
import parameters as para

DELAY = 0.1
MODEL = 'iaf_psc_exp'

p = default_params.p

# data path dict
p['data_path'] = {}
p['data_path']['data_root_path'] = 'data'
p['data_path']['project_name'] = 'sequence_learning_performance'
p['data_path']['parameterspace_label'] = 'sequence_learning_and_prediction'

# network parameters
p['M'] = 26                  # number of subpopulations. 

p['n_E'] = 240               # number of excitatory neurons per subpopulation
p['n_I'] = 1                 # number of inhibitory neurons per subpopulation
p['L'] = 1                   # number of subpopulations that represents one sequence element
p['pattern_size'] = 20       # sparse set of active neurons per subpopulation

# task parameters
p['task'] = {}
p['task']['vocabulary_size'] = 26                  # vocabulary size
p['task']['S'] = 1                                 # number of sequences
p['task']['C'] = 40                                # sequence length
p['task']['R'] = 0                                 # number of shared subsequences
p['task']['O'] = 0                                 # length of shared subsequences ("order")
p['task']['seed'] = 15                             # seed number
p['task']['seq_set_instance_size'] = 10         # sequence set instance size
p['task']['subset_size'] = None                    # subset size for sequences
p['task']['order'] = 'fixed'                       # sequence order type
p['task']['seq_activation_type'] = 'consecutive'   # sequence activation type ('consecutive', 'parallel')

# simulation parameters
p['start'] = 100.                                  # simulation start time
p['stop'] = 5000000.                               # simulation stop time
p['seed'] = para.ParameterRange([11, 12, 13])           # seed for NEST

# training control parameters
p['record_ts'] = True                              # record timestamps
p['early_abort'] = True                            # enable early abort functionality
p['early_break'] = True                            # enable early break functionality
p['K'] = 10                                        # performance assessment interval
p['M_abort'] = 20                                  # abort threshold
p['min_error'] = 0.01                              # minimum error threshold
