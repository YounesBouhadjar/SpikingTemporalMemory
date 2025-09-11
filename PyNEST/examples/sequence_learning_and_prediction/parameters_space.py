import parameters as para
import numpy as np

DELAY = 0.1
MODEL = 'iaf_psc_exp'

p = para.ParameterSpace({})

# data path dict
p['data_path'] = {}
p['data_path']['data_root_path'] = 'data'
p['data_path']['project_name'] = 'sequence_learning_performance'
p['data_path']['parameterspace_label'] = 'sequence_learning_and_prediction_task_complexity_2'

# parameters for setting up the network  
p['M'] = 26                  # number of subpopulations. 

p['n_E'] = 240               # number of excitatory neurons per subpopulation
p['n_I'] = 1                 # number of inhibitory neurons per subpopulation
p['L'] = 1                   # number of subpopulations that represents one sequence element
p['pattern_size'] = 20       # sparse set of active neurons per subpopulation

# connection details
p['rule'] = 'fixed_indegree'                          
p['connection_prob'] = 0.15

# neuron parameters of the excitatory neurons
p['soma_model'] = 'iaf_psc_exp_nonlineardendrite_neuron'
p['soma_params'] = {}
p['soma_params']['C_m'] = 250.        # membrane capacitance (pF)
p['soma_params']['E_L'] = 0.          # resting membrane potential (mV)
# p['soma_params']['I_e'] = 0.        # external DC currents (pA)
p['soma_params']['V_m'] = 0.          # initial potential (mV)
p['soma_params']['V_reset'] = 0.      # reset potential (mV)
p['soma_params']['V_th'] = 20.        # spike threshold (mV)
p['soma_params']['t_ref'] = 10.       # refractory period
p['soma_params']['tau_m'] = 10.       # membrane time constant (ms)
p['soma_params']['tau_syn1'] = 2.     # synaptic time constant: external input (receptor 1)
p['soma_params']['tau_syn2'] = 5.     # synaptic time constant: dendrtic input (receptor 2)
p['soma_params']['tau_syn3'] = 1.     # synaptic time constant: inhibitory input (receptor 3)
# dendritic action potential
p['soma_params']['I_p'] = 200.        # current clamp value for I_dAP during a dendritic action potenti
p['soma_params']['tau_dAP'] = 60.     # time window over which the dendritic current clamp is active
p['soma_params']['theta_dAP'] = 59.   # current threshold for a dendritic action potential
p['fixed_somatic_delay'] = 2          # this is an approximate time of how long it takes the soma to fire
                                      # upon receiving an external stimulus 

# neuron parameters for the inhibitory neuron
p['inhibit_model'] = 'iaf_psc_exp'
p['inhibit_params'] = {}
p['inhibit_params']['C_m'] = 250.         # membrane capacitance (pF)
p['inhibit_params']['E_L'] = 0.           # resting membrane potential (mV)
p['inhibit_params']['I_e'] = 0.           # external DC currents (pA)
p['inhibit_params']['V_m'] = 0.           # initial potential (mV)
p['inhibit_params']['V_reset'] = 0.       # reset potential (mV)
p['inhibit_params']['V_th'] = 15.         # spike threshold (mV)
p['inhibit_params']['t_ref'] = 2.0        # refractory period
p['inhibit_params']['tau_m'] = 5.         # membrane time constant (ms)
p['inhibit_params']['tau_syn_ex'] = .5    # synaptic time constant of an excitatory input (ms) 
p['inhibit_params']['tau_syn_in'] = 1.65  # synaptic time constant of an inhibitory input (ms)

# synaptic parameters
p['J_EX_psp'] = 1.1 * p['soma_params']['V_th']     # somatic PSP as a response to an external input
p['J_EI_psp'] = -2 * p['soma_params']['V_th']      # somatic PSP as a response to an inhibitory input
p['convergence'] = 5

# parameters for ee synapses (stdsp)
p['syn_dict_ee'] = {}
p['p_min'] = 0.
p['p_max'] = 8.
p['calibration'] = 0.
p['syn_dict_ee']['weight'] = 0.                      # synaptic weight
p['syn_dict_ee']['delay'] = 2.0
p['syn_dict_ee']['synapse_model'] = 'stdsp_synapse'  # synapse model
p['syn_dict_ee']['th_perm'] = 10.                    # synapse maturity threshold
p['syn_dict_ee']['tau_perm'] = 70000                 # time constant of permanence leakage (50000 (20) 80000 (40) ,50000 or 70000 (40)  70000 (60))
p['syn_dict_ee']['tau_tr_pre'] = 20.                 # plasticity time constant (potentiation)
p['syn_dict_ee']['tau_tr_post'] = 20. 
p['syn_dict_ee']['receptor_type'] = 2                # receptor corresponding to the dendritic input
p['syn_dict_ee']['lambda'] = 0.6                     # potentiation rate ( 0.6 (20) 0.6 (40) 0.7 (40), 0.6 (60))
p['syn_dict_ee']['mu_plus'] = 0.                     # permanence dependence exponent, potentiation
# p['syn_dict_ee']['mu_minus'] = 0.                  # permanence dependence exponent, depression
p['syn_dict_ee']['Wmax'] = 1.1 * p['soma_params']['theta_dAP'] / p['convergence']   # Maximum allowed weight
p['syn_dict_ee']['Pmax'] = 20.                       # Maximum allowed permanence
p['syn_dict_ee']['Pmin'] = 1.                        # Minimum allowed permanence
p['syn_dict_ee']['lambda_minus'] = 0.02              # depression rate (# 0.8 (20) 0.1 (40)  0.3 (40), 0.02 (60))
p['syn_dict_ee']['dt_min'] = 4.                      # minimum time lag of the STDP window
p['inh_factor'] = 7.
p['p_target'] = 0
p['syn_dict_ee_synapse_model'] = p['syn_dict_ee']['synapse_model']

# parameters of EX synapses (external to soma of E neurons)
p['conn_dict_ex'] = {}
p['syn_dict_ex'] = {}
p['syn_dict_ex']['receptor_type'] = 1                    # receptor corresponding to external input
p['syn_dict_ex']['delay'] = DELAY                        # dendritic delay
p['conn_dict_ex']['rule'] = 'all_to_all'                 # connection rule

# parameters of EdX synapses (external to dendrite of E neurons) 
p['conn_dict_edx'] = {}
p['syn_dict_edx'] = {}
p['syn_dict_edx']['receptor_type'] = 2                    # receptor corresponding to the dendritic input
p['syn_dict_edx']['delay'] = DELAY                        # dendritic delay
p['syn_dict_edx']['weight'] = 1.4 * p['soma_params']['theta_dAP']
p['conn_dict_edx']['rule'] = 'fixed_outdegree'            # connection rule
p['conn_dict_edx']['outdegree'] = p['pattern_size'] + 10   # outdegree

# parameters for IE synapses 
p['syn_dict_ie'] = {}
p['conn_dict_ie'] = {}
p['syn_dict_ie']['synapse_model'] = 'static_synapse'     # synapse model
p['syn_dict_ie']['delay'] = DELAY                        # dendritic delay

# parameters for EI synapses
p['syn_dict_ei'] = {}
p['conn_dict_ei'] = {}
p['syn_dict_ei']['synapse_model'] = 'static_synapse'     # synapse model
p['syn_dict_ei']['delay'] = DELAY                        # dendritic delay
p['syn_dict_ei']['receptor_type'] = 3                    # receptor corresponding to the inhibitory input  

# stimulus parameters
p['DeltaT'] = 40.                     # inter-stimulus interval
p['excitation_start'] = 30.           # time at which the external stimulation begins
p['time_dend_to_somatic'] = 20.       # time between the dAP activation and the somatic activation (only used if sparse_first_char is True)   
p['DeltaT_cue'] = 80.                 # inter-cue interval during replay

# simulation parameters 
p['dt'] = 0.1                                  # simulation time resolution (ms)
p['overwrite_files'] = True                    # if True, data will be overwritten,
                                               # if False, a NESTError is raised if the files already exist
#p['seed'] = para.ParameterRange([10, 20, 30])           # seed for NEST
p['seed'] = para.ParameterRange([11, 12, 13])           # seed for NEST
p['print_simulation_progress'] = False         # print the time progress.
p['n_threads'] = 16                             # number of threads per MPI process 
p['pad_time'] = 5.
p['idend_recording_interval'] = 10 * p['dt']   # dendritic current recording resolution
p['idend_record_time'] = 8.                    # time interval after the external stimulation at which the dendritic current is recorded
p['evaluate_performance'] = True               # if turned on, we monitor the dendritic current at a certain time steps
                                               # during the simulation. This then is used for the prediction performance assessment
p['evaluate_replay'] = False                     
p['record_idend_last_episode'] = True          # used for debugging, if turned on we record the dendritic current of all neurons
                                               # this can consume too much memory
p['store_training_data'] = True                                               
p['store_connections'] = False
p['load_connections'] = False
p['sparse_first_char'] = True                  # if turned on, the dAP of a subset of neurons in the subpopulation representing 
                                               # first sequence elements is activated externally 
p['active_weight_recorder'] = False            # if turned on, the weights are recorded every presynaptic spike

# task parameters
p['task'] = {}
p['task']['vocabulary_size'] = 26                  # vocabulary size
p['task']['S'] = 1                                 # number of sequences
p['task']['C'] = 40                                # sequence length
p['task']['R'] = 0                                 # number of shared subsequences
p['task']['O'] = 0                                 # length of shared subsequences ("order")
p['task']['seed'] = 15                             # seed number

# setup the training loop  
p['learning_episodes'] = 100                  # total number of training episodes ('repetitions of the sequence sets')
p['episodes_to_testing'] = 100                # number of episodes after which we measure the prediction perfomance
