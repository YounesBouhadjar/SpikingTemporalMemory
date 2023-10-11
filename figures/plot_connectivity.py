import os
import sys
import copy
from matplotlib import gridspec
import matplotlib.pyplot as plt
import numpy as np

from shtm.helper import load_data, load_spike_data, load_numpy_spike_data
import utils

path_dict = {}
path_dict['data_root_path'] = 'data'
path_dict['project_name'] = 'sequence_learning_performance'
path_dict['parameterspace_label'] = 'sequence_learning_and_prediction'

########################
# load data
#----------------------

# get parameters 
PS, PS_path = utils.get_parameter_set(path_dict)

# get data of the first PS_selseed (1st network realization)
PS_sel = copy.deepcopy(PS)

params = utils.parameter_set_list(PS_sel)[0]
num_neurons = params['M'] * params['n_E']
label = params['label']

# get trained sequences
# TODO load data training fails if the sequences are not of the same length
sequences = load_data(PS_path, f'{label}/training_data')
vocabulary = load_data(PS_path, f'{label}/vocabulary')

# get data path
data_path = utils.get_data_path(params['data_path'], params['label'])

print('#### sequences used for training ### ')
for i, sequence in enumerate(sequences): 
    seq = '' 
    for char in sequence:
        seq += str(char).ljust(2) 
    print('sequence %d: %s' % (i, seq))

# load connections
ee_connections = load_data(data_path, 'ee_connections')
characters_to_subpopulations = load_data(data_path, 'characters_to_subpopulations')

# load spiking data
somatic_spikes = load_numpy_spike_data(data_path, 'somatic_spikes')

# load excitation times
excitation_times = load_data(data_path, 'excitation_times')

# select spiking data of the last trial 
#TODO here we assume that the sequences are of the same length
number_elements_per_batch = len(sequences) * len(sequences[0])
start_time = excitation_times[-number_elements_per_batch] - params['pad_time'] 
end_time = excitation_times[-1] + params['pad_time']

# select data corresponding to the last training episode
idx_somatic_spikes = np.where((somatic_spikes[:,1] > start_time) & (somatic_spikes[:,1] < end_time))
somatic_spikes_times = somatic_spikes[:,1][idx_somatic_spikes]
somatic_spikes_senders = somatic_spikes[:,0][idx_somatic_spikes]

idx_exc_times = np.where((excitation_times > start_time) & (excitation_times < end_time))
excitation_times_sel = excitation_times[idx_exc_times]

# find the active neurons corresponding to each letter in each sequence
response_delay = 2*params['fixed_somatic_delay']
letters_to_active_neurons = utils.letters_to_active_neurons(sequences, somatic_spikes_times, somatic_spikes_senders, excitation_times_sel, response_delay)

connections_after_learning = ee_connections
# connections EE
th_mature_connections = 1. #PS_sel['syn_dict_ee']['th_perm']
connection_matrix = np.zeros((num_neurons, num_neurons))         
for conn in connections_after_learning:
    connection_matrix[int(conn[1])-1, int(conn[0])-1] = conn[2]

# ##################
# plot routing
# ------------------

# plot settings 
fig_size = (5.2, 5.5)
plt.rcParams['font.size'] = 8
plt.rcParams['legend.fontsize'] = 6
plt.rcParams['figure.figsize'] = fig_size
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['text.usetex'] = False
panel_label_pos_up = (-0.18,1.0)
panel_label_pos = (-0.18,1.0)
color_seq = ['#66DCFFFF', '#044486FF', 'red', 'orange']
s = 0.5
c = 1.
pmat_color = '#e8e8e8'
mat_color = '#808080'

# organize the characters for plotting purpose
subpopulation_indices = []
chars_per_subpopulation = []
for char in vocabulary:
    # shift the subpopulation indices for plotting purposes 
    char_to_subpopulation_indices = characters_to_subpopulations[char]
    subpopulation_indices.extend(char_to_subpopulation_indices)

    #sub_subpopulations.extend(h['characters_to_subpopulations'][char])
    chars_per_subpopulation.extend(char * len(characters_to_subpopulations[char]))

shifted_subpopulation_indices = np.array(subpopulation_indices) + 0.5

premature_connections = np.where((connection_matrix<th_mature_connections) & (connection_matrix>=0))
mature_connections = np.where(connection_matrix>th_mature_connections)
x_p = np.random.choice([False, True], len(premature_connections[0]), p=[c, 1-c])
x_m = np.random.choice([False, True], len(mature_connections[0]), p=[0, 1])

# plot premature connections, note that for clarity we plot only a small fraction of the connections not partcipating in the sequences (both mature and immature)
p1 = plt.scatter(premature_connections[0][x_p], premature_connections[1][x_p], s=s, marker='o', color=pmat_color, label='immature connections')

# plot mature connections
p2 = plt.scatter(mature_connections[0][x_m], mature_connections[1][x_m], s=s, marker='o', color=mat_color, zorder=1, label='mature connections')

#
plt.xlim(-1, num_neurons)
plt.ylim(-1, num_neurons)

#scale = num_selected_neurons 
scale = params['n_E']
conn_matrix_ticks_pos = shifted_subpopulation_indices
conn_matrix_ticks_label = chars_per_subpopulation

plt.xticks(conn_matrix_ticks_pos*scale, conn_matrix_ticks_label)
plt.yticks(conn_matrix_ticks_pos*scale, conn_matrix_ticks_label)

# Minor ticks
ticks_minor = (np.arange(1, params['M'])) * scale

plt.xlabel('source') 
plt.ylabel('target')

print('--------------------------------------------------')
path = 'img'
fname = 'connectivity_matrix'
print('save here: %s/%s.pdf ...' % (path, fname))
os.system('mkdir -p %s' % (path))
plt.savefig('%s/%s.pdf' % (path, fname))
