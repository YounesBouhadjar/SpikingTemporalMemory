import copy
from matplotlib import gridspec
import matplotlib.pyplot as plt
import numpy as np

from shtm.helper import load_data, load_spike_data, load_numpy_spike_data
import utils

def get_handle_lists(l):
    """returns a list of lists of handles.
    """
    tree = l._legend_box.get_children()[1]

    for subpopulation in tree.get_children():
        for row in subpopulation.get_children():
            yield row.get_children()[0].get_children()

path_dict = {}
path_dict['data_root_path'] = 'data'
path_dict['project_name'] = 'sequence_learning_performance'
path_dict['parameterspace_label'] = 'sequence_learning_and_prediction'

########################
# load data
#----------------------

# get parameters 
PS, PS_path = utils.get_parameter_set(path_dict)

# get data of the first seed (1st network realization)
PS_sel = copy.deepcopy(PS)

params = utils.parameter_set_list(PS_sel)[0]
num_neurons = params['M'] * params['n_E']

# get training data
sequences = load_data(PS_path, 'training_data')
vocabulary = load_data(PS_path, 'vocabulary')

print("#### sequences used for training ### ")
for i, sequence in enumerate(sequences):
    seq = ''
    for char in sequence:
        seq += str(char).ljust(2)
    print("sequence %d: %s" % (i, seq))

# get data path
data_path = utils.get_data_path(params['data_path'], params['label'])

# load connections
connections_before_learning = load_data(data_path, 'ee_connections_before')
connections_after_learning = load_data(data_path, 'ee_connections')
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
panel_label_pos_up = (-0.18,1.15)
panel_label_pos = (-0.18,1.15)
color_seq = ['#66DCFFFF', '#044486FF']
s = 0.5
c = 0.95
pmat_color = '#e8e8e8'
mat_color = '#808080'

fig = plt.figure(1)
gs = gridspec.GridSpec(2, 2, left=0.1, right=0.98, bottom=0.07, top=0.93, wspace=0, hspace=0.4)

#####################################################
# panel A (placeholder for svg figure to be inserted)
# ---------------------------------------------------
plt.subplot(gs[0,0])
plt.axis('off')
title = 'before learning' 
utils.panel_label('A', panel_label_pos_up, title)

#####################################################
# panel C (placeholder for svg figure to be inserted)
# ---------------------------------------------------
plt.subplot(gs[1,0])
plt.axis('off')
title = 'after learning'
utils.panel_label('C', panel_label_pos, title)

######################################
# panel B connectivity before learning
# ------------------------------------
ax = plt.subplot(gs[0,1])
utils.panel_label('B', panel_label_pos_up)
# substracted 0.01 to differentiate between non-existing connections and connections with zero weight 
connection_matrix = np.zeros((num_neurons, num_neurons)) - 0.01     
th_mature_connections = params['syn_dict_ee']['Wmax'] - 0.5

# connection EE
for i, conn in enumerate(connections_before_learning):
    connection_matrix[int(conn[1])-1, int(conn[0])-1] = conn[2]

premature_connections = np.where((connection_matrix<th_mature_connections) & (connection_matrix>=0))
mature_connections = np.where(connection_matrix>th_mature_connections)
x_p = np.random.choice([False, True], len(premature_connections[0]), p=[c, 1-c])
x_m = np.random.choice([False, True], len(mature_connections[0]), p=[c, 1-c])

# plot premature connections
p1 = plt.scatter(premature_connections[0][x_p], premature_connections[1][x_p], s=s, marker='o', color=pmat_color, label='immature connections')

# plot mature connections
p2 = plt.scatter(mature_connections[0][x_m], mature_connections[1][x_m], s=s, marker='o', color=mat_color, zorder=1, label='mature connections')

#plt.scatter(premature_connections[0][x], premature_connections[1][x], s=s, marker='o', color=pmat_color)
plt.xlim(-1, num_neurons)
plt.ylim(-1, num_neurons)
ax.set_aspect('equal')

conn_matrix_ticks_pos = shifted_subpopulation_indices * params['n_E']
conn_matrix_ticks_label = chars_per_subpopulation

ax.set_xticks(conn_matrix_ticks_pos)
ax.set_xticklabels(conn_matrix_ticks_label)
ax.set_yticks(conn_matrix_ticks_pos)
ax.set_yticklabels(conn_matrix_ticks_label)

ticks_minor = (np.arange(1, params['M'])) * params['n_E'] 
ax.set_xticks(ticks_minor, minor=True)
ax.set_yticks(ticks_minor, minor=True)

# Gridlines based on minor ticks
ax.grid(which='minor', color='black', linestyle='-', linewidth=0.5)
ax.tick_params(axis=u'both', which=u'both',length=0)

ax.set_xlabel('source') 
ax.set_ylabel('target')
#ax.set_title('EE connections')

#####################################
# panel D connectivity after learning
# -----------------------------------
ax = plt.subplot(gs[1,1])
utils.panel_label('D', panel_label_pos)
# substracted 0.01 to differentiate between non-existing connections and connections with zero weight
connection_matrix = np.zeros((num_neurons, num_neurons)) - 0.01        

# connections EE
for conn in connections_after_learning:
    connection_matrix[int(conn[1])-1, int(conn[0])-1] = conn[2]

premature_connections = np.where((connection_matrix<th_mature_connections) & (connection_matrix>=0))
mature_connections = np.where(connection_matrix>th_mature_connections)
x_p = np.random.choice([False, True], len(premature_connections[0]), p=[c, 1-c])
x_m = np.random.choice([False, True], len(mature_connections[0]), p=[0, 1])

# plot premature connections, note that for clarity we plot only a small fraction of the connections not partcipating in the sequences (both mature and immature)
p1 = plt.scatter(premature_connections[0][x_p], premature_connections[1][x_p], s=s, marker='o', color=pmat_color, label='immature connections')

# plot mature connections
p2 = plt.scatter(mature_connections[0][x_m], mature_connections[1][x_m], s=s, marker='o', color=mat_color, zorder=1, label='mature connections')

# color connections belonging to the trained sequences
for ns, seq in enumerate(sequences):
    for char, char_next in zip(seq, seq[1:]):
        sources = letters_to_active_neurons[ns][char]
        targets = letters_to_active_neurons[ns][char_next]

        for src in sources:
            for tg in targets:
                plot_pt = np.random.choice([True, False], 1, p=[1, 0])
                if plot_pt and connection_matrix[int(src-1),int(tg-1)] > th_mature_connections:
                    p3 = plt.scatter(src-1, tg-1, s=s, marker='o', color=color_seq[ns],zorder=2)

plt.xlim(-1, num_neurons)
plt.ylim(-1, num_neurons)
ax.set_aspect('equal')

conn_matrix_ticks_pos = shifted_subpopulation_indices * params['n_E']
conn_matrix_ticks_label = chars_per_subpopulation

ax.set_xticks(conn_matrix_ticks_pos)
ax.set_xticklabels(conn_matrix_ticks_label)
ax.set_yticks(conn_matrix_ticks_pos)
ax.set_yticklabels(conn_matrix_ticks_label)

# Minor ticks
ticks_minor = (np.arange(1, params['M'])) * params['n_E']
ax.set_xticks(ticks_minor, minor=True)
ax.set_yticks(ticks_minor, minor=True)

# Gridlines based on minor ticks
ax.grid(which='minor', color='black', linestyle='-', linewidth=0.5)
ax.tick_params(axis=u'both', which=u'both',length=0)

ax.set_xlabel('source') 
ax.set_ylabel('target')

l=plt.legend([p1, p2], ['immature connections', 'mature connections'], 
                        bbox_to_anchor=(0.42, 1.18), loc='upper left', scatterpoints=3, scatteryoffsets=[0.3,0.3,0.3])
l.get_frame().set_alpha(None)

l.legendHandles[0]._sizes = [3]
l.legendHandles[1]._sizes = [3]

handles_list = list(get_handle_lists(l))
handles = handles_list[0] 
handles[0].set_facecolors(['none', 'none', pmat_color])
handles[0].set_edgecolors(['none', 'none', pmat_color])

handles = handles_list[1] 
handles[0].set_facecolors([mat_color, color_seq[0], color_seq[1]])
handles[0].set_edgecolors([mat_color, color_seq[0], color_seq[1]])

master_fname = 'connectivity_matrix'  
plt.savefig('%s.pdf' % (master_fname))
exit()

###########################################
# combine matplotlib figure with inkscape 
# -----------------------------------------

# add panel A
ext_file_name = 'sketches/network_unconnected.pdf'  # here: created using inkscape
composite_figure_name_root = 'composite_A'

# generate composite figure (with dimensions inherited from the master figure)
pos_ext_figure = (-3.7,3.37)      # position of external file in composite figure (center = (0,0))
utils.create_composite_figure(composite_figure_name_root, master_fname, ext_file_name, fig_size, pos_ext_figure)

# add panel C
master_fname = 'composite_A'
ext_file_name = 'sketches/network_connected.pdf'  # here: created using inkscape
comp_fname = 'network_structure'

# generate composite figure (with dimensions inherited from the master figure)
pos_ext_figure = (-3.7,-3.56)      # position of external file in composite figure (center = (0,0))
utils.create_composite_figure(comp_fname, master_fname, ext_file_name, fig_size, pos_ext_figure)
