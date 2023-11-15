import copy
from matplotlib import gridspec
import matplotlib.pyplot as plt
import numpy as np
import sys

sys.path.append('./../shtm')
from helper import load_data, load_numpy_spike_data, load_spike_data
import utils

# retrieve network parameters 
path_dict = {}
path_dict['data_root_path'] = 'data'
path_dict['project_name'] = 'sequence_learning_performance'
path_dict['parameterspace_label'] = 'sequence_learning_and_prediction'

# get parameters 
PS, PS_path = utils.get_parameter_set(path_dict)
PS_sel = copy.deepcopy(PS)

params = utils.parameter_set_list(PS_sel)[0]
num_neurons = params['M'] * params['n_E']

# get trained sequences and vocabulary
sequences = load_data(PS_path, 'training_data')
vocabulary = load_data(PS_path, 'vocabulary')

print('#### sequences used for training ### ')
for i, sequence in enumerate(sequences):
    seq = ''
    for char in sequence:
        seq += str(char).ljust(2)
    print('sequence %d: %s' % (i, seq))

# get data path
data_path = utils.get_data_path(params['data_path'], params['label'])

# load spikes
somatic_spikes = load_numpy_spike_data(data_path, 'somatic_spikes')
inh_spikes = load_numpy_spike_data(data_path, 'inh_spikes')
idend = load_numpy_spike_data(data_path, 'idend_last_episode')

# load spike recordings
if params['evaluate_performance']:
    idend_recording_times = load_data(data_path, 'idend_recording_times')
characters_to_subpopulations = load_data(data_path, 'characters_to_subpopulations')
characters_to_time_excitation = load_data(data_path, 'excitation_times_soma')

# load excitation times
excitation_times = load_data(data_path, 'excitation_times')

# get dendritic AP
idx = np.where((idend[:, 2] > params['soma_params']['theta_dAP']))[0]
dendriticAP_currents = idend[:, 2][idx]
dendriticAP_times = idend[:, 1][idx]
dendriticAP_senders = idend[:, 0][idx]


start_time = characters_to_time_excitation[sequences[0][0]][-1] - params['pad_time']
end_time = characters_to_time_excitation[sequences[-1][-1]][-1] + params['pad_time']
print(start_time, end_time)
plt.rcParams["figure.figsize"] = (15,10)
plt.rcParams["font.size"] = 20
utils.plot_spikes(somatic_spikes, inh_spikes, idend, start_time, end_time, params['soma_params']['theta_dAP'], 150, 12)
plt.savefig("network_activity_last_episode.png")

sys.exit()
# organize the characters for plotting purpose
subpopulation_indices = []
chars_per_subpopulation = []
for char in vocabulary:
    # shift the subpopulation indices for plotting purposes 
    char_to_subpopulation_indices = characters_to_subpopulations[char]
    subpopulation_indices.extend(char_to_subpopulation_indices)

    chars_per_subpopulation.extend(char * len(characters_to_subpopulations[char]))

shifted_subpopulation_indices = np.array(subpopulation_indices) + 0.5

# ####################################################
# plotting routing
# ----------------------------------------------------

# plot settings 
fig_size = (5.2, 5.7)
plt.rcParams['font.size'] = 8
plt.rcParams['legend.fontsize'] = 6
plt.rcParams['figure.figsize'] = fig_size
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['text.usetex'] = False

create_figure = True
panel_label_pos = (-0.14,0.5)
panel_labels = ['B', 'D', 'F']
color_soma_spike = '#DB2763'
color_dendrite_spike = '#00B4BE' 
fc_bg = '#dcdcdc'
fraction_active = 3
delta_time = 5.
ymin = -0.1
ymax = 2
xmin = 0
master_file_name = 'network_activity'

if create_figure:

    # set up the figure frame
    fig = plt.figure()
    gs = gridspec.GridSpec(6, 2, height_ratios=[3, 15, 3, 15, 3, 15], bottom=0.07, right=0.95, top=1., wspace=0., hspace=0.)

    # panel A (placeholder for svg figure to be inserted; see below)
    panel_label_pos_shift = (-0.26, 0.5)
    plt.subplot(gs[0, 0])
    plt.axis('off')
    utils.panel_label('A', panel_label_pos_shift, 'before learning')

    # panel C (placeholder for svg figure to be inserted; see below)
    plt.subplot(gs[2, 0])
    plt.axis('off')
    utils.panel_label('C',panel_label_pos_shift, 'after learning')

    # panel C (placeholder for svg figure to be inserted; see below)
    plt.subplot(gs[4, 0])
    plt.axis('off')
    utils.panel_label('E', panel_label_pos_shift,)

    for j, (learn, seq_num) in enumerate(zip([False, True, True], [0, 0, 1])):

        ###################################
        # postprocessing of data
        # ---------------------------------
        if learn and seq_num == 0:
            start_time = characters_to_time_excitation[sequences[0][0]][-1] - params['pad_time']
            end_time = characters_to_time_excitation[sequences[0][-1]][-1] + params['pad_time']
        elif learn and seq_num == 1:
            start_time = characters_to_time_excitation[sequences[1][0]][-1] - params['pad_time']
            end_time = characters_to_time_excitation[sequences[1][-1]][-1] + params['pad_time']
        else:
            start_time = characters_to_time_excitation[sequences[0][0]][0] - params['pad_time']
            end_time = characters_to_time_excitation[sequences[0][-1]][0] + params['pad_time']

        # select data  corresponding to the different sequences
        idx_somatic_spikes = np.where((somatic_spikes[:,1] > start_time) & (somatic_spikes[:,1] < end_time))
        idx_dAP = np.where((dendriticAP_times > start_time) & (dendriticAP_times < end_time))

        # postprocess somatic spikes
        somatic_spikes_times = somatic_spikes[:,1][idx_somatic_spikes]
        somatic_spikes_senders = somatic_spikes[:,0][idx_somatic_spikes]
        initial_time = somatic_spikes_times[0]
        somatic_spikes_times -= initial_time
        xmax = somatic_spikes_times[-1] + delta_time

        # postporcess dendritic AP
        dAP_senders = dendriticAP_senders[idx_dAP]
        dAP_currents = dendriticAP_currents[idx_dAP]
        dAP_times = dendriticAP_times[idx_dAP]
        dAP_times -= initial_time

        idx_exc_times = np.where((excitation_times > start_time) & (excitation_times < end_time))
        excitation_times_sel = excitation_times[idx_exc_times]

        # ###############################
        # draw stimulus
        # -------------------------------
        plt.subplot(gs[2*j, 1])
        utils.panel_label(panel_labels[j],panel_label_pos)
        plt.axis('off')

        for i in range(len(sequences[seq_num])): 
          
            x = (excitation_times_sel[i]+delta_time-initial_time) / (xmax+delta_time)
            y = 0.26
            arrow_width = 0.03
            arrow_height = 0.2

            pos = [x, y]
            X = np.array([pos, [pos[0]+arrow_width, pos[1]], [pos[0]+arrow_width/2, pos[1]-arrow_height]])
            t1 = plt.Polygon(X, color='black')
            plt.gca().add_patch(t1)
            #plt.text(pos[0]+arrow_width/8, pos[1]+0.5, sequences[seq_num][i])
            plt.text(pos[0]-0.003, pos[1]+0.1, sequences[seq_num][i])

        # ###############################
        # show soma and dendritic spikes
        # -------------------------------  
        plt.subplot(gs[2*j+1, 1])

        senders_subsampled = somatic_spikes_senders[::fraction_active]
        line1 = plt.plot(somatic_spikes_times[::fraction_active], somatic_spikes_senders[::fraction_active], 'o', color=color_soma_spike, lw=0., ms=0.5, zorder=2)

        #for k,v in count_indices_ds.items():
        for sender in senders_subsampled:
            idx_sub = np.where(dAP_senders == sender)
            line2 = plt.plot(dAP_times[idx_sub], dAP_senders[idx_sub], color=color_dendrite_spike, lw=1., zorder=1)

        plt.xlim(-delta_time, xmax)
        plt.ylim(-10, num_neurons+10)

        ticks_pos = shifted_subpopulation_indices * params['n_E']
        ticks_label = chars_per_subpopulation
        subpopulation_indices_background = np.arange(params['M'])*params['n_E']

        plt.yticks(ticks_pos, ticks_label)
        plt.tick_params(labelbottom=False)

        for i in range(params['M'])[::2]:
            plt.axhspan(subpopulation_indices_background[i], subpopulation_indices_background[i]+params['n_E'], facecolor=fc_bg, zorder=0)

        if j == 2:
            plt.xlabel('time (ms)')
            plt.tick_params(labelbottom=True)

        if j == 0:
            labels = ['somatic spikes', 'dendritic AP']
            l=plt.legend((line1[0], line2[0]), labels)    
            l.get_frame().set_alpha(None)

    plt.savefig('%s.pdf' % (master_file_name))

###########################################
# combine matplotlib figure with inkscape 
# -----------------------------------------

# add panel A
ext_file_name = 'sketches/network_activity_before_learning.pdf'  # here: created using inkscape
composite_figure_name_root = 'composite_A'

# generate composite figure (with dimensions inherited from the master figure)
x = -3.5
y = 4.7
jump = 4.5
pos_ext_figure = (x,y)      # position of external file in composite figure (center = (0,0)
utils.create_composite_figure(composite_figure_name_root, master_file_name, ext_file_name, fig_size, pos_ext_figure)

# add panel C
master_file_name = 'composite_A'
ext_file_name = 'sketches/network_activity_after_learning.pdf'  # here: created using inkscape
composite_figure_name_root = 'composite_C'

# generate composite figure (with dimensions inherited from the master figure)
pos_ext_figure = (x,y-jump)      # position of external file in composite figure (center = (0,0)
utils.create_composite_figure(composite_figure_name_root, master_file_name, ext_file_name, fig_size, pos_ext_figure)

# add panel D
master_file_name = 'composite_C'
ext_file_name = 'sketches/network_activity_after_learning_seqnum_1.pdf'  # here: created using inkscape
composite_figure_name_root = 'composite_E'

# store figure
path = '.'
fname = 'network_activity'
comp_fname = '%s/%s' % (path, fname)

# generate composite figure (with dimensions inherited from the master figure)
pos_ext_figure = (x,y-2*jump)      # position of external file in composite figure (center = (0,0)
utils.create_composite_figure(comp_fname, master_file_name, ext_file_name, fig_size, pos_ext_figure)
