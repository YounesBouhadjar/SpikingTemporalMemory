import os
import sys 
import copy
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from collections import defaultdict

from shtm.helper import load_data
import utils
 
path_dict = {} 
path_dict['data_root_path'] = 'data'
path_dict['project_name'] = 'sequence_learning_performance' 

try: 
    pars_space = sys.argv[1]
except:
    pars_space = 'task1'

if pars_space == 'task2':
    path_dict['parameterspace_label'] = 'prediction_performance_task2' 
elif pars_space == 'task3':
    path_dict['parameterspace_label'] = 'prediction_performance_task3'
elif pars_space == 'task2_stdp':
    path_dict['parameterspace_label'] = 'prediction_performance_task2_stdp'
elif pars_space == 'task4':
    path_dict['parameterspace_label'] = 'prediction_performance_task4'
else:
    path_dict['parameterspace_label'] = 'sequence_learning_and_prediction'

# get parameters
# --------------
PS, PS_path = utils.get_parameter_set(path_dict)

PS_sel = copy.deepcopy(PS)
PL = utils.parameter_set_list(PS_sel)

# get parameters of first network realization
params = PL[0]
compute_overlap = True

# get training data
sequences = load_data(PS_path, 'training_data')

print("#### sequences used for training ### ")
for i, sequence in enumerate(sequences): 
    seq = '' 
    for char in sequence:
        seq += str(char).ljust(2) 
    print("sequence %d: %s" % (i, seq))

# load data
# ---------
fname = "prediction_performance"
data_diff_run = defaultdict(list)
for cP, params in enumerate(PL):

    # get data path
    data_path = utils.get_data_path(params['data_path'], params['label'])
    print("\t\t data set %d/%d: %s/%s" % (cP + 1, len(PL), data_path, fname))

    # load prediction performance
    data = load_data(data_path, fname)

    data_diff_run['error'].append(data['error'])

    data_diff_run['false_positive'].append(data['false_positive'])
    data_diff_run['false_negative'].append(data['false_negative'])
    data_diff_run['rel_active_neurons'].append(data['rel_active_neurons'])
    data_diff_run['overlap'].append(data['overlap'])

# plot settings 
# -------------
plt.rcParams['font.size'] = 8
plt.rcParams['legend.fontsize'] = 6
plt.rcParams['figure.figsize'] = (7.,2.)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['text.usetex'] = False
panel_label_pos = (-0.1,1.0)

# get x and z scans
x = data['ep_num']
xlabel = 'number of training episodes'
data_diff_run['ep_num'] = x

# store postprocessed
fname = "postprocessed_prediction_performance"
print("save data to %s/%s" % (PS_path, fname))
np.save("%s/%s" % (PS_path, fname), data_diff_run)

# plot prediction performance
# ---------------------------
path = '.'
s = params['pattern_size'] / params['n_E']   # sparsity level
utils.plot_prediction_performance(x, data_diff_run, s, saving_paths=[path], figure_name='%s_%s' % ('prediction_performance', pars_space), xmax=params['learning_episodes']-5)

if compute_overlap:

    # plot sequences overlap of the first network realization
    sequences_overlap = np.array(data_diff_run['overlap'])

    # get overlap for a letter
    letter_overlaps = sequences_overlap[:, :, -2] 
    x = np.arange(params["learning_episodes"])
    y = utils.mean_confidence_interval(letter_overlaps)

    ylabel = 'size overlap subpopulation B'
    title = 'sequences = [ADBE, FDBC]'
    utils.plot_data(x, y, xlabel=xlabel, ylabel=ylabel, title = title, saving_paths=[PS_path], figure_name='size_overlap')
