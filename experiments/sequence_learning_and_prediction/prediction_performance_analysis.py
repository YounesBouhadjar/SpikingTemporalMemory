#!/bin/bash
#
# This file is part of shtm.
#
# shtm is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# shtm is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with shtm.  If not, see <https://www.gnu.org/licenses/>.
#

"""
This script loads the spiking data and assesses the prediction performance

Authors
~~~~~~~
Younes Bouhadjar
"""

import os
import sys 
import copy
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

from shtm import helper
 
path_dict = {} 
path_dict['data_root_path'] = 'data'
path_dict['project_name'] = 'sequence_learning_performance' 
path_dict['parameterspace_label'] = 'sequence_learning_and_prediction'

# get parameters 
PS, PS_path = helper.get_parameter_set(path_dict)

PS_sel = copy.deepcopy(PS)
compute_overlap = True

PL = helper.parameter_set_list(PS_sel)

# get training data
sequences = helper.load_data(PS_path, 'training_data')

print("#### sequences used for training ### ")
for i, sequence in enumerate(sequences): 
    seq = '' 
    for char in sequence:
        seq += str(char).ljust(2) 
    print("sequence %d: %s" % (i, seq))

fname = 'prediction_performance'
for cP, params in enumerate(PL):

    data = {}

    # get data path
    data_path = helper.get_data_path(params['data_path'], params['label'])
    print("\t\t data set %d/%d: %s/%s" % (cP + 1, len(PL), data_path, fname))

    # load somatic spikes and dendritic current
    somatic_spikes = helper.load_spike_data(data_path, 'somatic_spikes')
    idend_eval = helper.load_spike_data(data_path, 'idend_eval')

    # load record and excitation times 
    idend_recording_times = helper.load_data(data_path, 'idend_recording_times')
    characters_to_subpopulations = helper.load_data(data_path, 'characters_to_subpopulations')
    excitation_times = helper.load_data(data_path, 'excitation_times')

    # compute prediction performance
    errors, false_positives, false_negatives, num_active_neurons = helper.compute_prediction_performance(somatic_spikes, idend_eval, idend_recording_times, characters_to_subpopulations, sequences, params)

    if compute_overlap:
        # sequences overlap
        sequences_overlap = helper.measure_sequences_overlap(sequences, somatic_spikes[:,1], somatic_spikes[:,0], excitation_times, params['fixed_somatic_delay'], params['learning_episodes'])
        data['overlap'] = sequences_overlap

    data['error'] = errors
    data['false_positive'] = false_positives
    data['false_negative'] = false_negatives
    data['rel_active_neurons'] = num_active_neurons/params['n_E']
    data['ep_num'] = params['episodes_to_testing'] * np.arange(int(params['learning_episodes']/params['episodes_to_testing'])+1)

    ep_to_sol = np.where(errors < 0.01)[0] 
    if len(ep_to_sol) == 0:
        print("number of episodes to convergence", params['learning_episodes'])
    else:    
        print("number of episodes to convergence", data['ep_num'][ep_to_sol][0])

    # save data
    np.save("%s/%s" % (data_path, fname), data)
