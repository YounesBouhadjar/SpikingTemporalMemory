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
This script instantiates the Spiking-TemporalMemory model, creates, connects, and simulates the network 

Authors
~~~~~~~
Younes Bouhadjar
"""

import nest
import sys
import time
import numpy as np

from shtm import model, helper

def generate_reference_data():

    # ###########################################################
    # import nestml module
    # ===========================================================
    nest.Install('../../module/nestml_active_dend_module')

    #############################################################
    # get network and training parameters 
    # ===========================================================
   
    #TODO: use argparse with default values
    try: 
        batch_id=int(sys.argv[1])
        batch_array_id=int(sys.argv[2])
        JOBMAX=int(sys.argv[3])
        pars_space = sys.argv[4]
        array_id=batch_id*JOBMAX+batch_array_id
    except:
        array_id = 0
        pars_space='task2'

    if pars_space == 'parameters_space_task2.py':
        import parameters_space_task2 as pars 
    elif pars_space == 'parameters_space_task3.py':
        import parameters_space_task3 as pars 
    elif pars_space == 'parameters_space_task2_stdp.py':
        import parameters_space_task2_stdp as pars 
    elif pars_space == 'parameters_space_task4.py':
        import parameters_space_task4 as pars 
    else:
        import parameters_space as pars 

    # get parameter space
    PS = pars.p

    # get parameter list
    PL = helper.parameter_set_list(PS) 

    params = PL[array_id]
    #params['idend_recording_interval'] = params['dt']

    # start time 
    time_start = time.time()

    # ###############################################################
    # specify sequences
    # ===============================================================
    sequences, _, vocabulary = helper.generate_sequences(params['task'], params['data_path'], params['label'])

    # ###############################################################
    # create network
    # ===============================================================
    model_instance = model.Model(params, sequences, vocabulary)
    time_model = time.time()

    model_instance.create()
    time_create = time.time()

    # ###############################################################
    # connect the netwok
    # ===============================================================
    model_instance.connect()
    time_connect = time.time()

    # ###############################################################
    # simulate the network
    # ===============================================================
    model_instance.simulate()
    time_simulate = time.time()

    if params['store_connections']:
        model_instance.save_connections()

    print(
        '\nTimes of Rank {}:\n'.format(
            nest.Rank()) +
        '  Total time:          {:.3f} s\n'.format(
            time_simulate -
            time_start) +
        '  Time to initialize:  {:.3f} s\n'.format(
            time_model -
            time_start) +
        '  Time to create:      {:.3f} s\n'.format(
            time_create -
            time_model) +
        '  Time to connect:     {:.3f} s\n'.format(
            time_connect -
            time_create) +
        '  Time to simulate:    {:.3f} s\n'.format(
            time_simulate -
            time_connect))

    # display prediction performance after learning    
    if params['evaluate_performance']:
    
        data_path = helper.get_data_path(model_instance.params['data_path'], model_instance.params['label'])

        #print('-------------------------------------')
        #print([nest.GetStatus(nest.NodeCollection([i+1 for i in range(1000)]))[i]['z'] for i in range(990)])

        # load spikes from reference data
        somatic_spikes = helper.load_spike_data(data_path, 'somatic_spikes')
        idend_eval = helper.load_spike_data(data_path, 'idend_eval')
        excitation_times = helper.load_data(data_path, 'excitation_times')

        # get recoding times of dendriticAP
        idend_recording_times = helper.load_data(data_path, 'idend_recording_times')
        characters_to_subpopulations = helper.load_data(data_path, 'characters_to_subpopulations')

        seq_avg_errors, seq_avg_false_positives, seq_avg_false_negatives, _ = helper.compute_prediction_performance(somatic_spikes, idend_eval, idend_recording_times, characters_to_subpopulations, model_instance.sequences, model_instance.params)

        # get number of active neuron for each element in the sequence
        number_elements_per_batch = sum([len(seq) for seq in model_instance.sequences])
        start_time = excitation_times[-number_elements_per_batch] - 5 
        end_time = excitation_times[-1] + 5

        idx_times = np.where((np.array(excitation_times) > start_time) & (np.array(excitation_times) < end_time))  
        excitation_times_sel = np.array(excitation_times)[idx_times]

        num_active_neurons = helper.number_active_neurons_per_element(model_instance.sequences, somatic_spikes[:,1], somatic_spikes[:,0], excitation_times_sel, params['fixed_somatic_delay'])

        print('\n##### testing sequences with number of somatic spikes ')
        count_false_negatives = 0
        for i, (sequence, seq_counts) in enumerate(zip(model_instance.sequences, num_active_neurons)): 
            seq = ''
            for j, (char, counts) in enumerate(zip(sequence, seq_counts)):
                seq += str(char)+'('+ str(seq_counts[char])+')'.ljust(2)

                if j != 0 and seq_counts[char] > 0.5*params['n_E']:
                    count_false_negatives += 1

            print('sequence %d: %s' % (i, seq))   

        print('False negative counts', count_false_negatives)   

    print('\n### Plasticity parameters')
    print('lambda plus: %0.4f' % model_instance.params['syn_dict_ee']['lambda_plus'])
    print('lambda homeostasis: %0.4f' % model_instance.params['syn_dict_ee']['lambda_h'])
    print('lambda minus: %0.4f' % model_instance.params['syn_dict_ee']['lambda_minus']) 
    print('inh factor:', params['inh_factor'])
    print('excitation step %0.1fms' % params['DeltaT']) #30-50  
    print('seed number: %d' % params['seed']) 
    print('number of learning episodes: %d' % params['learning_episodes'])

generate_reference_data()
