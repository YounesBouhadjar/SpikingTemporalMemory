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

import os
import wandb
import nest
import sys
import time
import copy
import numpy as np

import argparse
from argparse import ArgumentParser
from shtm import model, helper
import sequence_generator as sg


def create_parser():
    """
    Creates CLI parser
    Returns
    -------

    """
    parser_ = ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser_.add_argument("--yaml", dest="config_yaml", default=None)
    parser_.add_argument("--exp-params", dest="exp_params",  default="parameters_space.py")
    parser_.add_argument("--exp-params-idx", dest="exp_params_idx", type=int, default=0)
    parser_.add_argument("--batch-id", dest="batch_idx", type=int, default=0)
    parser_.add_argument("--jobmax", dest="jobmax", type=int, default=0)
    parser_.add_argument("--hwb", dest="run_hwb", type=bool, default=False, help="enabled if hyperparameter search is executed")
    parser_.add_argument("--wbm", dest="wb_mode", type=str, default="offline")
    return parser_


def generate_reference_data():

    parser = create_parser()
    args, unparsed = parser.parse_known_args()

    # ###########################################################
    # import nestml module
    # ===========================================================
    nest.Install('../../module/nestml_active_dend_module')

    #############################################################
    # get network and training parameters 
    # ===========================================================
    #PS = model.get_parameters()
    PS = __import__(args.exp_params.split(".")[0]).p

    # parameter-set id from command line (submission script)
    PL = helper.parameter_set_list(PS) 

    batch_id = args.batch_idx
    batch_array_id = args.exp_params_idx
    JOBMAX = args.jobmax
    array_id=batch_id*JOBMAX+batch_array_id

    params = PL[array_id]

    if args.run_hwb:
        wandb.init(mode=args.wb_mode,
                   project=PS['data_path']['project_name'],
                   config=wandb.config
                  )

        # TODO: alternatively these could be added to args (see above)
        #params['syn_dict_ee']['lambda_h'] = wandb.config['lambda_h']
        #params['syn_dict_ee']['lambda_minus'] =  wandb.config['lambda_minus']
        params['p_target'] =  wandb.config['w_dep']
        params['n_E'] =  wandb.config['n_E']
        #params['syn_dict_ee']['zt'] = wandb.config['zt']

    else:
        wandb.init(mode=args.wb_mode,
                   project=params['data_path']['project_name'],
                   name = params['label'],
                   config = params
                  )

    # start time 
    time_start = time.time()

    # ###############################################################
    # specify sequences
    # ===============================================================
    vocabulary_size = params['task']['vocabulary_size']          # vocabulary size (may be overwritten if redraw==False)
    R = int(params['task']['R'])                                 # number of shared subsequences
    O = int(params['task']['O'])                                 # length of shared subsequences ("order")
    S = int(2*R)                                                 # number of sequences
    C = int(O+2)                                                 # sequence length
    minimal_prefix_length = 1   # minimal prefix length
    minimal_postfix_length = 1  # minimal postfix length
    redraw = True              # if redraw == True: pre- and postfixes may contain repeating elements 
    seed = params['task']['seed']                      # RNG seed (int or None)
    alphabet = sg.latin_alphabet                       # function defining type of alphabet (only important for printing)
    
    ####################    
    
    seq_set, shared_seq_set, vocabulary = sg.generate_sequences(S, C, R, O, vocabulary_size, minimal_prefix_length, minimal_postfix_length, seed, redraw)

    sg.print_sequences(seq_set, shared_seq_set, vocabulary, label='(int)')
    
    shared_seq_set_transformed = sg.transform_sequence_set(shared_seq_set, alphabet)    
    seq_set_transformed = sg.transform_sequence_set(seq_set, alphabet)
    vocabulary_transformed = sg.transform_sequence(vocabulary, alphabet)

    sg.print_sequences(seq_set_transformed, shared_seq_set_transformed, vocabulary_transformed, label='(latin)')
    #params['M'] = len(vocabulary)
 
    #seq_set_transformed = [['B', 'C', 'E', 'F', 'B', 'C', 'E', 'C', 'F', 'B', 'C', 'E'], ['B', 'C', 'E', 'C', 'F', 'B', 'C', 'E', 'D', 'E', 'F', 'B']]
    #seq_set_transformed = [['A', 'D', 'D', 'E'], ['B', 'D', 'D', 'F']]
    #seq_set_transformed = [['A', 'D', 'B', 'E'], ['C', 'D', 'B', 'F']]
    #seq_set_transformed = [['A', 'D'], ['B', 'D']]
    #vocabulary_transformed = ['A', 'B', 'C', 'D', 'E', 'F']

    print(f"\n vocabulary size {len(vocabulary)}")

    if params['store_training_data']:
        fname = 'training_data'
        fname_voc = 'vocabulary'
        data_path = helper.get_data_path(params['data_path'])
        print("\nSave training data to %s/%s" % (data_path, fname))
        os.makedirs('%s/%s' % (data_path, params['label']), exist_ok=True)
        np.save('%s/%s/%s' % (data_path, params['label'], fname), seq_set_transformed)
        np.save('%s/%s/%s' % (data_path, params['label'], fname_voc), vocabulary_transformed)

    #sequences, _, vocabulary = helper.generate_sequences(params['task'], params['data_path'], params['label'])

    # ###############################################################
    # create network
    # ===============================================================
    params['M'] = len(vocabulary_transformed)
    model_instance = model.Model(params, seq_set_transformed, vocabulary_transformed)
    time_model = time.time()

    model_instance.create()
    time_create = time.time()

    # ###############################################################
    # connect the netwok
    # ===============================================================
    model_instance.connect()
    time_connect = time.time()
    
    # store connections before learning
    if params['store_connections']:
        model_instance.save_connections(fname='ee_connections_before')

    # ###############################################################
    # simulate the network
    # ===============================================================
    model_instance.simulate()
    time_simulate = time.time()

    # store connections after learning
    if params['store_connections']:
        model_instance.save_connections(fname='ee_connections')

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

    # display prediction performance only for debugging    
    if params['evaluate_performance']:
    
        data_path = helper.get_data_path(model_instance.params['data_path'], model_instance.params['label'])

        # print Ic
        #zs = np.array([nest.GetStatus(model_instance.exc_neurons)[i]['z'] for i in range(params['M']*params['n_E'])])
        #id_zs = np.where(zs>0.5)
        #print(zs[id_zs])

        # load spikes from reference data
        somatic_spikes = helper.load_numpy_spike_data(data_path, 'somatic_spikes')
        idend_eval = helper.load_numpy_spike_data(data_path, 'idend_eval')
        excitation_times = helper.load_data(data_path, 'excitation_times')

        # get recoding times of dendriticAP
        idend_recording_times = helper.load_data(data_path, 'idend_recording_times')
        characters_to_subpopulations = helper.load_data(data_path, 'characters_to_subpopulations')

        seq_avg_errors, seq_avg_false_positives, seq_avg_false_negatives, _ = helper.compute_prediction_performance(somatic_spikes, 
                                                                                                                    idend_eval, 
                                                                                                                    idend_recording_times, 
                                                                                                                    characters_to_subpopulations, 
                                                                                                                    model_instance.sequences, 
                                                                                                                    model_instance.params)

        # get number of active neuron for each element in the sequence
        number_elements_per_batch = sum([len(seq) for seq in model_instance.sequences])
        start_time = excitation_times[-number_elements_per_batch] - 5 
        end_time = excitation_times[-1] + 5

        idx_times = np.where((np.array(excitation_times) > start_time) & (np.array(excitation_times) < end_time))  
        excitation_times_sel = np.array(excitation_times)[idx_times]

        num_active_neurons = helper.number_active_neurons_per_element(model_instance.sequences, somatic_spikes[:,1], somatic_spikes[:,0], excitation_times_sel, params['fixed_somatic_delay'])

        print("\n##### testing sequences with number of somatic spikes ")
        count_false_negatives = 0
        for i, (sequence, seq_counts) in enumerate(zip(model_instance.sequences, num_active_neurons)): 
            seq = ''
            for j, (char, counts) in enumerate(zip(sequence, seq_counts)):
                seq += str(char)+'('+ str(counts)+')'.ljust(2)

                if j != 0 and counts > 0.5*params['n_E']:
                    count_false_negatives += 1

            print("sequence %d: %s" % (i, seq))   

        print("False negative counts", count_false_negatives)   

        wandb.log({"loss": seq_avg_errors[-1], "fp": seq_avg_false_positives[-1], "fn": seq_avg_false_negatives[-1]})


    wandb.finish()

    print("\n### Plasticity parameters")
    print("lambda plus: %0.4f" % params['syn_dict_ee']['lambda_plus'])
    print("lambda homeostasis: %0.4f" % params['syn_dict_ee']['lambda_h'])
    print("lambda minus: %0.4f" % model_instance.params['syn_dict_ee']['lambda_minus']) 
    print("excitation step %0.1fms" % params['DeltaT']) #30-50  
    print("seed number: %d" % params['seed']) 
    print("number of learning episodes: %d" % params['learning_episodes'])

if __name__ == '__main__':
    generate_reference_data()
