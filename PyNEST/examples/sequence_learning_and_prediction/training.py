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
# SPDX-License-Identifier: GPL-3.0-or-later

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
import hashlib
import argparse

import numpy as np
from pprint import pformat
from argparse import ArgumentParser

from spikingtemporalmemory import model, helper
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
    parser_.add_argument("--avg_seed", dest="avg_seed", action='store_true')
    #parser_.add_argument("--wbm", dest="wb_mode", type=str, default="offline")
    parser_.add_argument("--wbm", dest="wb_mode", type=str, default="online")
    parser_.add_argument("--assess_performance_only_at_end", action='store_true')

    # learning parameters
    parser_.add_argument("--tau_perm", type=float, default=None)
    parser_.add_argument("--lambda_plus", type=float, default=None)
    parser_.add_argument("--lambda_minus", type=float, default=None)

    # task parameters
    parser_.add_argument("--S", type=int, default=None)
    parser_.add_argument("--C", type=int, default=None)
    parser_.add_argument("--R", type=int, default=None)
    parser_.add_argument("--O", type=int, default=None)

    # seed value
    parser_.add_argument("--seed", type=int, default=None)

    return parser_


def train(PS, arr_id=None):

    plot_task = False

    #############################################################
    # get network and training parameters 
    # ===========================================================

    # parameter-set id from command line (submission script)
    PL = helper.parameter_set_list(PS) 

    batch_id = args.batch_idx
    batch_array_id = args.exp_params_idx
    JOBMAX = args.jobmax
    array_id=batch_id*JOBMAX+batch_array_id

    # select default parameters
    if arr_id is not None:
        params = PL[arr_id]
    else:
        params = PL[array_id]
 
    # adjust the parameters using the ones passed through command args
    if None not in (args.tau_perm, args.lambda_minus, args.lambda_plus):
        params['syn_dict_ee']['tau_perm'] = args.tau_perm
        params['syn_dict_ee']['lambda_minus'] = args.lambda_minus
        params['syn_dict_ee']['lambda'] = args.lambda_plus
    else:
        print("Warning: using default plasticity parameters")
        
    if None not in (args.S, args.C, args.R, args.O):
        params['task']['S'] = args.S
        params['task']['C'] = args.C
        params['task']['R'] = args.R
        params['task']['O'] = args.O
    else:
        print("Warning: using default task parameters")

    if args.seed != None:
        params['seed'] = args.seed
    else:
        print("Warning: using default seed value")

    wandb.init(mode=args.wb_mode,
               project=PS['data_path']['project_name'],
               config=params)

    params['label'] = hashlib.md5(pformat(dict(params)).encode('utf-8')).hexdigest()

    # ###########################################################
    # import nestml module
    # ===========================================================
    neuron_model = params['soma_model']
    synapse_model = params['syn_dict_ee_synapse_model']
   
    try:
        nest.Install('../../module/nestml_' + neuron_model + '_module')
        nest.Install('../../module/nestml_' + neuron_model + '_' + synapse_model + '_module')
    except:
        pass

    params['soma_model'] = neuron_model + '_nestml_' + '_with_' + synapse_model + '_nestml'
    params['syn_dict_ee']['synapse_model'] = synapse_model + '_nestml_' + '_with_' + neuron_model + '_nestml'
    # start time 
    time_start = time.time()

    # ###############################################################
    # specify sequences
    # ===============================================================
    vocabulary_size = params['task']['vocabulary_size']          # vocabulary size (may be overwritten if redraw==False)
    R = int(params['task']['R'])                                 # number of shared subsequences
    O = int(params['task']['O'])                                 # length of shared subsequences ("order")
    S = int(params['task']['S'])                                  # number of sequences
    C = int(params['task']['C'])                                  # sequence length

    # simulation parameters from parameter space
    start = params['start']
    stop = params['stop']
    inter_seq_intv_min    = params['DeltaT']
    inter_seq_intv_max    = params['DeltaT']
    inter_elem_intv_min = params['DeltaT_seq']
    inter_elem_intv_max = params['DeltaT_seq']

    seq_set_instance_size = params['task']['seq_set_instance_size']
    subset_size = params['task']['subset_size']
    order = params['task']['order']
    seq_activation_type = params['task']['seq_activation_type']
    seed = int(params['task']['seed'])

    if R > (S - 1) or O > (C - 2):

        wandb.log({"mse": -1,
                   "acc": -1})

        print('R', R, 'S', S, 'O', O, 'C', C)
        print("check: R > (S - 2) or O > (C - 2)")
        exit()


    minimal_prefix_length = 1   # minimal prefix length
    minimal_postfix_length = 1  # minimal postfix length
    redraw = True               # if redraw == True: pre- and postfixes may contain repeating elements 
    alphabet = sg.latin_alphabet                    # function defining type of alphabet (only important for printing)
   
    ####################    
    print("Generate sequences ...")
    seq_set, shared_seq_set, vocabulary, seq_set_intervals = sg.generate_sequences(S, C, R, O,
                                                                           vocabulary_size,
                                                                           minimal_prefix_length,
                                                                           minimal_postfix_length,
                                                                           seed,
                                                                           redraw,
                                                                           inter_elem_intv_min,
                                                                           inter_elem_intv_max)

    shared_seq_set_transformed = sg.transform_sequence_set(shared_seq_set, alphabet)    
    seq_set_transformed = sg.transform_sequence_set(seq_set, alphabet)
    vocabulary_transformed = sg.transform_sequence(vocabulary, alphabet)

    sg.print_sequences(seq_set_transformed,
                   shared_seq_set_transformed,
                   vocabulary_transformed,
                   seq_set_intervals,
                   label=' (latin)')

    print("Generate sequence instance ...")
    seq_set_instance, seq_ids = sg.generate_sequence_set_instance(
        seq_set,
        seq_set_intervals,
        start=start,
        stop=stop,
        seq_set_instance_size = seq_set_instance_size,
        subset_size           = subset_size,
        order                 = order,
        seq_activation_type   = seq_activation_type,
        inter_seq_intv_min    = inter_seq_intv_min,
        inter_seq_intv_max    = inter_seq_intv_max,
    )

    # Compute dynamically the simulation time and duration of the presented sequences
    sim_stop = seq_set_instance[max(seq_set_instance.keys())]['times'][-1]
    sim_stop = int(sim_stop) + 10.
    print('\nSim Stop:', sim_stop)

    duration = seq_set_instance[max(seq_set_instance.keys())]['times'][-1] - seq_set_instance[max(seq_set_instance.keys())-S]['times'][0]
    duration = int(duration)
    print('\nDuration:', duration)

    # convert sequence set instance to element activation times
    element_activations = sg.seq_set_instance_gdf(seq_set_instance)

    if params['store_training_data']:
        fname = 'training_data'
        fname_voc = 'vocabulary'
        data_path = helper.get_data_path(params['data_path'])
        print("\nSave training data to %s/%s" % (data_path, fname))
        os.makedirs('%s/%s' % (data_path, params['label']), exist_ok=True)
        np.save('%s/%s/%s' % (data_path, params['label'], fname), seq_set_transformed)
        np.save('%s/%s/%s' % (data_path, params['label'], 'seq_set_instance'), seq_set_instance)
        np.save('%s/%s/%s' % (data_path, params['label'], fname_voc), vocabulary_transformed)

    #sequences, _, vocabulary = helper.generate_sequences(params['task'], params['data_path'], params['label'])
    print(f"\n vocabulary_size {len(vocabulary_transformed)}, R={R}, O={O}, S={S}, C={C}")

    # ###############################################################
    # create network
    # ===============================================================
    params['M'] = len(vocabulary_transformed)
    model_instance = model.Model(params,
                                 vocabulary_transformed)
    time_model = time.time()

    model_instance.create()

    if args.assess_performance_only_at_end:
        idend_dict = {'interval': params['idend_recording_interval'],
                      'start': sim_stop-duration,
                      'stop': sim_stop}

        model_instance.set_status(model_instance.multimeter_idend,
                                  idend_dict)

    time_create = time.time()

    # ###############################################################
    # connect the netwok
    # ===============================================================
    model_instance.connect()
    time_connect = time.time()

    # ###############################################################
    # set input
    # ===============================================================
    model_instance.set_input(element_activations, seq_set_instance)

    # ###############################################################
    # simulate the network
    # ===============================================================
    exp_seq_set_instance_size = max(seq_set_instance.keys())
    if args.assess_performance_only_at_end:
        model_instance.simulate(sim_stop, save_data=True)

        err, fn, fp = model_instance.measure_fp_fn(S=S, #TODO
                                                   seq_set_instances=seq_set_instance,
                                                   seq_set_instance_id=exp_seq_set_instance_size,
                                                   load_data=True)
        err /= C 
        ts = np.nan
    else:
        #TODO move following to parameters.py
        err_prev = np.inf
        tprev = 0
        counter_abort = 0
        record_ts = params['record_ts']
        early_abort = params['early_abort']
        early_break = params['early_break']
        K = params['K']
        M_abort = params['M_abort']
        min_error = params['min_error']

        # assess the performance each Kth episode
        for i in range(S-1, exp_seq_set_instance_size-1, K*S):
            print("Simulate step", i)
            tnext = seq_set_instance[i+1]['times'][0]
            sim_time = int(tnext - tprev - 10.)
            model_instance.simulate(sim_time, save_data=False)

            err, fn, fp = model_instance.measure_fp_fn(S=S, #TODO
                                                       seq_set_instances=seq_set_instance,
                                                       seq_set_instance_id=i+1,
                                                       load_data=False)

            err /= C

            wandb.log({"err"+str(arr_id): err,
                       "fn"+str(arr_id): fn,
                       "fp"+str(arr_id): fp})

            if err < min_error and record_ts:
                wandb.log({"ts"+str(arr_id): i})
                ts = i 
                record_ts = False
            
                if early_break:
                    break

            time_simulate = time.time()
            tprev += sim_time

            # early abort for unsuccessful experiments
            e = err - err_prev
            if e >= 0:
                counter_abort += 1 
            else:
                counter_abort = 0

            if counter_abort > M_abort and early_abort:
                break

            err_prev = err

        if record_ts:
            wandb.log({"ts"+str(arr_id): None})
            ts = np.nan

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

    return err, fn, fp, ts


if __name__ == '__main__':

    parser = create_parser()
    args, unparsed = parser.parse_known_args()

    PS = __import__(args.exp_params.split(".")[0]).p

    print("args.exp_params", args.exp_params)

    if args.avg_seed:

        err_all = []
        fn_all = []
        fp_all = []
        ts_all = []
        for i in range(len(PS['seed'])):
            print("Experiment", i)
            nest.ResetKernel()
            errs, fns, fps, tss = train(PS, i)
        
            err_all.append(errs)
            fn_all.append(fns)
            fp_all.append(fps)
            ts_all.append(tss)

        err = sum(err_all) / len(err_all)
        fn = sum(fn_all) / len(fn_all)
        fp = sum(fp_all) / len(fp_all)
        ts = np.nanmean(ts_all)

        wandb.log({"err": err,
                   "fn": fn,
                   "fp": fp,
                   "ts": ts})
    else:
        err, fn, fp, ts = train(PS, args.exp_params_idx)

        wandb.log({"err": err,
                   "fn": fn,
                   "fp": fp,
                   "ts": ts})

    wandb.finish()
