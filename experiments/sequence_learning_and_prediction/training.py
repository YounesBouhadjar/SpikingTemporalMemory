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
import hashlib
import argparse

import numpy as np
from pprint import pformat
from argparse import ArgumentParser
sys.path.append('../../')
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
    #parser_.add_argument("--wbm", dest="wb_mode", type=str, default="offline")
    parser_.add_argument("--wbm", dest="wb_mode", type=str, default="online")
    return parser_


def generate_reference_data(PS, arr_id=None):

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

    if arr_id is not None:
        params = PL[arr_id]
    else:
        params = PL[array_id]

    if args.run_hwb:
        wandb.init(mode=args.wb_mode,
                   project=PS['data_path']['project_name'],
                   config=wandb.config #TODO: find a way to log params as well
                  )

        # TODO: alternatively these could be added to args (see above)
        #params['syn_dict_ee']['lambda_h'] = wandb.config['lambda_h']
        params['syn_dict_ee']['tau_perm'] = wandb.config['tau_perm']
        params['syn_dict_ee']['lambda_minus'] = wandb.config['lambda_minus']
        params['syn_dict_ee']['lambda'] = wandb.config['lambda']
        try:
            params['task']['S'] = wandb.config['S']
            params['task']['C'] = wandb.config['C']
            params['task']['R'] = wandb.config['R']
            params['task']['O'] = wandb.config['O']
        except:
            pass

        try:
            params['seed'] = wandb.config['seed']
        except:
            pass

        params['label'] = hashlib.md5(pformat(dict(params)).encode('utf-8')).hexdigest()

    else:
        wandb.init(mode=args.wb_mode,
                   project=params['data_path']['project_name'],
                   name = params['label'],
                   config = params
                  )
    
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
#    if R != 0:
#        S = int(2*R)                                             # number of sequences
#        C = int(O+2)                                             # sequence length
#    else:
    S = int(params['task']['S'])                                  # number of sequences
    C = int(params['task']['C'])                                  # sequence length

    start = 100.
    stop = 5000000.
    seq_set_instance_size = 1000
    subset_size           = None
    order                 = 'fixed'      ## 'fixed', 'random'
    #order                 = 'random'      ## 'fixed', 'random'    
    seq_activation_type   = 'consecutive' ## 'consecutive', 'parallel'
    #seq_activation_type   = 'parallel' ## 'consecutive', 'parallel'    
    inter_seq_intv_min    = 100.
    inter_seq_intv_max    = 105.

    inter_elem_intv_min = 50.
    inter_elem_intv_max = 50.

    if R > (S - 1) or O > (C - 2):

        wandb.log({"mse": -1,
                   "acc": -1})

        print('R', R, 'S', S, 'O', O, 'C', C)
        print("check: R > (S - 2) or O > (C - 2)")
        exit()


    minimal_prefix_length = 1   # minimal prefix length
    minimal_postfix_length = 1  # minimal postfix length
    redraw = True               # if redraw == True: pre- and postfixes may contain repeating elements 
    seed = int(params['task']['seed'])              # RNG seed (int or None)
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

    sim_stop = seq_set_instance[max(seq_set_instance.keys())]['times'][-1]
    num_last_instance = max(seq_set_instance.keys())
    print('\nNumber and time of last element in last sequence instance:', num_last_instance, sim_stop)
    print('\nStop:', stop)

    sim_stop = int(sim_stop) + 10.
    print('\nSim Stop:', sim_stop)

    duration = seq_set_instance[max(seq_set_instance.keys())]['times'][-1] - seq_set_instance[max(seq_set_instance.keys())-S]['times'][0]
    duration = int(duration)
    print('\nDuration:', duration)

    # convert sequence set instance to element activation times
    element_activations = sg.seq_set_instance_gdf(seq_set_instance)

    if plot_task:
        import matplotlib.pyplot as plt

        plt.rcParams.update({'font.size': 8})
        plt.figure(1,dpi=300,figsize=(5,3))
        plt.clf()

        ylim = (vocabulary[0],vocabulary[-1])
        sg.plot_seq_instance_intervals(seq_set,seq_ids,
                                       seq_set_instance,
                                       ylim,
                                       alpha=0.1,
                                       cm='jet')    

        colormap = plt.get_cmap('jet')
        colors = [colormap(k) for k in np.linspace(0, 1, len(seq_set))]    
        for cs in range(len(seq_set_instance)):
            clr = colors[seq_ids[cs]]
            plt.plot(seq_set_instance[cs]['times'],
                     seq_set_instance[cs]['elements'],
                     'o', ms=3, mfc=clr, mew=0.5, 
                     mec='k', rasterized=True)
            plt.text(seq_set_instance[cs]['times'][0],vocabulary[-1]+1,r"%d" % seq_ids[cs],
                     fontsize=5)
        plt.xlabel(r'time (ms)')
        plt.ylabel(r'element ID')
        plt.xlim(0, stop)
        plt.ylim(vocabulary[0]-0.5,
                 vocabulary[-1]+2)

        plt.setp(plt.gca(),
                 yticks = vocabulary)
        
        plt.subplots_adjust(left=0.13, right=0.95, bottom=0.15, top=0.95)
        plt.savefig('sequence_set_instance.pdf')

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
                                 seq_set,
                                 vocabulary_transformed)
    time_model = time.time()

    model_instance.create(element_activations,
                          seq_set_instance,
                          sim_stop, duration)
    time_create = time.time()

    # ###############################################################
    # connect the netwok
    # ===============================================================
    model_instance.connect()
    time_connect = time.time()
    
    # store connections before learning
    #if params['store_connections']:
    #    model_instance.save_connections(fname='ee_connections_before')

    # ###############################################################
    # simulate the network
    # ===============================================================
    model_instance.simulate(sim_stop)
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

        exp_seq_set_instance_size = max(seq_set_instance.keys())
        xt, labels, spt = model_instance.load_resampled_data(seq_set_instance,
                                                             exp_seq_set_instance_size)

        errs, fns, fps = model_instance.measure_fp_fn(seq_set_instance,
                                                      exp_seq_set_instance_size)

        spl_last = []
        for k, spl in enumerate(spt[-S:]):
            print(f'\n{k}: {spl}')
            spl_last += spl[1:]
        
        sp_last = np.mean(spl_last)
        acc, mse, readout = model_instance.train_readout(xt, labels, int(S))
        
        #model_instance.plot_readout(readout, np.concatenate(xt[-int(S):], axis=1))

        wandb.log({"errs"+str(arr_id): errs,
                   "fns"+str(arr_id): fns,
                   "fps"+str(arr_id): fps,
                   "mse"+str(arr_id): mse,
                   "acc"+str(arr_id): acc,
                   "sp"+str(arr_id) : sp_last})

        return errs, fns, fps, acc, mse, sp_last

    else:
        return 0, 1


    wandb.finish()

    print("\n### Plasticity parameters")
    print("lambda: %0.4f" % params['syn_dict_ee']['lambda'])
    print("lambda minus: %0.4f" % model_instance.params['syn_dict_ee']['lambda_minus']) 
    print("excitation step %0.1fms" % params['DeltaT']) #30-50  
    print("seed number: %d" % params['seed']) 
    print("number of learning episodes: %d" % params['learning_episodes'])

if __name__ == '__main__':

    parser = create_parser()
    args, unparsed = parser.parse_known_args()

    PS = __import__(args.exp_params.split(".")[0]).p

    # generate_reference_data()
    if args.run_hwb:
        err_all = []
        fn_all = []
        fp_all = []

        sp_all = []
        acc_all = []
        mse_all = []
        # go consecutively through the seed to compute an averaged loss
        # to be logged in and used by the hyperparameter search of wandb
        for i in range(len(PS['seed'])):
            print("Experiment", i)
            nest.ResetKernel()
            err, fn, fp, acc, mse, sp_last = generate_reference_data(PS, i)
        
            err_all.append(err)
            fn_all.append(fn)
            fp_all.append(fp)

            acc_all.append(acc)
            mse_all.append(mse)
            sp_all.append(sp_last)

        err = sum(err_all) / len(err_all)
        fn = sum(fn_all) / len(fn_all)
        fp = sum(fp_all) / len(fp_all)

        sp = sum(sp_all) / len(sp_all)
        acc = sum(acc_all) / len(acc_all)
        mse = sum(mse_all) / len(mse_all)

        wandb.log({"err": err,
                   "fn": fn,
                   "fp": fp,
                   "mse": mse,
                   "acc": acc,
                   "sp": sp})
    else:
        err, fn, fp, acc, mse, sp = generate_reference_data(PS, args.exp_params_idx)

        wandb.log({"err": err,
                   "fn": fn,
                   "fp": fp,
                   "mse": mse,
                   "acc": acc,
                   "sp": sp})
