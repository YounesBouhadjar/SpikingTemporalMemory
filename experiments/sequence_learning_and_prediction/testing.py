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


def generate_reference_data(arr_id=None):

    parser = create_parser()
    args, unparsed = parser.parse_known_args()

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
        params['task']['S'] = wandb.config['S']
        params['task']['C'] = wandb.config['C']
        params['task']['R'] = wandb.config['R']
        params['task']['O'] = wandb.config['O']

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
    S = int(params['task']['S'])                             # number of sequences
    C = int(params['task']['C'])                             # sequence length

    start = 100.
    stop = 5000.
    seq_set_instance_size = 10
    subset_size           = None
    #order                 = 'fixed'      ## 'fixed', 'random'
    order                 = 'random'      ## 'fixed', 'random'    
    seq_activation_type   = 'consecutive' ## 'consecutive', 'parallel'
    #seq_activation_type   = 'parallel' ## 'consecutive', 'parallel'    
    inter_seq_intv_min    = 50.
    inter_seq_intv_max    = 55.

    seed = 0 #None              ## RNG seed (int or None)

    if R > (S - 2) or O > (C - 2):

        wandb.log({"loss": -1,
                   "fp": -1,
                   "fn": -1})

        #exit()

    minimal_prefix_length = 1   # minimal prefix length
    minimal_postfix_length = 1  # minimal postfix length
    redraw = True               # if redraw == True: pre- and postfixes may contain repeating elements 
    seed = params['task']['seed']                      # RNG seed (int or None)
    alphabet = sg.latin_alphabet                       # function defining type of alphabet (only important for printing)
   
    ####################    
    label = params['label']
    params_path = helper.get_data_path(params['data_path'])
    seq_set = helper.load_data(params_path,  f'{label}/training_data')
    seq_set_instance = helper.load_data(params_path,  f'{label}/seq_set_instance')
    vocabulary_transformed = helper.load_data(params_path,  f'{label}/vocabulary')
    
    #sequences, _, vocabulary = helper.generate_sequences(params['task'],
    #                                                     params['data_path'],
    #                                                     params['label'])
    #print(f"\n vocabulary_size {len(vocabulary_transformed)}, R={R}, O={O}, S={S}, C={C}")

    # ###############################################################
    # load resampled data
    # ===============================================================
    #params['M'] = len(vocabulary_transformed)
    #print('\nseq_set_instance:', seq_set_instance)
    model_instance = model.Model(params,
                                 seq_set,
                                 vocabulary_transformed)

    seq_set_instance_size = max(seq_set_instance.keys())
    xt, labels = model_instance.load_resampled_data(seq_set_instance,
                                                    seq_set_instance_size)

    acc = model_instance.train_readout(xt, labels)

    return acc

if __name__ == '__main__':
    # generate_reference_data()
    acc = generate_reference_data()

    wandb.log({"acc": acc})
