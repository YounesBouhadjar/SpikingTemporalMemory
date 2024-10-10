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
Helper functions of the Spiking-TemporalMemory model

Authors
~~~~~~~
Younes Bouhadjar
"""

import os
import sys
import copy
import hashlib
import numpy as np
from pathlib import Path
from pprint import pformat
from collections import Counter


###############################################################################
def psp_max_2_psc_max(psp_max, tau_m, tau_s, R_m):
    """Compute the PSC amplitude (pA) injected to get a certain PSP maximum (mV) for LIF with exponential PSCs

    Parameters
    ----------
    psp_max: float
             Maximum postsynaptic pontential
    tau_m:   float
             Membrane time constant (ms).
    tau_s:   float
             Synaptic time constant (ms).
    R_m:     float
             Membrane resistance (Gohm).

    Returns
    -------
    float
        PSC amplitude (pA).
    """

    return psp_max / (
            R_m * tau_s / (tau_s - tau_m) * (
            (tau_m / tau_s) ** (-tau_m / (tau_m - tau_s)) -
            (tau_m / tau_s) ** (-tau_s / (tau_m - tau_s))
    )
    )


##########################################
def generate_sequences(params, data_path, fname):
    """Generate sequence of elements using three methods:
    1. randomly drawn elements from a vocabulary
    2. sequences with transition matrix
    3. higher order sequences: sequences with shared subsequences
    4. hard coded sequences

    Parameters
    ----------
    params : dict
        dictionary contains task parameters
    data_path   : dict
    fname       : str

    Returns
    -------
    sequences: list
    test_sequences: list
    vocabulary: list
    """

    task_name = params['task_name']
    task_type = params['task_type']
 
    # set of characters used to build the sequences
    vocabulary = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
                  'U', 'V', 'W', 'X', 'Y', 'Z'][:params['vocab_size']]
    sequences = []

    # create high order sequences, characters are drawn without replacement
    if task_name == "high_order":

        if (params['num_sequences'] % params['num_sub_seq'] != 0):
            raise ZeroDivisionError(
                'for high order sequences number of sequences needs ("num_sequences") to be divisible by num_sub_seq')

        num_sequences_high_order = int(params['num_sequences'] / params['num_sub_seq'])
        for i in range(num_sequences_high_order):
            characters_sub_seq = copy.copy(vocabulary)
            sub_seq = random.sample(characters_sub_seq, params["length_sequence"] - 2)
            for char in sub_seq:
                characters_sub_seq.remove(char)

            for j in range(params['num_sub_seq']):
                # remove the characters that were chosen for the end and the start of the sequence
                # this is to avoid sequences with adjacent letters of the same kind
                # we will add this feature to the code asap 
                end_char = random.sample(characters_sub_seq, 1)
                characters_sub_seq.remove(end_char[0])

                start_char = random.sample(characters_sub_seq, 1)
                characters_sub_seq.remove(start_char[0])

                sequence = start_char + sub_seq + end_char
                sequences.append(sequence)

                # randomly shuffled characters
    elif task_name == "random":
        sequences = [random.sample(vocabulary, length_seq) for _ in range(params['num_sequences'])]

    # create sequences using matrix transition 
    elif task_name == "structure":
        matrix_transition = defaultdict(list)
        for char in vocabulary:
            x = np.random.choice(2, len(vocabulary), p=[0.2, 0.8])
            matrix_transition[char] = x / sum(x)

        for _ in range(params['num_sequences']):
            sequence = random.sample(vocabulary, 1)
            last_char = sequence[-1]
            for _ in range(length_seq - 1):
                sequence += np.random.choice(vocabulary, 1, p=matrix_transition[last_char])[0]
                last_char = sequence[-1]

            sequences += [sequence]
    else:

        # hard coded sequences 
        if task_type == 1:
            sequences = [['A', 'D', 'B', 'E'], ['F', 'D', 'B', 'C']]
        elif task_type == 2:
            sequences = [['E', 'N', 'D', 'I', 'J'], ['L', 'N', 'D', 'I', 'K'], ['G', 'J', 'M', 'C', 'N'], 
                         ['F', 'J', 'M', 'C', 'I'], ['B', 'C', 'K', 'H', 'I'], ['A', 'C', 'K', 'H', 'F']]
        elif task_type == 3:
            sequences = [['E', 'N', 'D', 'I', 'J'], ['L', 'N', 'D', 'I', 'K'], ['G', 'J', 'M', 'E', 'N'], ['F', 'J', 'M', 'E', 'I'], ['B', 'C', 'K', 'B', 'I'], ['A', 'C', 'K', 'B', 'F']]
        elif task_type == 3.1:
            sequences = [['B', 'D', 'I', 'C', 'H'], ['E', 'D', 'I', 'C', 'F'], ['F', 'B', 'C', 'A', 'H'], ['G', 'B', 'C', 'A', 'D'], ['E', 'C', 'I', 'H', 'A'], ['D', 'C', 'I', 'H', 'G']]
        elif task_type == 4:
            sequences = [['A', 'D', 'B', 'G', 'H', 'E'], ['F', 'D', 'B', 'G', 'H', 'C']]
        elif task_type == 5:
            sequences = [['A', 'B', 'C', 'D'], ['E', 'F', 'G', 'H'], ['I', 'J', 'K', 'L']]
        else:
            sequences = [['A', 'D', 'B', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N','E'], ['F', 'D', 'B', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'C']]

    # test sequences used to measure the accuracy 
    test_sequences = sequences

    if params['store_training_data']:
        fname = 'training_data'
        fname_voc = 'vocabulary'
        data_path = get_data_path(data_path)
        print("\nSave training data to %s/%s" % (data_path, fname))
        np.save('%s/%s' % (data_path, fname), sequences)
        np.save('%s/%s' % (data_path, fname_voc), vocabulary)

    return sequences, test_sequences, vocabulary


###############################################################################
def derived_parameters(params):
    """Set additional parameters derived from base parameters.

    A dictionary containing all (base and derived) parameters is stored as model attribute params

    Parameters
    ----------
    params:    dict
               Parameter dictionary
    """

    params = copy.deepcopy(params)

    # connection rules for EE connections
    params['conn_dict_ee'] = {}
    params['conn_dict_ee']['rule'] = params['rule']
    params['conn_dict_ee']['indegree'] = int(params['connection_prob'] *
                                             params['M'] *
                                             params['n_E'])
    params['conn_dict_ee']['allow_autapses'] = False
    params['conn_dict_ee']['allow_multapses'] = False

    # compute neuron's membrane resistance
    params['R_m_soma'] = params['soma_params']['tau_m'] / params['soma_params']['C_m']
    params['R_m_inhibit'] = params['inhibit_params']['tau_m'] / params['inhibit_params']['C_m']

    # compute psc max from the psp max
    params['J_IE_psp'] = 1.2 * params['inhibit_params']['V_th']         # inhibitory PSP as a response to an input from E neuron

    #if params['evaluate_replay']:
    #    params['J_IE_psp'] /= params['n_E']
    #else:
    #    params['J_IE_psp'] /= params['pattern_size']
    params['J_IE_psp'] /= params['pattern_size']

    params['syn_dict_ex']['weight'] = psp_max_2_psc_max(params['J_EX_psp'], params['soma_params']['tau_m'],
                                                   params['soma_params']['tau_syn1'], params['R_m_soma'])
    params['syn_dict_ie']['weight'] = psp_max_2_psc_max(params['J_IE_psp'], params['inhibit_params']['tau_m'],
                                                   params['inhibit_params']['tau_syn_ex'],
                                                   params['R_m_inhibit'])
    params['syn_dict_ei']['weight'] = psp_max_2_psc_max(params['J_EI_psp'], params['soma_params']['tau_m'],
                                                   params['soma_params']['tau_syn3'], params['R_m_soma'])

    # set initial weights (or permanences in the case of the structural synapse)
    import nest

    if params['syn_dict_ee_synapse_model'][:5] == 'stdsp':
        params['syn_dict_ee']['p'] = nest.random.uniform(min=params['p_min'], max=params['p_max']) 
    else:
        params['syn_dict_ee']['weight'] = nest.random.uniform(min=params['w_min'], max=params['w_max'])
 
    params['DeltaT_seq'] = 2.5*params['DeltaT']                         # inter-sequence interval
    
    # clamp DeltaT_seq if it is below the duration of the dAP
    if params['DeltaT_seq'] < params['soma_params']['tau_dAP']:
        params['DeltaT_seq'] = params['soma_params']['tau_dAP']
   
    if 'add_bkgd_noise' not in params:
        params['add_bkgd_noise'] = False
    
    if 'add_stimulus_noise' not in params:
        params['add_stimulus_noise'] = False

    if 'record_voltage' not in params:
        params['record_voltage'] = False 

    if params['add_stimulus_noise']:
        params['syn_dict_ex']['weight'] /= params['spike_packet']['activity']

    # maximum time lag for the STDP window 
    if params['add_bkgd_noise']:
        # similar results can be obtained using -2.*params['DeltaT'] or -1.2*params['DeltaT']
        params['syn_dict_ee']['dt_max'] = -2.*params['DeltaT'] 
        #params['syn_dict_ee']['dt_max'] = -1.2*params['DeltaT']          
    else:
        params['syn_dict_ee']['dt_max'] = -2.*params['DeltaT']          

    print('\n#### postsynaptic potential ####')
    print('PSP maximum J_EX psp:  %f mV' % params['J_EX_psp'])
    print('PSP maximum J_IE psp:  %f mV' % params['J_IE_psp'])
    print('PSP maximum J_EI psp:  %f mV' % params['J_EI_psp'])

    print('\n#### postsynaptic current ####')
    print('PSC maximum J_EX:  %f pA' % params['syn_dict_ex']['weight'])
    print('PSC maximum J_IE:  %f pA' % params['syn_dict_ie']['weight'])
    print('PSC maximum J_EI:  %f pA' % params['syn_dict_ei']['weight'])

    return params


###############################################################################
def get_parameter_set(analysis_pars):
    """ Get parameter set from data directory at location specified in analysis_pars.
    
    Parameters
    ----------
    analysis_pars: dict
    
    Returns
    -------
    P: dict 
       ParameterSpace 
    """

    params_path = get_data_path(analysis_pars)
    sys.path.insert(0, str(params_path))

    import parameters_space as data_pars

    P = data_pars.p

    return P, params_path


###############################################################################
def parameter_set_list(P):
    """ Generate list of parameters sets
    
    Parameters
    ----------
    P : dict  
        parameter space 
    
    Returns
    -------
    l : list 
        list of parameter sets 
    """

    l = []
    for z in P.iter_inner():
        p = copy.deepcopy(dict(z))
        l.append(p)
        l[-1]['label'] = hashlib.md5(pformat(l[-1]).encode(
            'utf-8')).hexdigest()  ## add md5 checksum as label of parameter set (used e.g. for data file names) 

    return l


###############################################################################
def get_data_path(pars, ps_label='', add_to_path=''):
    """ Construct the path to the data directory
    
    Parameters
    ----------
    pars : dict
           path parameters 
    ps_label : string
    add_to_path : string

    Returns
    -------
    data_path : Pathlib instantiation  
    """

    try:
        home = pars['home']
    except:
        #home = '../..'
        #home = Path.home()
        home = '/work/users/bouhadjar'

    data_path = Path(home, pars['data_root_path'],
                     pars['project_name'],
                     pars['parameterspace_label'],
                     ps_label, add_to_path)

    return data_path


###############################################################################
def copy_scripts(pars, fname):
    """ Copying Python scripts to data folder

    Parameters
    ----------
    pars  : dict
    fname : string

    """

    print("\tCopying Python scripts to data folder ...")
    data_path = get_data_path(pars)
    os.system('mkdir -p %s' % (data_path))
    os.system('cp -r --backup=t  %s %s/%s' % (fname, data_path, 'parameters_space.py'))
    # os.system('mv %s/%s %s/%s' % (dat_path,fname,dat_path,'sim_'+fname))


##############################################
def load_numpy_spike_data(path, label, skip_rows=3):
    """Load spike data from files."""

    print("Load spiking data ...")

    fname = '%s/%s.npy' % (path, label)
    x = np.load(fname)

    return x

##############################################
def load_spike_data(path, label, skip_rows=3):
    """Load spike data from files.

    Parameters
    ---------
    path:           str
                    Path containing spike files.

    label:          str
                    Spike file label (file name root).

    skip_rows:      int, optional
                    Number of rows to be skipped while reading spike files (to remove file headers). The default is 3.

    Returns
    -------
    spikes:   numpy.ndarray
              Lx2 array of spike senders spikes[:,0] and spike times spikes[:,1] (L = number of spikes).
    """

    # get list of files names
    files = []
    for file_name in os.listdir(path):
        if file_name.endswith('.dat') and file_name.startswith(label):
            files += [file_name]
    files.sort()

    assert len(files) > 0, "No files of type '%s*.dat' found in path '%s'." % (label, path)

    # open spike files and read data
    spikes = []
    for file_name in files:
        try:
            spikes += [np.loadtxt('%s/%s' % (path,file_name),skiprows=skip_rows)] ## load spike file while skipping the header 
        except:
            print('Error: %s' % sys.exc_info()[1])
            print('Remove non-numeric entries from file %s (e.g. in file header) by specifying (optional) parameter "skip_rows".\n' % (file_name))
    
    try:
        spikes = np.concatenate([spike for spike in spikes if spike.size>0])
    except:
        print("All files are empty")

#    # open spike files and read data
#    spikes = []
#    for file_name in files:
#        try:
#            spikes += [np.loadtxt('%s/%s' % (path, file_name),
#                                  skiprows=skip_rows)]  ## load spike file while skipping the header
#            print(spikes)
#        except:
#            print("Error: %s" % sys.exc_info()[1])
#            print(
#                "Remove non-numeric entries from file %s (e.g. in file header) by specifying (optional) parameter 'skip_rows'.\n" % (
#                    file_name))
#
#    spikes = np.concatenate(spikes)
#
    return spikes


###############################################################################
def load_data(path, fname):
    """Load data

    Parameters
    ----------
    path: str
    fname: str

    Returns
    -------
    data: ndarray
    """

    #TODO: this is temporary hack!
    try:
      data = np.load('%s/%s.npy' % (path, fname), allow_pickle=True).item()
    except:
      data = np.load('%s/%s.npy' % (path, fname), allow_pickle=True)

    return data


###############################################################################
def number_active_neurons_per_element(test_sequences, times_somatic_spikes, senders_somatic_spikes, excitation_times,
                                      fixed_somatic_delay):
    """
    Finds the active neurons of each element in the sequences and return their number

    Parameters
    ----------
    test_sequences         : list
    times_somatic_spikes   : ndarray
    senders_somatic_spikes : ndarray
    excitation_times       : list
    fixed_somatic_delay    : float

    Returns
    -------
    num_active_neurons_per_sequence : list
    """

    num_active_neurons_per_sequence = []
    end_iterations = 0

    assert len(excitation_times) >= 2, "excitation times need to contain at leasts 2 components"
    DeltaT = excitation_times[1] - excitation_times[0]

    # for each sequence in the test sequences
    for seq in test_sequences:
        start_iterations = end_iterations
        end_iterations += len(seq)
        num_active_neurons = []

        # for each character in the sequence
        for k, (j, char) in enumerate(zip(range(start_iterations, end_iterations), seq)):
            indices_soma = np.where((times_somatic_spikes < excitation_times[j] + DeltaT) & 
                                    (times_somatic_spikes > excitation_times[j]))
            senders_soma = senders_somatic_spikes[indices_soma]

            num_active_neurons.append(len(senders_soma))

        num_active_neurons_per_sequence.append(num_active_neurons)

    return num_active_neurons_per_sequence


###############################################################################
def measure_sequences_overlap(test_sequences, times_somatic_spikes, senders_somatic_spikes, excitation_times,
                              fixed_somatic_delay, number_training_episodes):
    """Finds the shared active neurons between the last sequence elements

    Parameters
    ----------
    test_sequences         : list
    times_somatic_spikes   : ndarray
    senders_somatic_spikes : ndarray
    excitation_times       : list
    fixed_somatic_delay    : float
    number_training_episodes : int

    Returns
    -------
    episodes_overlap       : list
    """

    sequences_active_neurons = [[] for _ in range(len(test_sequences))]
    end_iterations = 0
    episodes_overlap = []

    for training_episodes in range(number_training_episodes):
        # for each sequence in the test sequences
        for i, seq in enumerate(test_sequences):
            start_iterations = end_iterations
            end_iterations += len(seq)
            active_neurons = []

            # for each character in the sequence
            for k, (j, char) in enumerate(zip(range(start_iterations, end_iterations), seq)):
                indices_soma = np.where((times_somatic_spikes < excitation_times[j] + fixed_somatic_delay) & (
                        times_somatic_spikes > excitation_times[j]))
                senders_soma = senders_somatic_spikes[indices_soma]

                active_neurons.append(senders_soma)

            sequences_active_neurons[i] = active_neurons

        # compute overlap 
        co = 0
        sequences_overlap = []
        # TODO: use variable for test_sequences[0]
        for q in range(len(test_sequences[0])):
            overlap = [value for value in sequences_active_neurons[co][q] if
                       value in sequences_active_neurons[co + 1][q]]
            size_overlap = len(overlap)
            sequences_overlap.append(size_overlap)
        # TODO here the overlap is computed only between two sequences
        co += 2

        episodes_overlap.append(sequences_overlap)

    return episodes_overlap


###############################################################################
def compute_prediction_performance(somatic_spikes, dendriticAP, dendriticAP_recording_times,
                                   characters_to_subpopulations, test_seq, params):
    """Computes prediction performance including: error, false positive and false negative
    The prediction error is computed as the Euclidean distance between the target vector and the output vector for each last character `q` in a sequence.
    The output vector `o` is an M dimensional binary vector, where oi = 1 if the ith subpopulation is predicted, and oi= 0 else.
    A subpopulation is considered predicted if it contains at least `ratio_fp_activation*n_E` neurons with a dAP.
    
    Parameters
    ----------
    somatic_spikes   : ndarray
        Lx2 array of spike senders somatic_spikes[:,0] and spike times somatic_spikes[:,1]
                       (L = number of spikes).
    dendriticAP      : ndarray
        Lx3 array of current senders dendriticAP[:,0], current times dendriticAP[:,1],
        and current dendriticAP[:,2] (L = number of recorded data points).
    dendriticAP_recording_times  : list
        list of list containing times at which the dendritic current is recorded for a given 
        element in each sequence
    characters_to_subpopulations : dict
    test_seq  : list
        list of list containing sequence elements
    params    : dict
        parameter dictionary
    """

    errors = [[] for _ in range(len(test_seq))]
    false_positives = [[] for _ in range(len(test_seq))]
    false_negatives = [[] for _ in range(len(test_seq))]
    last_char_active_neurons = [[] for _ in range(len(test_seq))]
    last_char_active_dendrites = [[] for _ in range(len(test_seq))]

    seqs = copy.copy(test_seq)

    if 'add_stimulus_noise' not in params:
        params['add_stimulus_noise'] = False

    for seq_num, seq in enumerate(test_seq):
        recording_times = dendriticAP_recording_times[seq_num]

        for it, rc_time in enumerate(recording_times):

            # find dendritic action potentials (dAPs)
            idx_q = np.where((dendriticAP[:, 1] < rc_time + params['idend_record_time'] + 1.) & 
                             (dendriticAP[:, 1] > rc_time))[0]

            idx_dAP = np.where(dendriticAP[:, 2][idx_q] > params['soma_params']['I_p'] - 1.)[0]
            
            senders_dAP = dendriticAP[:, 0][idx_q][idx_dAP]
 
            subpopulation_senders_dAP = [int((s - 1) // params['n_E']) for s in senders_dAP]

            # find somatic action potentials
            if params['add_stimulus_noise']:
                idx_soma = np.where((somatic_spikes[:, 1] < rc_time + params['DeltaT']+5) & 
                                    (somatic_spikes[:, 1] > rc_time + params['DeltaT']-5))[0]
            else:
                idx_soma = np.where((somatic_spikes[:, 1] < rc_time + 2*params['DeltaT']) & 
                                    (somatic_spikes[:, 1] > rc_time + params['DeltaT']))[0]

            senders_soma = somatic_spikes[:, 0][idx_soma]
            num_active_neurons = len(senders_soma)
            num_active_dendrites = len(senders_dAP)
 
            # create the target vector 
            excited_subpopulations = characters_to_subpopulations[seqs[seq_num][-1]]
            excited_subpopulations_prev = characters_to_subpopulations[seqs[seq_num][-2]]
            target = np.zeros(params['M'])
            target[excited_subpopulations] = 1

            # count false positives and construct the output vector
            output = np.zeros(params['M'])
            count_subpopulations = Counter(subpopulation_senders_dAP)
            counter_correct = 0

            #ratio_fn_activation = 0.8
            #ratio_fp_activation = 0.1
            ratio_fn_activation = 0.5
            ratio_fp_activation = 0.5

            for k, v in count_subpopulations.items():
                if k not in excited_subpopulations and v >= (ratio_fp_activation * params['pattern_size']):
                    #print('episode %d/%d count of a false positive %d, %d' % (it, len(recording_times), k, v))
                    output[k] = 1
                elif k in excited_subpopulations and v >= (ratio_fn_activation * params['pattern_size']):
                    counter_correct += 1

            # find false negatives
            if counter_correct == params['L']:
                output[excited_subpopulations] = 1
            #else:
            #    false_negative = 1

            error = 1/params['L'] * np.sqrt(sum((output - target) ** 2))
            false_positive = 1/params['L'] * sum(np.heaviside(output - target, 0))
            false_negative = 1/params['L'] * sum(np.heaviside(target - output, 0))

            # append errors, fp, and fn for the different sequences
            errors[seq_num].append(error)
            false_positives[seq_num].append(false_positive)
            false_negatives[seq_num].append(false_negative)
            last_char_active_neurons[seq_num].append(num_active_neurons)
            last_char_active_dendrites[seq_num].append(num_active_dendrites)

        print('#### Prediction performance ####')
        print('Sequence:', seqs[seq_num])
        print('Error:', errors[seq_num][-1])
        print('False positives:', false_positives[seq_num][-1])
        print('False negatives:', false_negatives[seq_num][-1])
        print('Number of active neurons in %s: %d' % (seqs[seq_num][-1], last_char_active_neurons[seq_num][-1]))
        print('Number of active dendrites in %s: %d' % (seqs[seq_num][-1], last_char_active_dendrites[seq_num][-1]))

    seq_avg_errors = np.mean(errors, axis=0)
    seq_avg_false_positives = np.mean(false_positives, axis=0)
    seq_avg_false_negatives = np.mean(false_negatives, axis=0)
    seq_avg_last_char_active_neurons = np.mean(last_char_active_neurons, axis=0)

    return seq_avg_errors, seq_avg_false_positives, seq_avg_false_negatives, seq_avg_last_char_active_neurons


###############################################################################
def hebbian_contribution(facilitate_factor, tau_plus, W_max, delta_t=40.):
    """Computes the increment of the facilitate function of the additive STDP 
    
    Parameters
    ----------
    facilitate_factor : float
    delta_T           : float
    tau_plus          : float
    W_max             : float

    Returns
    -------
    increment : float
    """

    increment = facilitate_factor * W_max * np.exp(-delta_t / tau_plus)
    #increment = facilitate_factor * W_max

    return increment


###############################################################################
def homeostasis_contribution(hs, Wmax=1, r_d=0, r_t=1):
    """ homeostasis plastic change

    Parameters
    ----------
    hs   : float
    r_d  : float            
    r_t  : float
    """

    return hs * (r_t - r_d) * Wmax


#################################################
def get_state_matrix(somatic_spikes,
                     seq_set_instances,
                     seq_set_instance_size,
                     params,
                     mode='train',
                     debug=False):
    """
    compute state matrix

    Parameters
    ----------
    test_sequences         : list
    times_somatic_spikes   : ndarray
    senders_somatic_spikes : ndarray
    excitation_times       : list

    Returns
    -------
    state matrix : numpy array
    """

    #assert len(excitation_times) >= 2, "excitation times need to contain at leasts 2 components"
    #DeltaT = excitation_times[1] - excitation_times[0] - stim_dur

    xt = []
    labels = []
    ls = []
    shift = 0.
    interval = 14.

    end_iterations = 0

    for q in range(seq_set_instance_size):

        # initialize state matrix
        length_seqs = len(seq_set_instances[q]['elements'])

        if len(somatic_spikes[0]) != 0:
            x = np.zeros((params['n_E']*params['M'], length_seqs))
            times_somatic_spikes = somatic_spikes[:, 1]
            senders_somatic_spikes = somatic_spikes[:, 0]
            # for each sequence in the test sequences

            acc_seq = 0
            ls = []
                # for each character in the sequence
            for k, (ele, time) in enumerate(zip(seq_set_instances[q]['elements'], seq_set_instances[q]['times'])):

                #indices_soma = np.where((times_somatic_spikes < (excitation_times[j] + s_t+min(interval, deltaT))) & 
                #                        (times_somatic_spikes > (excitation_times[j] + s_t-min(interval, s_t)-shift)))
                indices_soma = np.where((times_somatic_spikes < (time + interval)) & 
                                        (times_somatic_spikes > (time - shift)))
                senders_soma = np.array(senders_somatic_spikes[indices_soma], int)

                x[senders_soma-1, k] = 1
                tg = np.zeros(params['task']['vocabulary_size'])
                tg[ele-1] = 1 
                ls.append(tg)

            #acc_seq += len(seq)

        else:
            x = np.zeros((params['n_E']*params['M'], length_seqs))

        xt.append(x)
        labels.append(ls)

    if debug:
        print(f"plot somatic state matrix during {mode}: matrix_{mode}_soma.pdf")
        import matplotlib.pyplot as plt 

        plt.figure()
        plt.imshow(x[:,:20], aspect='auto', origin='lower')
        plt.xlabel("x")
        plt.ylabel("y")
        plt.colorbar()
        plt.savefig(f"matrix_{mode}_somatic.pdf")
        plt.close()

    return xt, labels

###############################################################################
def synaptic_plastic_change(facilitate_factor, tau_plus, w_max, hs, delta_t=40.):
    """ compute the plastic change due to Hebbian learning and homeostasis

    Parameters
    ----------
    facilitate_factor   : float
    tau_plus            : float
    w_max               : float
    hs                  : float
    delta_t             : float

    Returns
    -------
    w_tot               : float
    """

    w_inc = hebbian_contribution(facilitate_factor, tau_plus, w_max, delta_t)
    w_hom = homeostasis_contribution(hs, w_max)

    w_tot = w_inc + w_hom

    return w_tot
