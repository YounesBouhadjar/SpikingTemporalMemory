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

"""PyNEST Spiking-TemporalMemory: Model Class [1]
-------------------------------------------------

Main file of the Spiking-TemporalMemory defining ``Model`` class with function
to build, connect and simulate the network.

References
~~~~~~~~~~

.. [1] Bouhadjar Y, Wouters DJ, Diesmann M, Tetzlaff T (2021), Sequence learning, prediction, and replay
       in networks of spiking neurons

Authors
~~~~~~~

Younes Bouhadjar
"""

import random
import nest
import copy
import numpy as np
from collections import defaultdict
from collections import Counter

from shtm import helper
from fna.decoders.readouts import Readout

import sys
import wandb
sys.path.append("../../figures")
from utils import plot_spikes
import matplotlib.pyplot as plt

class Model:
    """Instantiation of the Spiking-TemporalMemory model and its PyNEST implementation.

    the model provides the following member functions: 

    __init__(parameters)
    create()
    connect()
    simulate(t_sim)

    In addition, each model may implement other model-specific member functions.
    """

    def __init__(self, params, vocabulary):
        """Initialize model and simulation instance, including

        1) parameter setting,
        2) generate sequence data,
        3) configuration of the NEST kernel,
        4) setting random-number generator seed, and

        Parameters
        ----------
        params:    dict
                   Parameter dictionary
        """

        print('\nInitialising model and simulation...')

        # set parameters derived from base parameters
        self.params = helper.derived_parameters(params)

        # data directory
        if self.params['evaluate_replay']:
            self.data_path = helper.get_data_path(self.params['data_path'],
                                                  self.params['label'],
                                                  'replay')
        else:
            self.data_path = helper.get_data_path(self.params['data_path'],
                                                  self.params['label'])

        if nest.Rank() == 0:
            if self.data_path.is_dir():
                message = "Directory already existed."
                if self.params['overwrite_files']:
                    message += "Old data will be overwritten."
            else:
                self.data_path.mkdir(parents=True, exist_ok=True)
                message = "Directory has been created."
            print("Data will be written to: {}\n{}\n".format(self.data_path, message))

        # set network size
        self.num_subpopulations = params['M']
        self.num_exc_neurons = params['n_E'] * self.num_subpopulations

        # initialize RNG        
        np.random.seed(self.params['seed'])
        random.seed(self.params['seed'])

        self.vocabulary = vocabulary
        self.vocab_size = len(vocabulary)
        #self.length_sequence = len(self.sequences[0])
        #self.num_sequences = len(self.sequences)

        # initialize the NEST kernel
        self.__setup_nest()

        # get time constant for dendriticAP rate
        #self.params['soma_params']['tau_h'] = self.__get_time_constant_dendritic_rate(
        #    calibration=self.params['calibration'])


    def __setup_nest(self):
        """Initializes the NEST kernel.
        """

        #nest.ResetKernel()
        nest.SetKernelStatus({
            'resolution': self.params['dt'],
            'print_time': self.params['print_simulation_progress'],
            'local_num_threads': self.params['n_threads'],
            'rng_seed': self.params['seed'],
            'dict_miss_is_error': True,
            'overwrite_files': self.params['overwrite_files'],
            'data_path': str(self.data_path),
            'data_prefix': ''
        })
        nest.set_verbosity("M_ERROR")

    def create(self):
        """Create and configure all network nodes (neurons + recording and stimulus devices)
        """

        print('\nCreating and configuring nodes...')

        # create excitatory population
        self.__create_neuronal_populations()

        # create spike generators
        self.__create_spike_generators()

        # create recording devices
        self.__create_recording_devices()

#        # create weight recorder
#        if self.params['active_weight_recorder']:
#            self.__create_weight_recorder()
#
#        if self.params['add_bkgd_noise']:
#            self.__create_noise_sources()
#
    def connect(self):
        """Connects network and devices
        """

        print('\nConnecting network and devices...')
        # TODO: take into account L (number of subpopulations per character) when connecting the network

        # connect excitatory population (EE)
        if self.params['load_connections']:
            self.__load_connections(label='ee_connections')
        else:
            self.__connect_excitatory_neurons()

            # connect inhibitory population (II, EI, IE)
        self.__connect_inhibitory_neurons()

        # connect external input
        self.__connect_external_inputs_to_subpopulations()

        # connect neurons to the spike recorder
        nest.Connect(self.exc_neurons, self.spike_recorder_soma)
        nest.Connect(self.inh_neurons, self.spike_recorder_inh)

        nest.Connect(self.multimeter_idend, self.exc_neurons)

        self.__set_min_synaptic_strength()


    def simulate(self, sim_time, save_data=False):
        """Run simulation.
        """
        # the simulation time is set during the creation of the network  
        if nest.Rank() == 0:
            print('\nSimulating {} ms.'.format(sim_time))
        nest.Simulate(sim_time)

        if save_data:
            # record somatic spikes
            times = nest.GetStatus(self.spike_recorder_soma)[0]['events']['times']
            senders = nest.GetStatus(self.spike_recorder_soma)[0]['events']['senders']
            data = np.array([senders, times]).T

            fname = 'somatic_spikes'
            print("save data to %s/%s ..." % (self.data_path, fname))
            np.save('%s/%s' % (self.data_path, fname), data)

            # record dendritic current
            senders = nest.GetStatus(self.multimeter_idend)[0]['events']['senders']
            times = nest.GetStatus(self.multimeter_idend)[0]['events']['times']
            I_dends = nest.GetStatus(self.multimeter_idend)[0]['events']['I_dend']

            data = np.array([senders, times, I_dends]).T
                    
            fname = 'idend'
            print("save data to %s/%s ..." % (self.data_path, fname))
            np.save('%s/%s' % (self.data_path, fname), data)

        #else:
            # reset recorders
        #    self.spike_recorder_soma.n_events = 0
        #    self.multimeter_idend_last_episode.n_events = 0
        #    self.multimeter_idend.n_events = 0

    def __create_neuronal_populations(self):
        """'Create neuronal populations
        """

        # create excitatory population
        self.exc_neurons = nest.Create(self.params['soma_model'],
                                       self.num_exc_neurons,
                                       params=self.params['soma_params'])

        # create inhibitory population
        self.inh_neurons = nest.Create(self.params['inhibit_model'],
                                       self.params['n_I'] * self.num_subpopulations,
                                       params=self.params['inhibit_params'])

    def __create_spike_generators(self):
        """Create spike generators
        """

        # create external spike sources
        self.input_excitation_soma = nest.Create('spike_generator', self.vocab_size)
        if self.params['sparse_first_char']:
            self.input_excitation_dend = nest.Create('spike_generator', self.vocab_size)


    def set_input(self, element_activations, seq_set_instances):

        # round element activation times to simulation grid
        element_activations[:, 1] = np.round(element_activations[:, 1]/self.params['dt'])*self.params['dt']
        self.element_activations = element_activations

        # set element activation times
        for cx in range(len(self.input_excitation_soma)):
            act_times = np.array(self.element_activations[self.element_activations[:, 0] == cx, 1])
            self.input_excitation_soma[cx].set({'spike_times': np.sort(act_times)})

        if self.params['sparse_first_char']:
            seq_set_instances_size = len(seq_set_instances.keys())

            first_eles = np.array([seq_set_instances[q]['elements'][0] for q in range(seq_set_instances_size)])
            first_times = np.array([round(seq_set_instances[q]['times'][0], 1) - self.params['time_dend_to_somatic'] for q in range(seq_set_instances_size)])
            
            counts_eles = Counter(first_eles)

            #self.input_excitation_dend = {}
            for ele, v in counts_eles.items():
                
                x = np.where(first_eles == ele)
                nest.SetStatus(self.input_excitation_dend[ele], {'spike_times': first_times[x]})


    def __create_recording_devices(self):
        """Create recording devices
        """

        self.spike_recorder_soma = nest.Create('spike_recorder')
            
        self.multimeter_idend = nest.Create('multimeter',
                                            params={'record_from': ['I_dend'],
                                                    'interval': self.params['idend_recording_interval']})

        # create a spike recorder for inh neurons
        self.spike_recorder_inh = nest.Create('spike_recorder')


    def __create_weight_recorder(self):
        """Create weight recorder
        """

        self.wr = nest.Create('weight_recorder')
        #self.params['syn_dict_ee']['weight_recorder'] = self.wr
        nest.CopyModel(self.params['syn_dict_ee']['synapse_model'],
                       'stdsp_synapse_rec', {'weight_recorder': self.wr})
        self.params['syn_dict_ee']['synapse_model'] = 'stdsp_synapse_rec'

    def __create_noise_sources(self):
        """Create noise sources
        """

        # create poisson generator
        self.poisson_generator = nest.Create('poisson_generator')
        nest.SetStatus(self.poisson_generator, {'rate': self.params['bkgd_noise']['rate']})

    def __get_subpopulation_neurons(self, index_subpopulation):
        """Get neuron's indices (NEST NodeCollection) belonging to a subpopulation
        
        Parameters
        ---------
        index_subpopulation: int

        Returns
        -------
        NEST NodeCollection
        """

        neurons_indices = [int(index_subpopulation) * self.params['n_E'] + i for i in
                           range(self.params['n_E'])]

        return self.exc_neurons[neurons_indices]

    def __connect_excitatory_neurons(self):
        """Connect excitatory neurons
        """

        if 'delay' in self.params['syn_dict_ee'].keys():
            nest.CopyModel(self.params['syn_dict_ee']['synapse_model'],
                           'stdsp_synapse_d', {'d': self.params['syn_dict_ee']['delay']})
            self.params['syn_dict_ee']['synapse_model'] = 'stdsp_synapse_d'

        nest.Connect(self.exc_neurons, self.exc_neurons,
                     conn_spec=self.params['conn_dict_ee'],
                     syn_spec=self.params['syn_dict_ee'])

    def __connect_inhibitory_neurons(self):
        """Connect inhibitory neurons
        """

        for k, subpopulation_index in enumerate(range(self.num_subpopulations)):
            # connect inhibitory population 
            subpopulation_neurons = self.__get_subpopulation_neurons(subpopulation_index)

            # connect neurons within the same mini-subpopulation to the inhibitory population
            nest.Connect(subpopulation_neurons, self.inh_neurons[k], syn_spec=self.params['syn_dict_ie'])

            # connect the inhibitory neurons to the neurons within the same mini-subpopulation
            nest.Connect(self.inh_neurons[k], subpopulation_neurons, syn_spec=self.params['syn_dict_ei'])

    def __connect_external_inputs_to_subpopulations(self):
        """Connect external inputs to subpopulations
        """

        for subpopulation_index, sub_inp_excitation in enumerate(self.input_excitation_soma):

            subpopulation_neurons = self.__get_subpopulation_neurons(subpopulation_index)

            # receptor type 1 correspond to the feedforward synapse of the 'iaf_psc_exp_multisynapse' model
            nest.Connect(sub_inp_excitation, subpopulation_neurons,
                         self.params['conn_dict_ex'], syn_spec=self.params['syn_dict_ex'])
                #nest.Connect(self.input_excitation_dend[char], subpopulation_neurons,
                #             self.params['conn_dict_edx'], syn_spec=self.params['syn_dict_edx'])

                #if self.params['add_stimulus_noise']:
                #    nest.Connect(self.input_excitation_soma[char], self.packet_parrot)

        if self.params['sparse_first_char']:
            #for subpopulation_index in self.input_excitation_dend.keys():
            for subpopulation_index in range(self.vocab_size):
                subpopulation_neurons = self.__get_subpopulation_neurons(subpopulation_index)
                nest.Connect(self.input_excitation_dend[subpopulation_index], subpopulation_neurons,
                                 self.params['conn_dict_edx'], syn_spec=self.params['syn_dict_edx'])

    def __connect_noise_sources(self):
        """Connect noise source
        """
 
        conn = {'rule': 'all_to_all'}
            
        # set parameters
        exc_weight = self.params['bkgd_noise']['J']
        inh_weight = -exc_weight
        inh_syn_spec = {'receptor_type':4, 'delay': 0.1, 'weight': inh_weight}
        exc_syn_spec = {'receptor_type':4, 'delay': 0.1, 'weight': exc_weight}
            
        # connect poisson generator
        nest.Connect(self.poisson_generator, self.exc_neurons, conn, syn_spec=inh_syn_spec) 
        nest.Connect(self.poisson_generator, self.exc_neurons, conn, syn_spec=exc_syn_spec) 

    def __set_min_synaptic_strength(self):
        """Set synaptic Wmin
        """

        print('\nSet min synaptic strength ...')
        connections = nest.GetConnections(synapse_model=self.params['syn_dict_ee']['synapse_model'])

        syn_model = self.params['syn_dict_ee_synapse_model']
        if syn_model[:5] == 'stdsp':
            connections.set({'Pmin': connections.p})
        else:
            connections.set({'Wmin': connections.weight})

    def get_sparsity_level(self):
    
        somatic_spikes = helper.load_numpy_spike_data(self.data_path, 'somatic_spikes')

        return spr_level

    def measure_fp_fn(self, S, seq_set_instances, seq_set_instance_id, load_data=False):
   
        if load_data:
            somatic_spikes = helper.load_numpy_spike_data(self.data_path, 'somatic_spikes')
            #dendritic_current = helper.load_numpy_spike_data(self.data_path, 'idend_last_episode')
            dendritic_current = helper.load_numpy_spike_data(self.data_path, 'idend')
        else:
            # record somatic spikes
            times = nest.GetStatus(self.spike_recorder_soma)[0]['events']['times']
            senders = nest.GetStatus(self.spike_recorder_soma)[0]['events']['senders']

            somatic_spikes = np.array([senders, times]).T

            #  record dendritic currents
            senders = nest.GetStatus(self.multimeter_idend)[0]['events']['senders']
            times = nest.GetStatus(self.multimeter_idend)[0]['events']['times']
            i_dends = nest.GetStatus(self.multimeter_idend)[0]['events']['I_dend']

            dendritic_current = np.array([senders, times, i_dends]).T

            self.spike_recorder_soma.n_events = 0
            self.multimeter_idend.n_events = 0

        error, fn, fp = helper.measure_fp_fn(somatic_spikes,
                                             dendritic_current,
                                             S,
                                             seq_set_instances,
                                             seq_set_instance_id,
                                             self.params,
                                             mode='train')
        return error, fn, fp

    def load_resampled_data(self, seq_set, seq_set_instances, seq_set_instance_size):
    
        somatic_spikes = helper.load_numpy_spike_data(self.data_path, 'somatic_spikes')

        state_matrix_soma, labels, sps = helper.get_state_matrix(somatic_spikes,
                                                                 seq_set,
                                                                 seq_set_instances,
                                                                 seq_set_instance_size,
                                                                 self.params,
                                                                 mode='train')
        return state_matrix_soma, labels, sps

    def plot_readout(self, readout, state_matrix):

        if not isinstance(readout.fit_obj, np.ndarray):
            output = readout.fit_obj.predict(state_matrix.T)
        else:
            output = np.dot(readout.weights.weights.T, state_matrix)

        fig=plt.figure()
        plt.imshow(output.T, aspect='auto', origin='lower')

        plt.ylabel("readout neurons")
        plt.xlabel("time")

        print("test_readout.pdf")
        plt.savefig("test_readout.pdf")

        wandb.log({"fig_r": wandb.Image(fig)})

    def plot_activity(self, stop, duration=500, load_data=False):

        if load_data:

            somatic_spikes = helper.load_numpy_spike_data(self.data_path, 'somatic_spikes')
            #dendritic_current = helper.load_numpy_spike_data(self.data_path, 'idend_last_episode')
            try:
                #dendritic_current = helper.load_numpy_spike_data(self.data_path, 'idend_last_episode')
                dendritic_current = helper.load_numpy_spike_data(self.data_path, 'idend')
            except:
                dendritic_current = [[]]

        else:

            # record somatic spikes
            times = nest.GetStatus(self.spike_recorder_soma)[0]['events']['times']
            senders = nest.GetStatus(self.spike_recorder_soma)[0]['events']['senders']

            somatic_spikes = np.array([senders, times]).T

            #  record dendritic currents
            senders = nest.GetStatus(self.multimeter_idend)[0]['events']['senders']
            times = nest.GetStatus(self.multimeter_idend)[0]['events']['times']
            i_dends = nest.GetStatus(self.multimeter_idend)[0]['events']['I_dend']

            dendritic_current = np.array([senders, times, i_dends]).T

            self.spike_recorder_soma.n_events = 0
            self.multimeter_idend.n_events = 0


        start_time = stop - duration
        end_time = stop

        fig = plot_spikes(somatic_spikes, [[]],
                    dendritic_current,
                    start_time,
                    end_time,
                    self.params['soma_params']['I_p']-5,
                    self.params['M']*self.params['n_E'],
                    self.params['M'])

        subpopulation_indices_background = np.arange(self.params['M'])*self.params['n_E']

        for i in range(self.params['M'])[::2]:
            plt.axhspan(subpopulation_indices_background[i],
                        subpopulation_indices_background[i]+self.params['n_E'],
                        facecolor='0.2',
                        alpha=0.1)

        print("test_net_activity.pdf")
        plt.savefig("test_net_activity.pdf")

        wandb.log({"fig_a": wandb.Image(fig)})

        plt.close()

    def train_readout(self, state_matrix, labels, num_ele=10):

        r_params = {}
        r_params["task"] = "prediction"
        #r_params["algorithm"] = "ridge"
        r_params["algorithm"] = "pinv"
        r_params["extractor"] = "nothing-here"
        rng = np.random.default_rng(0)  # doesn't matter, just make it reproducible

        # create new Readout for each subtask
        readout = Readout(f"readout-offline-prediction", r_params, rng)

        ls = np.concatenate(labels[-num_ele:], axis=0) #np.array(labels[i])
        state_matrices = np.concatenate(state_matrix[-num_ele:], axis=1)

        #ls = np.concatenate(labels, axis=0) #np.array(labels[i])
        #state_matrices = np.concatenate(state_matrix, axis=1)

        readout.train("batch_label", state_matrices, ls)

        performance = readout.evaluate(process_output_method="k-WTA",
                                       symbolic=True,
                                       vocabulary=self.vocabulary)

        acc = abs(performance['label']['cc'])
        mse = performance['raw']['MSE']
        print(f'Performance accuracy: {acc} and mse: {mse}')

        ############################
        if False:
            fig=plt.figure()

            plt.pcolor(state_matrices,
                      # aspect='auto',
                       #origin='lower',
                       vmin = 0, 
                       vmax = 1,
                       rasterized=True)
                       #interpolation='nearest')

            plt.ylabel("neurons")
            plt.xlabel("time")

            print("test_state_1.png")
            plt.savefig("test_state_1.png", dpi=300)

            wandb.log({"fig_s": wandb.Image(fig)})

            fig=plt.figure()
            plt.imshow(ls.T, aspect='auto', origin='lower')

            plt.ylabel("neurons")
            plt.xlabel("time")

            #print("test_state.pdf")
            #plt.savefig("test_state.pdf")

            wandb.log({"fig_l": wandb.Image(fig)})

        return acc, mse, readout

    def save_connections(self, fname='ee_connections'):
        """Save connection matrix

        Parameters
        ----------
        label: str
            name of the stored file
        """

        print('\nSave connections ...')
        connections_all = nest.GetConnections(synapse_model=self.params['syn_dict_ee']['synapse_model'])

        if self.params['syn_dict_ee_synapse_model'][:5] == 'stdsp':
            connections = nest.GetStatus(connections_all, ['target', 'source', 'weight', 'p'])
        else:
            connections = nest.GetStatus(connections_all, ['target', 'source', 'weight'])

        np.save('%s/%s' % (self.data_path, fname), connections)

    def __load_connections(self, label='ee_connections'):
        """Load connection matrix
        
        Parameters
        ----------
        label: str
            name of the stored file
        """

        assert self.params['syn_dict_ee']['synapse_model'] != 'stdsp_synapse_rec', "synapse model not tested yet"

        print('\nLoad connections ...')
        data_path = helper.get_data_path(self.params['data_path'], self.params['label'])
        #conns = np.load('%s/%s.npy' % (data_path, label))
        conns = np.load('/work/users/bouhadjar/data/sequence_learning_performance/sequence_learning_and_prediction_task_complexity_2/80e0201deb1a3d50935aca587d82bb6a/ee_connections.npy')
        conns_tg = [int(conn[0]) for conn in conns]
        conns_src = [int(conn[1]) for conn in conns]
        conns_weights = [conn[2] for conn in conns]

        if self.params['syn_dict_ee_synapse_model'][:5] == 'stdsp':
            conns_perms = [conn[3] for conn in conns]

        if self.params['evaluate_replay']:
            syn_dict = {'receptor_type': 2,
                        'delay': [self.params['syn_dict_ee']['delay']] * len(conns_weights),
                        'weight': conns_weights}
            nest.Connect(conns_src, conns_tg, 'one_to_one', syn_dict)
        else:
            del self.params['syn_dict_ee']['p']
            syn_dict_ee = copy.deepcopy(self.params['syn_dict_ee'])

            del syn_dict_ee['synapse_model']
            #del syn_dict_ee['w']
            del syn_dict_ee['receptor_type']
            #if self.params['syn_dict_ee']['synapse_model'] == 'stdsp_synapse':
            #    del syn_dict_ee['permanence']

            nest.SetDefaults(self.params['syn_dict_ee']['synapse_model'], syn_dict_ee)

            if self.params['syn_dict_ee_synapse_model'][:5] == 'stdsp':
                syn_dict = {'synapse_model': self.params['syn_dict_ee']['synapse_model'],
                            'receptor_type': 2,
                            'weight': conns_weights,
                            'p': conns_perms}
            else:
                syn_dict = {'synapse_model': 'stdsp_synapse',
                            'receptor_type': 2,
                            'weight': conns_weights}

            nest.Connect(conns_src, conns_tg, 'one_to_one', syn_dict)

    def __get_time_constant_dendritic_rate(self, DeltaT=40., DeltaT_seq=100., calibration=100, target_firing_rate=1):
        """Compute time constant of the dendritic AP rate,

        The time constant is set such that the rate captures how many dAPs a neuron generated
        all along the period of a batch
         
        Parameters
        ----------
        calibration : float
        target_firing_rate : float

        Returns
        -------
        float
           time constant of the dendritic AP rate
        """

        t_exc = ((self.length_sequence-1) * DeltaT + DeltaT_seq + calibration) \
                * self.num_sequences

        print("\nDuration of a sequence set %d ms" % t_exc)

        return target_firing_rate * t_exc


    def __normalize_incoming_weights(self): 
        """Normalizes weights of incoming synapses
        """
 
        for i, neuron in enumerate(self.exc_neurons):

            conn = nest.GetConnections(target=neuron, synapse_model='stdsp_synapse')
            per = np.array(conn.permanence)
            x = np.where(per > self.params['syn_dict_ee']['th_perm'])

            s = len(x[0])
            per_dep = per - self.params['p_target']
            per_norm = per_dep * sig(self.params['lr']-s) #/ y

            nest.SetStatus(conn, 'permanence', per_norm)

##############################################
def get_parameters():
    """Import model-parameter file.

    Returns
    -------
    params: dict
        Parameter dictionary.
    """

    import parameters_space
    params = parameters_space.p

    return params


###########################################
def sig(x):
     return 1/(1 + np.exp(-x))


###########################################
def load_input_encoding(path, fname):
    """Load input encoding: association between sequence element and subpopulations

    Parameters
    ----------
    path: str
    fname: str

    Returns
    -------
    characters_to_subpopulations: dict
    """

    characters_to_subpopulations = helper.load_data(path, fname)

    return characters_to_subpopulations
