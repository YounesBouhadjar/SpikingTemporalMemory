import os
import sys
import copy
import pdb
import matplotlib.pyplot as plt
import matplotlib
from scipy.stats import binom
from scipy.special import comb
import numpy as np
from tqdm import tqdm


def plastic_change(facilitate_factor, delta_t, tau_plus, W_max, hs=0.3, depression_factor=4): 
    """
    Computes the plastic change of the synapse, includes stdp potentiation, homeostasis and depression 
    
    Parameters: 
    -----------
    facilitate_factor : float
    delta_T           : float
    tau_plus          : float
    W_max             : float
    hs                : float
    alpha             : float

    Returns:
    --------
    increment : float

    """

    weight_increment = facilitate_factor * W_max * np.exp(-delta_t/tau_plus) + hs
    constant_depression = weight_increment/depression_factor

    return weight_increment - constant_depression 


def measure_patterns_uniqueness(pattern_size, population_size, connection_probability, 
                                spike_threshold, delta_perm_increase, scale_initial_perm, P_max, p_th):
    """
    Computes the uniqueness of patterns as a response to a given set of patterns

    with M = population_size, s = pattern_size, p=connection_probability.

    Parameters:
    -----------
    population_size        : int
    pattern_size           : int
    connection_probability : float 
    spike_threshold        : float
    delta_perm_increase  : float
    scale_initial_perm   : float, or array(float)
    W_max                  : float
                  total synaptic strength of incoming synapses to a post-neuron 
    
    Returns:
    -------
    steps : array(int)
        training steps
    
    sparsity : array(int)
        number of active neurons per population after convergence 

    """

    training_steps = 100
    num_subpopulations = 2 
    sparsity = []
    steps = []
    conn_skelton = np.random.choice([1, 0], size=(population_size, population_size), p=[connection_probability, 1-connection_probability])
    permanences = scale_initial_perm * np.random.rand(population_size, population_size)
    initial_conn = conn_skelton * permanences 
    normalize = False

    # TODO: this is only a first test
    # we could generalize this to N patterns
    # with a controllable amount of overlap
    pattern_1 = np.arange(0, pattern_size) 
    pattern_2 = np.arange(pattern_size, 2*pattern_size)


    overlaps = []

    for delta in delta_perm_increase: 
        conn = copy.copy(initial_conn)
        continue_1 = True
        continue_2 = True
        for i in range(training_steps):
            
            # find potential connectivity
            x = np.where(conn == 0)

            # synaptic facilitation
            if continue_1:
                conn[pattern_1] += delta

            if continue_2:
                conn[pattern_2] += delta
            
            conn[x] = 0
            conn[conn>P_max] = P_max 
            
            # apply synaptic normalization
            #TODO

            # find connected synapses
            conn_connected = conn>p_th

            conn_connected_1 = conn_connected[pattern_1]
            conn_connected_2 = conn_connected[pattern_2]

            active_neurons_1 = np.sum(conn_connected_1, axis=0) >= spike_threshold
            active_neurons_2 = np.sum(conn_connected_2, axis=0) >= spike_threshold

            num_active_neurons_1 = np.sum(active_neurons_1)
            num_active_neurons_2 = np.sum(active_neurons_2)

            if num_active_neurons_1 >= pattern_size:
                continue_1 = False

            if num_active_neurons_2 >= pattern_size:
                continue_2 = False
        
            overlap = len(np.where((active_neurons_1 == True) & (active_neurons_2 == True))[0])
            if not(continue_1) and not(continue_2):
                break
        
        steps.append(i)
        overlaps.append(overlap)
 
    return overlaps, steps

def measure_patterns_uniqueness_trials(pattern_size, population_size, connection_probability, spike_threshold, delta_perm_increase, scale_initial_perm, W_max, p_th):
    """
    Computes the uniqueness of patterns as a response to a given set of patterns after training for a number of trials

    Parameters:
    -----------
    population_size        : int
    pattern_size           : int
    connection_probability : float 
    spike_threshold        : float
    delta_perm_increase  : float
    scale_initial_perm   : float, or array(float)
    W_max                  : float
    
    Returns:
    -------
    steps : array(int)
        training steps
    
    sparsity : array(int)
        number of active neurons per population after convergence 
    """

    trials = 20
    all_steps = []
    all_sparsity = []
    for i in tqdm(range(trials)):
        steps, sparsity = sparsity_measure(pattern_size, population_size, connection_probability, spike_threshold, delta_perm_increase, scale_initial_perm, W_max, p_th)

        all_steps.append(steps)
        all_sparsity.append(sparsity)
    
    mean_steps = np.mean(all_steps, 0)
    mean_sparsity = np.mean(all_sparsity, 0)

    return mean_steps, mean_sparsity 

# set parameters 
population_size = 150         # number of neurons per subpopulation      
pattern_size = 20             # pattern size
connection_probability = 0.3  # connection probability
prediction_threshold = 5      # correspond to params["soma_params"]["theta_dAP"] / params["syn_dict_ee"]['Wmax'] 
max_initial_perm = 8.

facilitate_factor = np.arange(0.01,0.11,0.01)
tau_plus = 20.
delta_t = 40
P_max = 20.
th_perm = 10.

def example1():

    delta_perm_increase = plastic_change(facilitate_factor, delta_t, tau_plus, P_max)

    overlaps, steps = measure_patterns_uniqueness(pattern_size, population_size, 
                                                  connection_probability, prediction_threshold, 
                                                  delta_perm_increase, max_initial_perm, P_max, th_perm)

    fig = plt.figure(1,figsize=(7, 4))
    plt.clf()
    ax1 = fig.add_subplot(111)

    ax1.plot(facilitate_factor, overlaps,'-',lw=2, color='0.0')
    ax1.set_ylabel('pattern overlap')
    ax1.set_ylim(0, max(overlaps)+1.)
    ax1.hlines(y=pattern_size, xmin=facilitate_factor[0], xmax=facilitate_factor[-1], linestyles='dotted', lw=1)

    ax2 = ax1.twinx()
    ax2.plot(facilitate_factor, steps,'-',lw=2, color='0.7') 
    ax2.set_ylabel('steps to convergence', color='0.7')
    ax2.set_ylim(0, max(steps)+1.)
    ax2.tick_params(axis='y', labelcolor='0.6')

    ax1.set_xlim(facilitate_factor[0], facilitate_factor[-1])
    ax1.set_xlabel('facilitate factor $\lambda$')
    #plt.legend(loc=1)
    plt.title(r'$M=%d$, $s=%d$, $p=%0.1f$' % (population_size, pattern_size, connection_probability)) 

    path = 'img'
    fname = 'patterns_uniqueness'
    print('save figure to %s/%s.pdf ...' % (path, fname))
    os.system('mkdir -p %s' % (path))
    plt.savefig('%s/%s.pdf' % (path, fname), bbox_inches = 'tight')

example1() 
