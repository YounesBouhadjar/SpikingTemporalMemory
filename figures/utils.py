import os
import sys
import copy
import hashlib
import getpass
import numpy as np
import parameters as para
import matplotlib.pyplot as plt
from pathlib import Path
from pprint import pformat
from matplotlib import gridspec
from collections import defaultdict
from matplotlib.lines import Line2D

from shtm.helper import load_data, load_spike_data, compute_prediction_performance


###############################################################################
def mean_confidence_interval(y, use_percentile=True):
    """
    Computes the mean and standard deviation or median and percentile (5%, 95%) of the variable y
    

    Parameters
    ----------
    y                : ndarray
    use_percentile   : bool

    Returns
    -------
    mean_y       : ndarray
    mean_y_lower : ndarray
    mean_y_upper : ndarray
    """

    if use_percentile:
        mean_y = np.median(y, axis=0)
        mean_y_lower = np.percentile(y, 5, axis=0)
        mean_y_upper = np.percentile(y, 95, axis=0)
    else:
        mean_y = np.mean(y, axis=0)
        std_y = np.std(y, axis=0)
        mean_y_lower = mean_y - std_y
        mean_y_upper = mean_y + std_y

    return mean_y, mean_y_lower, mean_y_upper


###############################################################################
def running_mean(x, y, N):
    """Compute the moving average of a time series 
    
    Parameters
    ----------
    x : ndarray 
    y : ndarray
    N : integer 

    Returns
    -------
    x_mv : ndarray 
    (mean_y_mv, mean_y_lower_mv, mean_y_upper_mv) : list
    """

    mean_y, mean_y_lower, mean_y_upper = y
    
    mean_y_mv = np.convolve(mean_y, np.ones(N)/N, mode='valid')
    mean_y_lower_mv = np.convolve(mean_y_lower, np.ones(N)/N, mode='valid')
    mean_y_upper_mv = np.convolve(mean_y_upper, np.ones(N)/N, mode='valid')
    
    len_mv = len(mean_y_mv)
    x_mv = x[:len_mv]

    return x_mv, (mean_y_mv, mean_y_lower_mv, mean_y_upper_mv)


###############################################################################
def letters_to_active_neurons(test_sequences, times_somatic_spikes, senders_somatic_spikes, excitation_times,
                              fixed_somatic_delay):
    """Finds the active neurons of each element in the sequences and return their indices

    #TODO

    Parameters
    ----------
    test_sequences         : list
    times_somatic_spikes   : ndarray
    senders_somatic_spikes : ndarray
    excitation_times       : list
    fixed_somatic_delay    : float

    Returns
    -------
    seqs_letters_to_active_neurons : list
    """

    seqs_letters_to_active_neurons = []
    end_iterations = 0

    # for each sequence in the test sequences
    for seq in test_sequences:
        start_iterations = end_iterations
        end_iterations += len(seq)
        letters_to_active_neurons = defaultdict(list)

        # for each character in the sequence
        for k, (j, char) in enumerate(zip(range(start_iterations, end_iterations), seq)):
            indices_soma = np.where((times_somatic_spikes < excitation_times[j] + fixed_somatic_delay) & (
                    times_somatic_spikes > excitation_times[j]))
            senders_soma = senders_somatic_spikes[indices_soma]

            letters_to_active_neurons[char] = senders_soma

        seqs_letters_to_active_neurons.append(letters_to_active_neurons)

    return seqs_letters_to_active_neurons


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
def plot_spikes(somatic_spikes, inh_spikes, dendritic_current, start_time, end_time, dAP_threshold, n, m):
    """Plot somatic spikes and dendritic action potential

    Parameters
    ----------
    somatic_spikes   : ndarray
        Lx2 array of spike senders somatic_spikes[:,0] and spike times somatic_spikes[:,1]
        (L = number of spikes).
    inh_spikes   : ndarray
        Lx2 array of spike senders somatic_spikes[:,0] and spike times somatic_spikes[:,1]
        (L = number of spikes).
    dendritic_current : ndarray
        Lx3 array of current senders dendriticAP[:,0], current times dendriticAP[:,1],
        and current dendriticAP[:,2] (L = number of recorded data points).
    start_time : float
        time from which start plotting somatic_spikes and dendritic_current
    end_time   : float
        time at which stop plotting somatic_spikes and dendritic_current
    dAP_threshold : float
        current threshold for generating dendriticAP
    n             : int
        number of neurons per subpopulation
    m             : int
        number of subpopulations
    """

    plt.figure(constrained_layout=True)

    # select data to show
    if len(inh_spikes[0]) != 0:
        idx_inh = np.where((inh_spikes[:, 1] > start_time) & (inh_spikes[:, 1] < end_time))
        plt.plot(inh_spikes[:, 1][idx_inh], inh_spikes[:, 0][idx_inh], 'o', color='green', lw=0., ms=1.)
   
    if len(somatic_spikes[0]) != 0:
        idx_somatic = np.where((somatic_spikes[:, 1] > start_time) & (somatic_spikes[:, 1] < end_time))
        plt.plot(somatic_spikes[:, 1][idx_somatic], somatic_spikes[:, 0][idx_somatic], 'o', color='red', lw=0., ms=1., zorder=2,
                label='somatic_spikes')

    if len(dendritic_current[0]) != 0:    
        ind = np.where((dendritic_current[:, 2] > dAP_threshold))[0]
        dendriticAP_times = dendritic_current[:, 1][ind]
        dendriticAP_currents = dendritic_current[:, 0][ind]
        idx_dAP = np.where((dendriticAP_times > start_time) & (dendriticAP_times < end_time))
        plt.plot(dendriticAP_times[idx_dAP], dendriticAP_currents[idx_dAP], 'o', color='#00B4BE', lw=0., ms=0.5, zorder=1,
                label='dendriticAP')

    plt.xlabel('time (ms)')

    legend_elements = [Line2D([0], [0], marker='o', color='red', ms=1, lw=0., label='somatic_spikes'),
                       Line2D([0], [0], color='#00B4BE', lw=1., label='dendriticAP')]

    plt.legend(handles=legend_elements, loc='best')

    # plt.ylabel('neuron id')
    # plt.legend()


###############################################################################
def plot_data(x, y, xlabel, ylabel, title, saving_paths, figure_name, label_z_scan=None, values_z_scan=None,
              num_figure=None):
    """ Plot data

    Parameters
    ----------
    x       : array
    y       : array
    xlabel  : string
    ylabel  : string
    title   : string
    saving_paths : list of string
    figure_name  : string
    label_z_scan : string
    values_z_scan : array
    num_figure    : int
    """

    plt.figure(num_figure, constrained_layout=True)

    mean_y, mean_y_lower, mean_y_upper = y

    if values_z_scan == None:
        plt.plot(x, mean_y, lw=1.5, color='black')
        plt.fill_between(x, mean_y_lower, mean_y_upper, facecolor='grey', alpha=0.2)

    else:
        # Have a look at the colormaps here and decide which one you'd like:
        # http://matplotlib.org/1.2.1/examples/pylab_examples/show_colormaps.html
        colormap = plt.cm.gray
        step = 100
        num_plots = len(values_z_scan)
        tt_steps = num_plots * step
        colors = [colormap(i) for i in range(0, tt_steps, step)]
        plt.gca().set_prop_cycle(plt.cycler('color', colors))

        for i, value_z_scan in enumerate(values_z_scan):
            plt.plot(x, mean_y[i], lw=1.5, label=label_z_scan + " %0.1f" % value_z_scan)
            plt.fill_between(x, mean_y_lower[i], mean_y_upper[i], facecolor='grey', alpha=0.2)

    plt.xlim((min(x), max(x)))
    plt.ylim(ymin=0)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    if values_z_scan != None and len(values_z_scan) != 1:
        plt.legend()

    print("--------------------------------------------------")
    for saving_path in saving_paths:
        print("saving figure to %s/%s.pdf ..." % (saving_path, figure_name))
        os.system('mkdir -p %s' % (saving_path))
        plt.savefig("%s/%s.pdf" % (saving_path, figure_name))


###############################################################################
def plot_prediction_performance(x, data, s, saving_paths, figure_name, lw_s=2., comp=None, xmax=None):
    """ Plot prediction performance versus number of training episodes.
    Prediction performance includes error, false positives, and false negatives

    Parameters
    ----------
    x     : array
        contains x-axis values (episodes)
    data  : dict
        dictionary containing errors, false negatives, and false positives
    s   : float
        sparsity level
    saving_paths : list of string
        list containing paths to where to store the figure
    figure_name  : string
    """

    color_ohtm = 'grey'
    color_shtm = 'black'
    std_color = '#e8e8e8'  # "#dcdcdc"  
    std_color_ohtm = '#e8e8e8' #"#dcdcdc"
    color_sparsity = 'grey'
    lw = 1.5
    lw_s = lw_s
    N = 4

    # plot prediction error    
    # ----------------------- 
    plt.figure(figsize=(5.2, 3), constrained_layout=True)

    y = mean_confidence_interval(data['error'])
    x, y = running_mean(x, y, N)
    mean_y, mean_y_lower, mean_y_upper = y

    plt.plot(x, mean_y, lw=lw, color=color_shtm, zorder=2, label="Spiking model")
    plt.fill_between(x, mean_y_lower, mean_y_upper, facecolor=std_color)

    if comp:
        y = mean_confidence_interval(comp['error'])
        x_o, y = running_mean(comp['ep_num'], y, N)
        mean_y, mean_y_lower, mean_y_upper = y
        plt.plot(x_o, mean_y, lw=lw, color=color_ohtm, zorder=1, label="Original model")
        plt.fill_between(x_o, mean_y_lower, mean_y_upper, facecolor=std_color_ohtm)

        plt.legend()

    if xmax:
        max_x = xmax
    else:
        max_x = max(x)

    plt.xlim((min(x), max_x))
    plt.xlabel("number of training episodes")
    plt.ylabel("prediction error")
    plt.ylim(ymin=-0.01)

    # combine prediction error and false negative/positive
    # using subplots
    # -----------------------------------------------------
    plt.figure()
    gs = gridspec.GridSpec(1, 3, left=0.08, right=0.98, bottom=0.2, top=0.93, wspace=0.35, hspace=0.01)

    # plot prediction error    
    # -----------------------
    ax = plt.subplot(gs[0, 0])
    panel_label_pos = (-0.23, 1.08)
    panel_label('A', panel_label_pos)

    y = mean_confidence_interval(data['error'])
    _, y = running_mean(x, y, N)
    mean_y, mean_y_lower, mean_y_upper = y

    plt.plot(x, mean_y, lw=lw, color=color_shtm, zorder=2, label="spiking model")
    plt.fill_between(x, mean_y_lower, mean_y_upper, facecolor=std_color)

    if comp:
        yc = mean_confidence_interval(comp['error'])
        _, yc = running_mean(comp['ep_num'], yc, N)
        mean_yc, mean_yc_lower, mean_yc_upper = yc
        plt.plot(x_o, mean_yc, lw=lw, color=color_ohtm, zorder=1, label="original model")
        plt.fill_between(x_o, mean_yc_lower, mean_yc_upper, facecolor=std_color_ohtm)

        plt.legend()

    # ax.set_xticklabels([])
    plt.xlim((min(x), max_x))
    plt.ylabel("prediction error")
    plt.ylim(ymin=-0.01)
    plt.yticks(np.arange(0, max(mean_y) + 0.2, 0.2))
 
    # plot false positive and negative
    # --------------------------------
    plt.subplot(gs[0, 1])
    panel_label('B', panel_label_pos)

    y_fp = mean_confidence_interval(data['false_positive'])
    _,y_fp=running_mean(x, y_fp, N)
    mean_y_fp, mean_y_fp_lower, mean_y_fp_upper = y_fp

    y_fn = mean_confidence_interval(data['false_negative'])
    _,y_fn=running_mean(x, y_fn, N)
    mean_y_fn, mean_y_fn_lower, mean_y_fn_upper = y_fn

    p1 = plt.plot(x, mean_y_fn, lw=lw, color=color_shtm, linestyle='dashed', zorder=2, label='false negative')
    p2 = plt.plot(x, mean_y_fp, lw=lw, color=color_shtm, linestyle='solid', zorder=2, label='false positive')

    plt.fill_between(x, mean_y_fn_lower, mean_y_fn_upper, facecolor=std_color)
    plt.fill_between(x, mean_y_fp_lower, mean_y_fp_upper, facecolor=std_color)

    plt.legend()

    if comp:
        y_fp = mean_confidence_interval(comp['false_positive'])
        _, y_fp = running_mean(comp['ep_num'], y_fp, N)
        mean_y_fp, mean_y_fp_lower, mean_y_fp_upper = y_fp

        y_fn = mean_confidence_interval(comp['false_negative'])
        _, y_fn = running_mean(comp['ep_num'], y_fn, N)
        mean_y_fn, mean_y_fn_lower, mean_y_fn_upper = y_fn

        plt.plot(x_o, mean_y_fn, lw=lw, color=color_ohtm, linestyle='dashed', zorder=1)
        plt.plot(x_o, mean_y_fp, lw=lw, color=color_ohtm, linestyle='solid', zorder=1)

        plt.fill_between(x_o, mean_y_fn_lower, mean_y_fn_upper, facecolor=std_color_ohtm)
        plt.fill_between(x_o, mean_y_fp_lower, mean_y_fp_upper, facecolor=std_color_ohtm)

    plt.xlim((min(x), max_x))
    plt.xlabel("number of training episodes")
    plt.ylabel("rel. frequency")
    plt.yticks(np.arange(0, max(mean_y) + 0.2, 0.2))
    plt.ylim(ymin=-0.01)

    plt.legend()

    # plot number of activate neurons
    # --------------------------------
    plt.subplot(gs[0, 2])
    panel_label('C', panel_label_pos)

    y = mean_confidence_interval(data['rel_active_neurons'])
    _,y=running_mean(x, y, N)
    mean_y, mean_y_lower, mean_y_upper = y

    p1 = plt.plot(x, mean_y, lw=lw, color=color_shtm, linestyle='solid')
    plt.fill_between(x, mean_y_lower, mean_y_upper, facecolor=std_color)

    # add target sparsity level
    plt.hlines(s, x[0], x[-1], color=color_sparsity, ls='dotted', lw=lw_s)

    if comp:
        y = mean_confidence_interval(comp['rel_active_neurons'])
        mean_y, mean_y_lower, mean_y_upper = y

        plt.plot(comp['ep_num'], mean_y, lw=lw, color=color_ohtm, linestyle='solid')
        plt.fill_between(comp['ep_num'], mean_y_lower, mean_y_upper, facecolor=std_color_ohtm)

    plt.xlim((min(x), max_x))
    plt.ylabel("rel. no. of active neurons")
    plt.ylim(ymin=-0.01)

    print("--------------------------------------------------")
    for saving_path in saving_paths:
        os.system('mkdir -p %s' % (saving_path))
        print("saving figure to %s/%s.pdf ..." % (saving_path, figure_name))
        plt.savefig("%s/%s.pdf" % (saving_path, figure_name))
        print("saving figure to %s/%s.eps ..." % (saving_path, figure_name))
        plt.savefig("%s/%s.eps" % (saving_path, figure_name))


###############################################################################
def plot_stimulus_timing_analysis(x, data, s, saving_paths, figure_name):
    """ Plot prediction performance versus interstimulus intervals. The prediction performance includes error,
    false negative and false positive

    Parameters
    ----------
    x     : array
        contains x-axis values
    data  : dict
    s     : sparsity level
        float
        dictionary containing errors, false negatives, and false positives
    saving_paths : list of string
        list containing paths to where to store the figure
    figure_name  : string
    """

    color_shtm = 'black'
    std_color = '#e8e8e8'
    color_lrs = 'brown'
    std_color_lrs = '#eeb9b9ff' #'#964b004d'
    color_sparsity = 'grey'
    lw = 1.5
    lw_s = 1.5
    lw_fn = 2.5
    N = 1

    # plot prediction error    
    # -----------------------
    plt.figure(figsize=(5.2, 3), constrained_layout=True)

    y = mean_confidence_interval(data['error'])
    mean_y, mean_y_lower, mean_y_upper = y

    plt.plot(x, mean_y, lw=lw, color=color_shtm)
    plt.fill_between(x, mean_y_lower, mean_y_upper, facecolor=std_color)

    plt.ylabel("prediction error")
    plt.xlabel("interstimulus interval $\Delta T$ (ms)")

    plt.xlim((min(x), max(x)))
    plt.ylim(ymin=-0.1)

    # combine prediction error and false negative/positive
    # using subplots
    # -----------------------------------------------------
    plt.figure()
    gs = gridspec.GridSpec(1, 4, width_ratios=[15,0.01,15,15], left=0.08, right=0.98, bottom=0.2, top=0.93, wspace=0.5, hspace=0.01)

    ax = plt.subplot(gs[0, 0])

    panel_label_pos = (-0.25, 1.08)
    panel_label('A', panel_label_pos)

    y = mean_confidence_interval(data['error'])
    mean_y, mean_y_lower, mean_y_upper = y

    yts = mean_confidence_interval(data['time_to_solution'])
    mean_yts, mean_yts_lower, mean_yts_upper = yts

    plt.plot(x, mean_y, lw=lw, color=color_shtm)
    plt.fill_between(x, mean_y_lower, mean_y_upper, facecolor=std_color)

    plt.ylabel("prediction error")
    plt.xlabel("interstimulus interval $\Delta T$ (ms)")
    plt.ylim(ymin=-0.01)
    plt.yticks(np.arange(0, max(mean_y) + 0.2, 0.2))
    plt.xticks(np.arange(15.,max(x),15.))

    # plot time to solution in the second axis
    ax2 = ax.twinx()
    
    plt.plot(x, mean_yts, lw=lw, color=color_lrs)
    plt.fill_between(x, mean_yts_lower, mean_yts_upper, facecolor=std_color_lrs)

    plt.ylabel("episodes-to-solution", color="brown")
    plt.ylim(ymin=-0.01)

    ax2.tick_params(axis='y', labelcolor='brown')
    plt.xlim((min(x), max(x)))

    # plot false positive and negative
    # --------------------------------
    plt.subplot(gs[0, 2])
    panel_label('B', panel_label_pos)

    y_fp = mean_confidence_interval(data['false_positive'])
    mean_y_fp, mean_y_fp_lower, mean_y_fp_upper = y_fp

    y_fn = mean_confidence_interval(data['false_negative'])
    mean_y_fn, mean_y_fn_lower, mean_y_fn_upper = y_fn

    plt.plot(x, mean_y_fn, lw=lw_fn, color=color_shtm, linestyle='dashed', label='false negative')
    plt.plot(x, mean_y_fp, lw=lw, color=color_shtm, linestyle='solid', label='false positive')
    plt.fill_between(x, mean_y_fn_lower, mean_y_fn_upper, facecolor=std_color)
    plt.fill_between(x, mean_y_fp_lower, mean_y_fp_upper, facecolor=std_color)

    plt.xlim((min(x), max(x)))
    plt.xlabel("interstimulus interval $\Delta T$ (ms)")
    plt.ylabel("rel. frequency")
    plt.ylim(ymin=-0.01)
    plt.xticks(np.arange(15.,max(x),15.))

    ax = plt.gca()
    leg = ax.legend(loc='upper center')

    for line in leg.get_lines():
        line.set_linewidth(lw)

    # plot relative number of activate neurons
    # ----------------------------------------
    plt.subplot(gs[0, 3])
    panel_label('C', panel_label_pos)

    y = mean_confidence_interval(data['rel_active_neurons'])
    x, y = running_mean(x, y, N)
    mean_y, mean_y_lower, mean_y_upper = y

    mean_y[0] = mean_y[1]

    p1 = plt.plot(x, mean_y, lw=lw, color=color_shtm, linestyle='solid')
    plt.fill_between(x, mean_y_lower, mean_y_upper, facecolor=std_color)

    # add target sparsity level
    plt.hlines(s, x[0], x[-1], color=color_sparsity, ls='dotted', lw=lw_s)

    plt.xlim((min(x), max(x)))
    plt.xlabel("interstimulus interval $\Delta T$ (ms)")
    plt.ylabel("rel. no. of active neurons")
    plt.ylim(ymin=-0.01)
    plt.xticks(np.arange(15.,max(x),15.))

    print('------------------------------------------------------------------')
    for saving_path in saving_paths:
        os.system('mkdir -p %s' % (saving_path))
        print("saving figure to " + saving_path + "/" + figure_name + '.pdf')
        plt.savefig(saving_path + "/" + figure_name + '.pdf')
        print("saving figure to " + saving_path + "/" + figure_name + '.eps')
        plt.savefig(saving_path + "/" + figure_name + '.eps')


###############################################################################
def plot_learning_speed(x, data, saving_paths, hs, figure_name):
    """ Plot time to solution versus learning rate

    Parameters
    ----------
    x     : array
            contains x-axis values
    data  : dict
            dictionary containing errors, false negatives, and false positives
    saving_paths : list of string
                   list containing paths to where to store the figure
    figure_name  : string
    """

    gs = gridspec.GridSpec(2, 1, bottom=0.1, right=0.95, top=0.93, wspace=0., hspace=0.1)

    # plot prediction error    
    # -----------------------
    ax = plt.subplot(gs[0, 0])

    # panel_label_pos = (-0.15, 1)
    # panel_label('A', panel_label_pos)

    y = mean_confidence_interval(data['error'])
    mean_y, mean_y_lower, mean_y_upper = y

    plt.plot(x, mean_y, lw=2, color='black')
    plt.fill_between(x, mean_y_lower, mean_y_upper, facecolor='grey', alpha=0.2)

    plt.title("homeostasis factor %0.3f" % hs)
    plt.ylabel("prediction error")
    ax.set_xticklabels([])

    plt.xlim((min(x), max(x)))
    plt.ylim(ymin=-0.1)

    # plot prediction error    
    # -----------------------
    plt.subplot(gs[1, 0])

    # panel_label_pos = (-0.15, 1)
    # panel_label('B', panel_label_pos)

    y = mean_confidence_interval(data['times_to_solution'])
    mean_y, mean_y_lower, mean_y_upper = y

    plt.plot(x, mean_y, lw=2, color='black')
    plt.fill_between(x, mean_y_lower, mean_y_upper, facecolor='grey', alpha=0.2)

    plt.xlabel("learning rate ($\lambda$)")
    plt.ylabel("time to solution")

    # plt.xlim((min(x), max(x)))
    plt.ylim(ymin=-0.1)

    print('------------------------------------------------------------------')
    for path in saving_paths:
        print("saving figure to %s/%s_hs%0.3f.pdf" % (path, figure_name, hs))
        plt.savefig("%s/%s_hs%0.3f.pdf" % (path, figure_name, hs))


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
        #home = '..'
        #home = Path.home()
        username = getpass.getuser()
        home = f'/work/users/{username}'

    data_path = Path(home, pars["data_root_path"],
                     pars["project_name"],
                     pars['parameterspace_label'],
                     ps_label, add_to_path)

    return data_path


###############################################################################
def matrix_prediction_performance(parameter_key_list, data_address, sequences=[], add_to_path='', fname=''):
    """For each spiking data s_j defined by parameter space in data_address, 
    compute the precition error and time to solution and store these as follow:
    data_i[j_p1,j_p2,...,j_pm]=error_j  
    data_i[j_p1,j_p2,...,j_pm]=time_to_solution  
    j_p1,j_p2,...,j_pm are positions associated with parameters in parameter_key_list
    
    For nested parameters, use "." notation in key lists: 
    to adress parameter p['a']['b'], use key 'a.b'.
    
    Parameters
    ----------
    parameter_key_list : list [pk1,pk2,...,pkm] of parameter keys 
    data_address       : dict
                         dictionary specifying data and parameter location:                                             
                         data_address['data_root_path']
                         data_address['parameterspace_label']

    Returns
    -------
    data_dict: dict
               dictionary containing data matrices data_1,...,data_n

    """

    print("\t gathering data...")

    # analysis_pars=get_analysis_parameters(data_address)
    P, PS_data_path = get_parameter_set(data_address)

    # parameter list
    PL = parameter_set_list(P)

    # get parameter arrays
    parameters = {}
    for pk in parameter_key_list:
        parameters[pk] = get_parameter_array(P, pk)

    # load first data set to obtain data types and dimensions
    params = PL[0]

    # create data matrices 
    data = {}
    data["error"] = np.zeros([len(parameters[pk]) for pk in parameter_key_list] + [1], dtype=np.float64)
    data["time_to_solution"] = np.zeros([len(parameters[pk]) for pk in parameter_key_list] + [1], dtype=np.float64)
    # data["overlap"] = np.zeros([len(parameters[pk]) for pk in parameter_key_list] + [1], dtype=np.float64)

    for cp, params in enumerate(PL):

        data_path = get_data_path(params['data_path'], params['label'], add_to_path)

        print("\t\t data set %d/%d: %s/%s" % (cp + 1, len(PL), data_path, fname))
        #print("\n lambda_plus: %0.3f, lambda_h: %0.3f, seed: %d" % (params["syn_dict_ee"]["lambda_plus"], params["syn_dict_ee"]["lambda_h"], params["seed"]))
        print("\n")

        # construct index vector
        ind = []
        for pk in parameter_key_list:
            ind += [get_index(parameters[pk], getByDotNotation(params, pk))]

        # load spikes from reference data
        pred = load_data(data_path, 'prediction_performance')

        error = pred['error']
        # overlap = pred['overlap']

        time_to_solution = np.where(error < 0.001)[0]
        try:
            initial_time_to_solution = time_to_solution[0] * params['episodes_to_testing']
        except:
            initial_time_to_solution = params['learning_episodes']

        data['error'][tuple(ind)] = error[-1]
        data['time_to_solution'][tuple(ind)] = initial_time_to_solution
        # data['overlap'][tuple(ind)] = np.mean(overlap[-1])
    
    return data, P


###############################################################################
def panel_label(s, pos, title='', sz=10):
    """Creates a panel label (A,B,C,...) for the current axis object of a matplotlib figure.

    Parameters
    ----------
    s:   str
        panel label
    pos: tuple
        x-/y- position of panel label (in units relative to the size of the current axis)

    title: str
        additional text describing the figure 
    """

    ax = plt.gca()
    plt.text(pos[0], pos[1], s, transform=ax.transAxes, horizontalalignment='center', verticalalignment='top', size=sz,
             weight='bold')
    # plt.text(pos[0], pos[1], r'\bfseries{}%s %s' % (s, title), transform=ax.transAxes, horizontalalignment='center', verticalalignment='center',
    #         size=10)
    plt.text(pos[0] + 0.1, pos[1], title, transform=ax.transAxes, verticalalignment='top', size=sz)

    return 0


###############################################################################
def create_composite_figure(composite_figure_name_root, master_file_name, ext_file_name, fig_size,
                            pos_ext_figure=(0, 0), final_name='', draw_grid=False):
    """
    Creates a composite figure composed of a master figure and an external figure, both availabe as pdf files. The resulting file is saved in pdf format.

    Parameters
    ----------
    composite_figure_name_root: str
                                File name root of resulting composite figure

    master_file_name:           str
                                File name of master figure

    ext_file_name:              str
                                File name of external figure

    pos_ext_figure:             tuple
                                Position of external figure within the composite figure (the master figure is always centered at position (0,0)).

    draw_grid:                  bool
                                if True: draws some grid lines to assist positioning of external figure
    """

    file = open('%s.tex' % composite_figure_name_root, 'w')
    file.write(r"\documentclass{article}")
    file.write("\n")
    file.write(r"\usepackage{geometry}")
    file.write("\n")
    print(fig_size[0], fig_size[1])
    file.write(r"\geometry{paperwidth=%.3fin, paperheight=%.3fin, top=0pt, bottom=0pt, right=0pt, left=0pt}" % (
        fig_size[0], fig_size[1]))
    file.write("\n")
    file.write(r"\usepackage{tikz}")
    file.write("\n")
    file.write(r"\usepackage{graphicx}")
    file.write("\n")
    file.write(r"\pagestyle{empty}")
    file.write("\n")
    file.write(r"\begin{document}")
    file.write("\n")
    file.write(r"\noindent")
    file.write("\n")
    file.write(r"\resizebox{\paperwidth}{!}{%")
    file.write("\n")
    file.write(r"  \begin{tikzpicture}%")
    file.write("\n")
    file.write(r"    \node[inner sep=-1pt] (matplotlib_figure) at (0,0)")
    file.write("\n")
    file.write(r"    {\includegraphics{%s}};" % (master_file_name))
    file.write("\n")
    file.write(r"    \node[inner sep=-1pt,rectangle] (inkscape_sketch) at (%.4f,%.4f)" % (
        pos_ext_figure[0], pos_ext_figure[1]))
    file.write("\n")
    file.write(r"    {\includegraphics{%s}};" % (ext_file_name))
    file.write("\n")
    if draw_grid:
        file.write(r"    \draw[style=help lines] (-6,-4) grid (6,4);")
        file.write("\n")
    file.write(r"  \end{tikzpicture}%")
    file.write("\n")
    file.write(r"}")
    file.write("\n")
    file.write(r"\end{document}")
    file.write("\n")

    file.close()

    # execute tex script
    os.system('pdflatex %s.tex' % composite_figure_name_root)
    cfnr = composite_figure_name_root
    os.system('rm %s.aux %s.log %s.tex' % (cfnr, cfnr, cfnr))
    os.system('rm %s.pdf' % (master_file_name))

    if final_name != '':
        os.system('mv %s.pdf %s' % (composite_figure_name_root, final_name))

    print('\nsave %s.pdf ...' % (composite_figure_name_root))

    return 0


###############################################################################
def getByDotNotation(obj, ref):
    """Access nested dictionary
    from http://www.velvetcache.org/2012/03/13/addressing-nested-dictionaries-in-pythontems via dot notation.
    """

    val = obj
    for key in ref.split('.'):
        val = val[key]
    return val


###############################################################################
def get_index(y, x):
    """Return index i=argmin_i |y[i]-x|. 
    """

    if type(x) == str:
        ind = np.where(np.array(y) == x)[0][0]
    else:
        if np.isinf(x):
            ind = np.where(np.isinf(y))[0][0]
        else:
            d = np.abs(np.array(y) - x)
            ind = np.where(d == np.min(d))[0][0]
    return ind


###############################################################################
def get_parameter_array(P, item):
    """Converts ParameterRange  or parameter value P[item] to list of parameter values. 
    """

    if type(P[item]) == para.ParameterRange:
        x = P[item]._values
    else:
        x = [P[item]]
    return x


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
 
    import importlib.util
    spec = importlib.util.spec_from_file_location('parameters_space', 
                                                  '%s/%s' % (str(params_path), 'parameters_space.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    print('\t Get parameters from %s/%s' % (str(params_path), 'parameters_space.py'))

    P = mod.p

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
def gather_data(parameter_key_list, data_key_list, data_address, num_timestep=None, add_to_path='', fname=''):
    """For each data set p_j defined by parameter space in data_location_pars, 
    a) load data d1_j,...,dn_j (with dimensions nd1,...,ndn) 
    associated with data keys dk1,...,dkn in data_key_list, and 
    b) write to data arrays data_1,...,data_n at positions j_p1,j_p2,...,j_pm 
    associated with parameters in parameter_key_list, i.e.
    data_i[j_p1,j_p2,...,j_pm,:]=di_j represents data obtained for parameter set p_j.
    
    For nested parameters, use "." notation in key lists: 
    to adress parameter p['a']['b'], use key 'a.b'.
    
    Parameters
    ----------
    parameter_key_list : list [pk1,pk2,...,pkm] of parameter keys 
    data_key_list      : list [dk1,dk2,...,dkn] of data keys
    data_address       : dict
                         dictionary specifying data and parameter location:                                             
                         data_address['data_root_path']
                         data_address['parameterspace_label']

    Returns
    -------
    data_dict: dict
               dictionary containing data matrices data_1,...,data_n

               data_1: array of size np1 x np2 x ... x npm x nd1
                       data corresponding to data key dk2
               data_2: array of size np1 x np2 x ... x np x nd2
                       data corresponding to data key dk2
               ...
               data_n: array of size np1 x np2 x ... x npm x ndn
                       data corresponding to data key dkn
    """

    print("\t gathering data...")

    # analysis_pars=get_analysis_parameters(data_address)
    P, PS_data_path = get_parameter_set(data_address)

    # parameter list
    PL = parameter_set_list(P)

    # get parameter arrays
    parameters = {}
    for pk in parameter_key_list:
        parameters[pk] = get_parameter_array(P, pk)

    # load first data set to obtain data types and dimensions
    params = PL[0]
    data_path = get_data_path(params['data_path'], params['label'], add_to_path)
    dat = load_data(data_path, fname)

    # create data matrices 
    data = {}
    for dk in data_key_list:
        if num_timestep:
            ld = 1
        else:
            try:
                ld = len(dat[dk])
            except:
                ld = 1  # len(dat[dk])

        if dk in dat:
            # data[dk]=np.zeros([len(parameters[pk]) for pk in parameter_key_list] + list(np.array(dat[dk]).shape),dtype=np.float64)
            data[dk] = np.zeros([len(parameters[pk]) for pk in parameter_key_list] + [ld], dtype=np.float64)
        else:
            data[dk] = np.zeros([len(parameters[pk]) for pk in parameter_key_list])
            print("WARNING: Key %s doesn't exist." % (dk))

    for cp, params in enumerate(PL):
        # load data
        data_path = get_data_path(params['data_path'], params['label'], add_to_path)
        dat = load_data(data_path, fname)

        print("\t\t data set %d/%d: %s/%s" % (cp + 1, len(PL), data_path, fname))

        # construct index vector
        ind = []
        for pk in parameter_key_list:
            ind += [get_index(parameters[pk], getByDotNotation(params, pk))]

        for dk in data_key_list:
            if dk in dat:
                try:
                    if num_timestep == None:
                        data[dk][tuple(ind)] = dat[dk]
                    else:
                        data[dk][tuple(ind)] = dat[dk][num_timestep]
                except:
                    print("Warning: dk:%s empty" % dk)
                    print("Warning: indices", ind)
                    data[dk][tuple(ind)] = np.nan
            else:
                data[dk][tuple(ind)] = None

    return data, P
