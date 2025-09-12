#!/bin/bash
#
# This file is part of spikingtemporalmemory.
#
# spikingtemporalmemory is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# spikingtemporalmemory is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with spikingtemporalmemory.  If not, see <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later


import numpy as np
import matplotlib
import matplotlib.pylab as plt

##############################################################################

def generate_disjoint_sequences(S,C,vocabulary,redraw=False):
    '''
    Generates a set of S non-overlapping sequences of length C with non-repeating elements from a given vocabulary. 
    In addition, the function returns a reduced vocabulary, i.e., the list of remaining elements 
    that have not been used for the sequence set.
    
    Parameters
    -----------
    S: int
       Number of sequences.

    C: int
       Length of sequences (all sequences have identical length).

    vocabulary: list(int)
                List of integer numbers constituting the vocabulary.

    Returns
    --------
    seq_set: list(list(int))
             List of S sequences.

    reduced_vocabulary: list(int)
                        Reduced vocabulary after removal of all elements occuring in seq_set, with randomized order.

    '''

    assert(type(S)==int and S>0)
    assert(type(C)==int and C>0)
    assert(type(vocabulary)==list)    
    if redraw == False:
        assert(len(vocabulary)>=S*C)
    
    reduced_vocabulary = list(np.random.permutation(vocabulary))
    seq_set = []
    for cs in range(S):
        seq, reduced_vocabulary = select_random_elements_from_vocabulary(C,reduced_vocabulary,redraw=redraw)
        seq_set += [seq]
    return seq_set,reduced_vocabulary

##############################################################################

def select_random_elements_from_vocabulary(N, vocabulary, redraw=False):
    '''
    Randomly draws N elements from a given vocabulary (with or without redrawing). 
    If redraw=False, drawn elements are removed from the returned vocabulary.

    Parameters
    -----------
    N: int
       Number charcters to be drawn form the vocabulary.

    vocabulary: list(int)
                Vocabulary.

    redraw: bool (default: False)
            If False, each element can be drawn at most once, and will be removed from the vocabulary.

    Returns
    --------
    chars: list(int)
           List of elements randomly chosen from vocabulary.

    vocabulary: list(int)
                List of elements constituting the vocabulary after randomization.
                If redraw=False, all elements occuring in chars are removed.
    '''

    assert(type(N)==int and N>=0)
    assert(type(vocabulary)==list)    

    if redraw:
        chars = np.random.choice(vocabulary,size=N,replace=redraw).tolist()
    
    else:
        assert(N<=len(vocabulary))
        
        #vocabulary = list(np.random.permutation(vocabulary))
        vocabulary = np.random.permutation(vocabulary).tolist()        
        chars = vocabulary[:N]
        vocabulary=vocabulary[N::]  ## remove N elements from list of available elements (no redrawing)

    return chars, vocabulary

##############################################################################
def generate_multiple_sequences(S, C, R, O, vocabulary_size, minimal_prefix_length = 0, minimal_postfix_length = 0, seeds = [None], redraw = False, inter_elem_intv_min = 50., inter_elem_intv_max = 50.):    
    seq_sets = []
    seq_sets_intervals = []
    for seed in seeds:
        seq_set, shared_seq_set, vocabulary, seq_set_intervals = generate_sequences(S, C, R, O,
                                                                           vocabulary_size,
                                                                           minimal_prefix_length,
                                                                           minimal_postfix_length,
                                                                           seed,
                                                                           redraw,
                                                                           inter_elem_intv_min,
                                                                           inter_elem_intv_max)

        
        shared_seq_set_transformed = transform_sequence_set(shared_seq_set, latin_alphabet)    
        seq_set_transformed = transform_sequence_set(seq_set, latin_alphabet)
        vocabulary_transformed = transform_sequence(vocabulary, latin_alphabet)

        print_sequences(seq_set_transformed,
                   shared_seq_set_transformed,
                   vocabulary_transformed,
                   seq_set_intervals,
                   label=' (latin)')

        seq_sets += seq_set
        seq_sets_intervals += seq_set_intervals

        import pdb
        pdb.set_trace()

    return seq_sets, vocabulary, seq_sets_intervals

##############################################################################
def generate_sequences(S, C, R, O, vocabulary_size, minimal_prefix_length = 0, minimal_postfix_length = 0, seed = None, redraw = False, inter_elem_intv_min = 50., inter_elem_intv_max = 50.):    
    '''
    Generates a set of S sequences of length C from a vocabulary of defined size. 

    Each sequence is composed of

    1) a prefix consisting of at least minimal_prefix_length elements,
    2) a subsequence randomly drawn from a set of R non-overlapping subsequences of length O with non-repeating elements,
    3) a postfix consisting of at least minimal_postfix_length elements.

    The subsequences (2) create overlap between sequences (i.e., a context dependence). 
    If redraw=False, the pre- and the postfixes are non-overlapping without recurring elements.

    Time intervals between sequence elements are drawn randomly and independently from a uniform distribution.

    Parameters
    -----------
    S: int
       Number of sequences.

    C: int
       Length of sequences (all sequences have identical length).

    R: int
       Number of shared subsequences.

    O: int
       Length of shared subsequences ("order").

    vocabulary_size: int
                     Number of elements in vocabulary.

    minimal_prefix_length: int (default minimal_prefix_length=0)
                           Minimal length of the sequence prefix.

    minimal_postfix_length: int (default minimal_postfix_length=0)
                            Minimal length of the sequence postfix.

    seed: int or None (default seed = None)
          RNG seed. If None, the RNG seed is not set.

    redraw: bool (default  redraw = False)
            If False, the pre- and the postfixes are non-overlapping without recurring elements.

    inter_elem_intv_min: float
                         Minimum inter-element interval (ms; default: 50.).

    inter_elem_intv_max: float
                         Maximum inter-element interval (ms; default: 50.).

    Returns
    --------
    seq_set: list(list(int))
             List of S sequences.

    shared_seq_set: list(list(int))
                    List of shared R subsequences.

    vocabulary: list(int)
                Vocabulary after removal of elements occuring in shared_seq_set. 

    seq_set_intervals: list(list(float))
                       Inter-element intervals for each sequence.

    '''

    redraw_overlaps = True
    redraw_postfix = True
    redraw_prefix = False
    if S==0 or C==0:
        seq_set = []
        shared_seq_set = []
        vocabulary = []
        seq_set_intervals = []
        
    else:
        
        assert(type(S)==int)
        assert(type(C)==int)
        assert(type(R)==int)
        assert(type(O)==int)
        assert(type(minimal_prefix_length)==int)
        assert(type(minimal_postfix_length)==int)
        #assert(type(seed)==int or type(seed)==NoneType)
        assert(type(redraw)==bool)

        assert(C>=minimal_prefix_length+minimal_postfix_length+O)
        ## sequence length needs to be at least minimal_prefix_length+minimal_postfix_length+O

        if seed!=None:
            np.random.seed(seed)

        seq_set = []
        seq_set_intervals = []    

        if R==0 or O==0:  ## no shared subsequences
            
            vocabulary = list(range(vocabulary_size))
                    
            for cs in range(S):            
                seq, reduced_vocabulary = select_random_elements_from_vocabulary(C,
                                                                                 vocabulary,
                                                                                 redraw=True)
                seq_set += [seq]
                seq_set_intervals += [np.random.uniform(low=inter_elem_intv_min,
                                                        high=inter_elem_intv_max,
                                                        size=C-1)]                
            shared_seq_set = []
            
        else:
            
#            if redraw_prefix==False and redraw_postfix==False:
#                vocabulary_size_old = vocabulary_size
#                vocabulary_size = R*O + S*(C-O)  ## set vocabulary size to the required minimum
#
#                if vocabulary_size_old != vocabulary_size:
#                    print("\n##################################################################")
#                    print("WARNING: vocabulary size is changed to %d." % vocabulary_size)
#                    print("##################################################################\n")
#
#                #assert(vocabulary_size >= R*O + S*(C-O))
#            elif redraw_overlaps == True:
#                #vocabulary_size = R*O
#                assert(vocabulary_size >= R*O)
            if redraw_prefix == False:
                assert(vocabulary_size >= R)

            vocabulary = list(range(vocabulary_size))

            shared_seq_set, reduced_vocabulary = generate_disjoint_sequences(R, O,
                                                                             vocabulary,
                                                                             redraw_overlaps)    

            for cs in range(S):
                this_shared_seq = shared_seq_set[np.random.randint(R)]  ## pick random shared subsequence
                ## starting position of shared subsequence
                start_position = 1#np.random.randint(low=minimal_prefix_length,
                                 #                  high=C-O-minimal_postfix_length+1) 

                prefix, reduced_vocabulary = select_random_elements_from_vocabulary(start_position,
                                                                                    reduced_vocabulary,
                                                                                    redraw=redraw_prefix)
                postfix, reduced_vocabulary = select_random_elements_from_vocabulary(C-(start_position+O),
                                                                                     reduced_vocabulary,
                                                                                     redraw=redraw_postfix)
                seq_set += [prefix + this_shared_seq + postfix]
                seq_set_intervals += [np.random.uniform(low=inter_elem_intv_min,
                                                        high=inter_elem_intv_max,size=C-1)]
        
    return seq_set, shared_seq_set, vocabulary, seq_set_intervals

##############################################################################

def latin_alphabet():
    '''
    Generates latin alphabet.

    Parameters
    -----------
    -
                    
    Returns
    --------
    A: list(str)
       List of characters from the latin alphabet.

    '''
    
    uppercase = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
    lowercase = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']    
    A = uppercase + lowercase
    return A

##############################################################################

def transform_sequence(seq, alphabet):
    '''
    Transforms sequence of integer elements to a sequence with elements specified by alphabet.

    Parameters
    -----------
    seq: list(int)
             Sequence of integer elements.

    alphabet: function
              Function returning list of characters representing some alphabet (example: alphabet=latin_alphabet).

    Returns
    --------
    seq_transformed: list(str)
                     Sequence of characters corresponding to alphabet.

    '''

    assert(type(seq)==list)
    #assert(type(alphabet)==function)
    
    A = alphabet()

    assert(max(seq)<=len(A))
    ## alphabet too small
   
    seq_transformed = []
    for s in seq:
        seq_transformed += [A[s]]
    return seq_transformed

##############################################################################

def transform_sequence_set(seq_set, alphabet):
    '''

    Parameters
    -----------
    seq_set: list(list(int))
             Set of sequences of integer elements.

    alphabet: function
              Function returning list of characters representing some alphabet (example: alphabet=latin_alphabet).

    Returns
    --------
    seq_set_transformed: list(list(str))
                         Set of sequence of characters corresponding to alphabet.

    '''

    assert(type(seq_set)==list)
    #assert(type(alphabet)==function)
    
    seq_set_transformed = []
    for seq in seq_set:
        seq_set_transformed += [transform_sequence(seq, alphabet)]
    return seq_set_transformed

##############################################################################

def print_sequences(seq_set,shared_seq_set,vocabulary,seq_set_intervals,label=''):
    '''
    Prints sequence set, shared sequence set and vocabulary to screen.

    Parameters
    -----------
    seq_set,shared_seq_set,vocabulary,label=''

    Returns
    --------
    -

    '''
    print('\n######################################\n')
    print('Sequences%s:' % label)
    print('--------------------------------------')
    for cs,s in enumerate(seq_set):
        print("%d:\t%s" % (cs+1,s))
    print()
    print('Shared sequences%s:' % label)
    print('--------------------------------------')    
    for css,ss in enumerate(shared_seq_set):
        print("%d:\t%s" % (css+1,ss))
    print()
    print('Vocabulary%s:' % label)
    print('--------------------------------------')    
    print("%s (size=%d)" % (vocabulary,len(vocabulary)))
    print()
    print('Intervals (ms)')
    print('--------------------------------------')
    for cs,sintvs in enumerate(seq_set_intervals):
        print("%d:" % (cs+1), end="")
        for intv in sintvs:
            print("\t%.1f" % intv, end="")
        print()
    print()

##############################################################################

def generate_multiple_sequence_set_instance(
        seq_set,
        seq_set_intervals,        
        start,
        stop,
        seq_set_instance_size = None,
        subset_size           = None,
        order                 = 'fixed',
        seq_activation_type   = 'consecutive',
        inter_seq_intv_min    = 100.,
        inter_seq_intv_max    = 100.,
):

    for i in range(2):
        seq_set_instance, seq_ids = generate_sequence_set_instance(
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
            start_id              = i * seq_set_instance_size
        )

        if i == 0:
            seq_set_instances = seq_set_instance
        else:
            seq_set_instances = seq_set_instances | seq_set_instance

    return seq_set_instances


def generate_sequence_set_instance(
        seq_set,
        seq_set_intervals,        
        start,
        stop,
        seq_set_instance_size = None,
        subset_size           = None,
        order                 = 'fixed',
        seq_activation_type   = 'consecutive',
        inter_seq_intv_min    = 100.,
        inter_seq_intv_max    = 100.,
        start_id              = 0
):
    
    '''
    Generates an instance of a sequence ensemble 
    - composed of (repetitions of all or some) sequences defined in seq_set,
    - specifying the occurence times of sequence elements.

    Parameters
    -----------

    seq_set:               list(list(int))
                           List of sequences.

    seq_set_intervals:     list(list(float))
                           Inter-element intervals for each sequence.

    start:                 float
                           Start time of sequence set instance (ms).

    stop:                  float
                           Stop time of sequence set instance (ms)

    seq_set_instance_size: int or None
                           Total number of sequences in sequence set instance.
                           If None (default), seq_set_instance_size is set to len(seq_set).
                           If seq_set_instance_size>subset_size, individual sequences
                           are repeated. 

    subset_size:           int or None
                           Number of sequences to choose from seq_set.
                           If None (default), subset_size is set to len(seq_set).
              
    order:                 str
                           Type of sequence ordering (default: 'fixed'):

                              - 'fixed': Order of chosen sequences is fixed 
                                         (identical to the order given in seq_set).

                              - 'random': Order of chosen sequences is random.
    
    seq_activation_type:   str
                           Type of sequence activation (default: 'consecutive').

                              - 'consecutive': Consecutive activation of sequences. 
                                               Intervals drawn from uniform distribution in [inter_seq_intv_min,inter_seq_intv_max].

                              - 'parallel': Parallel activation of sequences at random times.

    inter_seq_intv_min:    float
                           Minimum inter-sequence interval (ms; default: 100.).

    inter_seq_intv_max:    float
                           Maximum inter-sequence interval (ms; default: 100.).

#    inter_elem_intv_min:   float
#                           Minimum inter-element interval (ms; default: 50.).

#    inter_elem_intv_max:   float
#                           Maximum inter-element interval (ms; default: 50.).

    Returns
    --------
    seq_set_instance: dict(dict)
                      Sequence ID `seq`, `elements` and corresponding occurrence `times` (ms) 
                      for each sequence in the sequence set instance.

    seq_ids: list(int)
             List of sequence IDs in the sequence set instance.

    '''

    if len(seq_set) == 0:
        seq_set_instance = {}
        seq_ids = []
        
    else:

        if seq_set_instance_size == None:
            seq_set_instance_size  = len(seq_set)

        if subset_size == None:
            subset_size = len(seq_set)

        assert(subset_size<=len(seq_set))


        ## restrict sequence set instance to subset of sequences
        seq_set = seq_set[:subset_size]
        S = len(seq_set)

        ## generate relative element occurence times
        seq_set_times = []    
        for cs in range(len(seq_set)):
            elem_intvs = seq_set_intervals[cs]
            seq_set_times += [[0] + list(np.cumsum(elem_intvs))] ## element occurence times for current sequence        
            #print(seq_set_times[cs])

        ## create list of sequence ids
        if order == 'fixed':
            seq_ids = int(np.ceil(seq_set_instance_size/S)) * list(range(S))
            seq_ids = seq_ids[:seq_set_instance_size]

        elif order == 'random':        
            seq_ids = list(np.random.choice(list(range(S)), size = seq_set_instance_size))

        elif order == 'consecutive':
            block = np.repeat(np.arange(S), int(seq_set_instance_size/ (S+1) ))
            reps = -(-seq_set_instance_size // len(block))
            seq_ids = np.tile(block, reps)[:seq_set_instance_size]

        #print(seq_ids)
        assert(len(seq_ids)==seq_set_instance_size)

        ## create instance of list of sequences
        seq_set_instance = {}
        t = start
        to_be_removed = []
        for cs, cis in enumerate(range(start_id, seq_set_instance_size+start_id)):
            seq_set_instance[cis] = {}
        
            ## sequence elements
            seq_set_instance[cis]['elements'] = seq_set[seq_ids[cs]]
            seq_set_instance[cis]['seq'] = seq_ids[cs]
            
            ## sequence times
            seq_set_instance[cis]['times'] = list(t + np.array(seq_set_times[seq_ids[cs]])) ## element occurence times for current sequence

            if seq_activation_type == 'parallel':
                t =  np.random.uniform(low  = start, high = stop)

            elif seq_activation_type == 'consecutive':            
                t = seq_set_instance[cis]['times'][-1] + \
                    np.random.uniform(low  = inter_seq_intv_min, high = inter_seq_intv_max) ## start of next sequence

            ## truncate sequence set instance at stop time
            elements = np.array(seq_set_instance[cis]['elements'])
            times = np.array(seq_set_instance[cis]['times'])        
            ind = np.where(times>stop)[0]
            if len(ind) > 0:
                print("\nWARNING: sequence %d truncated to fit stop time." % (cs))
                times=np.delete(times,ind)
                elements=np.delete(elements,ind)
                seq_set_instance[cis]['elements']=list(elements)
                seq_set_instance[cis]['times']=list(times)            

                ## remove empty sequences
                if len(times)==0:
                    print("WARNING: Empty sequence %d removed.\n" % (cs))                
                    seq_set_instance.pop(cs)  ## note: this is save because 'cs' is a key, rather than an index.

        print('\n######################################\n')        
        print("Number of sequences in sequence set instance: %d" % (len(seq_set_instance)))
 
    return seq_set_instance, seq_ids

##############################################################################
def seq_set_instance_perturbation(seq_set_instance, pert_type='none', pert_prob=0.0, print_results=False):
    '''
    Perturb sequence set according to specified rules.

    Parameters
    -----------

    pert_type: str
               Type of sequence perturbation.

               'none':           No perturbation (default).
               'cue':            Remove all elements in each sequene, except the first (the "cue").
               'random_replace': In each sequence, replace each element (except the first) 
                                 by a randomly chosen element (from the learned vocabulary) 
                                 with probability pert_prob.

    pert_prob: float
               Probability of applying the perturbation (only for pert_type='random_replace'). Default: 0.0.

    print_results:     bool
                       Print results of perturbation to screen if True (default: False).

    Returns
    --------

    seq_set_instance: dict(dict)
                      Sequence set instance after perturbation (see generate_sequence_set_instance()).

    added_elements:   list(list)
                      ID and time of added elements.

    removed_elements: list(list)
                      ID and time of removed elements.

    '''

    vocabulary = []
    for k in range(len(seq_set_instance)):
        vocabulary += seq_set_instance[k]['elements']
    vocabulary = np.unique(np.array(vocabulary))

    added_elements = []
    removed_elements = []    
    for cs in range(len(seq_set_instance)):

        ## remove all but first element (for testing recall)
        if pert_type == 'cue':

            for i in range(1,len(seq_set_instance[cs]['elements'])):
                removed_elements += [[seq_set_instance[cs]['elements'][i],seq_set_instance[cs]['times'][i]]]
            
            seq_set_instance[cs]['elements']=[seq_set_instance[cs]['elements'][0]]
            seq_set_instance[cs]['times']=[seq_set_instance[cs]['times'][0]]

        ## randomly replace sequence elements with given probability (but not the first element)
        elif pert_type == 'random_replace':
            seq = np.array(seq_set_instance[cs]['elements'])
            ind = np.where(np.random.rand(len(seq)-1)<pert_prob)[0] + 1

            for i in ind:
                removed_elements += [[seq_set_instance[cs]['elements'][i], seq_set_instance[cs]['times'][i]]]
            
            ## remove elements to be perturbed from vocabulary to avoid chance redrawing of the same element
            reduced_vocabulary = np.delete(vocabulary,ind)
            
            seq[ind]=np.random.choice(reduced_vocabulary,size=len(ind))
            seq_set_instance[cs]['elements']=list(seq)

            for i in ind:
                added_elements += [[seq_set_instance[cs]['elements'][i], seq_set_instance[cs]['times'][i]]]

    if print_results:
        print()
        print("Perturbations")
        print("-------------")
        print("added elements: ", added_elements)
        print("removed elements: ", removed_elements)
        print()
            
    return seq_set_instance, added_elements, removed_elements

##############################################################################
def seq_set_instance_gdf(seq_set_instance):
    '''
    Converts sequence set instance to gdf (*) format: (element id, time).

    (*) gdf - Gerstein data format

     Parameters
    -----------
    seq_set_instance: dict(dict)
                      'elements' and 'times' of sequence elements and corresponding occurence time (ms) for each
                      sequence in the seuence set instance.
       
    Returns
    --------
    element_activations: ndarray()
                         Dx2 array containing the D activations of sequence elements in the sequence set instance.
                         element_activations[:,0] = element id
                         element_activations[:,1] = activation time

    '''
    
    element_activations=[]
        
    for cs in range(len(seq_set_instance)):
        for cel,el in enumerate(seq_set_instance[cs]['elements']):
            element_activations += [[el,seq_set_instance[cs]['times'][cel]]]

    element_activations = np.array(element_activations)
    
    return element_activations

##############################################################################
def print_seq_set_instance(seq_set_instance):
    '''
    Prints out sequences (emlements and times) in a sequence set instance to screen.

    Parameters
    -----------

    seq_set_instance: dict(dict)
                      'elements' and 'times' of sequence elements and corresponding occurence time (ms) for each
                      sequence in the seuence set instance.
       
    Returns
    --------

    -

    '''

    print()
    print('######################################\n')     
    print("Sequence set instance:")
    print("----------------------")
    
    for cs in range(len(seq_set_instance)):
        print("\nsequence %d: " % cs)
        seq = seq_set_instance[cs]
        print("  elements:", end="")
        for ce in range(len(seq['elements'])): 
            print("\t%d" % seq['elements'][ce], end="")
        print()
        print("  times (ms):", end="")
        for ce in range(len(seq['elements'])): 
            print("\t%.1f" % seq['times'][ce], end="")
        print()

    return 

##############################################################################
def plot_seq_instance_intervals(seq_set,seq_ids,seq_set_instance,ylim,alpha=0.1,cm='jet'):
    '''
    Plot intervals of sequences in a sequence set.
    '''

    colormap = plt.get_cmap(cm)

    colors = [colormap(k) for k in np.linspace(0., 1., np.max(seq_ids) - np.min(seq_ids) + 1)]

    #colors = [colormap(k) for k in np.linspace(0, 1, np.max([5,len(seq_set)]))]
    #np.random.seed(1)
    #colors = np.random.permutation(colors)
    
    for cs in range(len(seq_set_instance)):
        clr = colors[seq_ids[cs]]
        plt.fill([ seq_set_instance[cs]['times'][0],
                   seq_set_instance[cs]['times'][-1],
                   seq_set_instance[cs]['times'][-1],
                   seq_set_instance[cs]['times'][0]
                  ],[
                      ylim[0],
                      ylim[0],
                      ylim[-1]+2,
                      ylim[-1]+2
                  ],
                 color = clr,alpha=alpha,edgecolor=None,zorder=-10)


##############################################################################

def example():
    '''
    A simple example.

    '''

    ####################
    ## parameters

    vocabulary_size = 26         ## vocabulary size (may be overwritten if redraw==False)
    S=10                          ## number of sequences
    C=10                          ## sequence length
    R=4                          ## number of shared subsequences
    O=4                          ## length of shared subsequences ("order")
    minimal_prefix_length = 1    ## minimal prefix length
    minimal_postfix_length = 1   ## minimal postfix length
    redraw = False               ## if redraw == True: pre- and postfixes may contain repeating elements    
    alphabet = latin_alphabet    ## function defining type of alphabet (only important for printing)
    inter_elem_intv_min   = 10.  ## minimum inter-element interval (ms)
    inter_elem_intv_max   = 100. ## maximum inter-element interval (ms)

    start=100.
    stop=5000.
    seq_set_instance_size = 10
    subset_size           = None
    #order                 = 'fixed'      ## 'fixed', 'random'
    order                 = 'random'      ## 'fixed', 'random'    
    #seq_activation_type   = 'consecutive' ## 'consecutive', 'parallel'
    seq_activation_type   = 'parallel' ## 'consecutive', 'parallel'    
    inter_seq_intv_min    = 100.
    inter_seq_intv_max    = 200.

    seed = 0 #None              ## RNG seed (int or None)

    ####################    
    
    seq_set, shared_seq_set, vocabulary, seq_set_intervals = generate_sequences(S, C, R, O,
                                                                                vocabulary_size,
                                                                                minimal_prefix_length,
                                                                                minimal_postfix_length,
                                                                                seed,
                                                                                redraw)
    print_sequences(seq_set,shared_seq_set,vocabulary,seq_set_intervals,label=' (int)')
    
    shared_seq_set_transformed = transform_sequence_set(shared_seq_set, alphabet)    
    seq_set_transformed = transform_sequence_set(seq_set, alphabet)
    vocabulary_transformed = transform_sequence(vocabulary, alphabet)

    print_sequences(seq_set_transformed,
                    shared_seq_set_transformed,
                    vocabulary_transformed,
                    seq_set_intervals,
                    label=' (latin)')
    
    seq_set_instance, seq_ids = generate_sequence_set_instance(
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

    print_seq_set_instance(seq_set_instance)
        
    ####
    plt.rcParams.update({'font.size': 8})
    plt.figure(1,dpi=300,figsize=(5,3))
    plt.clf()

    ylim = (vocabulary[0],vocabulary[-1])
    plot_seq_instance_intervals(seq_set,seq_ids,seq_set_instance,ylim,alpha=0.1,cm='jet')    

    colormap = plt.get_cmap('jet')
    colors = [colormap(k) for k in np.linspace(0, 1, len(seq_set))]    
    for cs in range(len(seq_set_instance)):
        clr = colors[seq_ids[cs]]
        plt.plot(seq_set_instance[cs]['times'],seq_set_instance[cs]['elements'],'o',ms=3,mfc=clr,mew=0.5,mec='k',rasterized=True)
        plt.text(seq_set_instance[cs]['times'][0],vocabulary[-1]+1,r"%d" % seq_ids[cs],fontsize=5)
    plt.xlabel(r'time (ms)')
    plt.ylabel(r'element ID')
    plt.xlim(0,stop)
    plt.ylim(vocabulary[0]-0.5,vocabulary[-1]+2)

    plt.setp(plt.gca(),yticks = vocabulary)
    
    plt.subplots_adjust(left=0.13,right=0.95,bottom=0.15,top=0.95)
    plt.savefig('example_sequence_set_instance.pdf')
    
    return seq_set, vocabulary, seq_set_instance, seq_ids

##############################################################################

if __name__ == '__main__':

    seq_set, vocabulary, seq_set_instance, seq_ids = example()

    print("seq_set", seq_set)
    print("vocabulary", vocabulary)
    print("seq_set_instance", seq_set_instance)
    print("seq_ids", seq_ids)

    element_activations = seq_set_instance_gdf(seq_set_instance)

    print("element_activations", element_activations)

    exit()

##############################################################################




