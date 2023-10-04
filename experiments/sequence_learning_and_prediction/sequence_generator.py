'''
Sequence generator creating a set of high-order (context dependent) sequences 
with parameterizable complexity.

For details, see help(generate_sequences)

---

Tom Tetzlaff (December 2022)
'''

import numpy as np
    
##############################################################################

def generate_disjoint_sequences(S, C, vocabulary, redraw=True):
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
    # TODO
    #assert(len(vocabulary)>=S*C)
    
    reduced_vocabulary = list(np.random.permutation(vocabulary))
    seq_set = []
    for cs in range(S):
        seq, reduced_vocabulary = select_random_elements_from_vocabulary(C, reduced_vocabulary, redraw=redraw)
        seq_set += [seq]
    return seq_set, reduced_vocabulary

##############################################################################

def select_random_elements_from_vocabulary(N, vocabulary, redraw=False):
    '''
    Randomly draws N elements from a given vocabulary (with or without redrawing). 
    If redraw=False, drawn elements are removed from the returned vocabulary.

    Parameters
    -----------
    N: int
       Number charcters to be drawn from the vocabulary.

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

    assert(type(N)==int and N>0)
    assert(type(vocabulary)==list)    
    assert(N<=len(vocabulary))
    
    vocabulary = list(np.random.permutation(vocabulary))
    chars = vocabulary[:N]
    if redraw == False:
        vocabulary=vocabulary[N::]  ## remove N elements from list of available elements (no redrawing)
    return chars, vocabulary

##############################################################################

def generate_sequences(S, C, R, O, vocabulary_size, minimal_prefix_length = 0, minimal_postfix_length = 0, seed = None, redraw = False):    
    '''
    Generates a set of S sequences of length C from a vocabulary of defined size. 

    Each sequence is composed of

    1) a prefix consisting of at least minimal_prefix_length elements,
    2) a subsequence randomly drawn from a set of R non-overlapping subsequences of length O with non-repeating elements,
    3) a postfix consisting of at least minimal_postfix_length elements.

    The subsequences (2) create overlap between sequences (i.e., a context dependence). 
    If redraw=False, the pre- and the postfixes are non-overlapping without recurring elements.

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

    Returns
    --------
    seq_set: list(list(int))
             List of S sequences.

    shared_seq_set: list(list(int))
                    List of shared R subsequences.

    vocabulary: list(int)
                Vocabulary after removal of elements occuring in shared_seq_set. 


    '''

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
# TODO    
#    if redraw==False:
#        vocabulary_size = R*O + S*(C-O)  ## set vocabulary size to the required minimum
#        #assert(vocabulary_size >= R*O + S*(C-O))
#    else:
#        #vocabulary_size = R*O
#        assert(vocabulary_size >= R*O)
        
    if seed!=None:
        np.random.seed(seed)
    
    vocabulary = list(range(vocabulary_size))
    
    if S > 1 and R != 0:
        shared_seq_set, reduced_vocabulary = generate_disjoint_sequences(R, O, vocabulary)
        
        seq_set = []
        for cs in range(S):
            this_shared_seq = shared_seq_set[np.random.randint(R)]  ## pick random shared subsequence
            ## starting position of shared subsequence
            start_position = np.random.randint(low=minimal_prefix_length,high=C-O-minimal_postfix_length+1) 
            prefix, reduced_vocabulary = select_random_elements_from_vocabulary(start_position,reduced_vocabulary,redraw = redraw)
            postfix, reduced_vocabulary = select_random_elements_from_vocabulary(C-(start_position+O),reduced_vocabulary,redraw = redraw)
            seq_set += [prefix + this_shared_seq + postfix]   
    else:
        seq_set = []
        shared_seq_set = []
        reduced_vocabulary = list(np.copy(vocabulary))
        for cs in range(S):
            print(reduced_vocabulary)
            sequence, reduced_vocabulary = select_random_elements_from_vocabulary(C, vocabulary=reduced_vocabulary, redraw=False)
            seq_set += [sequence]   

    return seq_set, shared_seq_set, vocabulary

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

def print_sequences(seq_set,shared_seq_set,vocabulary,label=''):
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

##############################################################################

def example():
    '''
    A simple example.

    '''

    ####################
    ## parameters

    vocabulary_size = 26        ## vocabulary size (may be overwritten if redraw==False)
    S=10                        ## number of sequences
    C=10                         ## sequence length
    R=10                         ## number of shared subsequences
    O=8                         ## length of shared subsequences ("order")
    minimal_prefix_length = 1   ## minimal prefix length
    minimal_postfix_length = 1  ## minimal postfix length
    redraw = True              ## if redraw == True: pre- and postfixes may contain repeating elements 
    seed = 0 #None              ## RNG seed (int or None)
    alphabet = latin_alphabet   ## function defining type of alphabet (only important for printing)
    
    ####################    
    
    seq_set, shared_seq_set, vocabulary = generate_sequences(S, C, R, O, vocabulary_size, minimal_prefix_length, minimal_postfix_length, seed, redraw)

    print_sequences(seq_set,shared_seq_set,vocabulary,label=' (int)')
    
    shared_seq_set_transformed = transform_sequence_set(shared_seq_set, alphabet)    
    seq_set_transformed = transform_sequence_set(seq_set, alphabet)
    vocabulary_transformed = transform_sequence(vocabulary, alphabet)

    print_sequences(seq_set_transformed,shared_seq_set_transformed,vocabulary_transformed,label=' (latin)')

    return seq_set_transformed, vocabulary_transformed

##############################################################################

if __name__ == '__main__':
    seq_set, vocabulary = example()

##############################################################################
