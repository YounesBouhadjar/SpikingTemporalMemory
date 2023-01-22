import numpy as np

from shtm.helper import get_data_path


##########################################
def generate_sequences(task_params, data_path, fname):
    """Generate sequence of elements using three methods:
    1. randomly drawn elements from a vocabulary
    2. sequences with transition matrix
    3. higher order sequences: sequences with shared subsequences
    4. hard coded sequences

    Parameters
    ----------
    task_params : dict
        dictionary contains task parameters
    data_path   : dict
    fname       : str

    Returns
    -------
    sequences: list
    test_sequences: list
    vocabulary: list
    """

    vocab_size=task_params["vocab_size"]
    task_name=task_params["task_name"]
    task_type=task_params["task_type"]
 
    # set of characters used to build the sequences
    vocabulary = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
                  'U', 'V', 'W', 'X', 'Y', 'Z'][:vocab_size]
    sequences = []

    if task_name == 'hard_coded':

        # hard coded sequences 
        if task_type == 0:
            sequences = [['A', 'B']]
        elif task_type == 1:
            sequences = [['A', 'D', 'B', 'E'], ['F', 'D', 'B', 'C']]
        elif task_type == 2:
            sequences = [['A', 'D', 'D', 'D', 'D', 'C'], ['B', 'D', 'D', 'D', 'D', 'E']]
        else:
            raise "specify task type"

    else:
        raise "specify task name"

    # test sequences used to measure the accuracy 
    test_sequences = sequences

    if task_params['store_training_data']:
        fname = 'training_data'
        fname_voc = 'vocabulary'
        data_path = get_data_path(data_path)
        print("\nSave training data to %s/%s" % (data_path, fname))
        np.save('%s/%s' % (data_path, fname), sequences)
        np.save('%s/%s' % (data_path, fname_voc), vocabulary)

    return sequences, test_sequences, vocabulary
