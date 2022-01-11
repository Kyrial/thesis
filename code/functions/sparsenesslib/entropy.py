#!/usr/bin/env python
import numpy as np
import itertools

def np_setdiff(A,B):
    """
    Returns the set difference between the np.arrays a and b.
    Both arrays can be in the format
    
    array([[x11, x12, ...],[x21, x22, ...],...)
    
    From: https://stackoverflow.com/questions/8317022/get-intersecting-rows-across-two-2d-numpy-arrays
    
    """
    nrows, ncols = A.shape
    dtype={'names':['f{}'.format(i) for i in range(ncols)],
           'formats':ncols * [A.dtype]}

    C = np.setdiff1d(A.view(dtype), B.view(dtype))
    C = C.view(A.dtype).reshape(-1, ncols)
    return(C)


def get_ordinal_probabilities(data, dx=3, dy=1, tau_x=1, tau_y=1):
    """
    This function obtains the ordinal probabilities from data.
    
    Parameters
    ----------
    data: a np.array in the format [x_1, x_2, x_3, ..., x_n] (1xn - a time series)
          or a matrix [[x_11, x_21, x_31, ..., x_n1], 
                       [x_11, x_21, x_31, ..., x_n1], ..., 
                       [x_1m, x_2m, x_3m, ..., x_nm]] (nxm)
    dx: the embedding dimension in the horizontal axis (Default: 3)
    dy: the embedding dimension in the vertical axis, should me 1 for a time series (Default: 1)
    tau_x: the embedding delay in the horizontal axis (Default: 1)
    tau_y: the embedding delay in the vertical axis (Default: 1)
    ----------
    Returns (occurring states, missing states, ordinal probabilities).
    
    """
    try:
        ny, nx = np.shape(data)
    except:
        nx = np.shape(data)[0]
        ny = 1
        data = np.array([data])
    partitions = np.concatenate(
        [
            [np.concatenate(data[j:j+dy*tau_y:tau_y,i:i+dx*tau_x:tau_x]) for i in range(nx-(dx-1)*tau_x)] 
            for j in range(ny-(dy-1)*tau_y)
        ]
    )
    
    states = np.apply_along_axis(np.argsort, 1, partitions)
    
    unique_states, occurences = np.unique(states, return_counts=True, axis=0)
    probabilities = occurences/len(partitions)
    
    posstible_states = np.asarray(list(map(list,list(itertools.permutations(np.arange(0,dx*dy))))))
    missing_states = np_setdiff(posstible_states,unique_states)
    probabilities = np.concatenate([probabilities,np.zeros(len(missing_states))])
    
    return(unique_states,missing_states,probabilities)


def Shannon_entropy(ordinal_probabilities):
    """
    This function calculates the normalized Shannon entropy from the ordinal_probabilities.
    
    Parameters
    ----------
    ordinal_probabilities: a np.array containing the ordinal probabilities
    ----------
    Returns the normalized entropy value
    """
    Smax = np.log(len(ordinal_probabilities))
    ordinal_probabilities = ordinal_probabilities[ordinal_probabilities!=0]
    S = -np.sum(ordinal_probabilities*np.log(ordinal_probabilities))/Smax

    return(S)


def Jensen_Shannon_divergence(ordinal_probabilities):
    """
    This function calculates the normalized Jensen Shannon divergence between 
    the ordinal_probabilities and the uniform distribution.
    
    Parameters
    ----------
    ordinal_probabilities: a np.array containing the ordinal probabilities
    ----------
    Returns the normalized divergence value
    """
    n_states = len(ordinal_probabilities)
    uniform_probabilities = np.full(n_states,1./n_states)
    
    P_plus_U_over_2 = (uniform_probabilities + ordinal_probabilities)/2.  
    S_of_P_plus_U_over_2 = -np.sum(P_plus_U_over_2*np.log(P_plus_U_over_2))

    ordinal_probabilities = ordinal_probabilities[ordinal_probabilities!=0]
    S_of_P_over_2 = -np.sum(ordinal_probabilities*np.log(ordinal_probabilities))/2
    S_of_U_over_2 = np.log(n_states)/2.

    JS_div_max = -0.5*( ((n_states+1)/n_states)*np.log(n_states+1) 
                       + np.log(n_states) - 2*np.log(2*n_states)
                      )    
    JS_div = S_of_P_plus_U_over_2 - S_of_P_over_2 - S_of_U_over_2

    return(JS_div/JS_div_max)


def ordinal_probabilities(data, dx=3, dy=1, tau_x=1, tau_y=1, sort='none'):
    """
    This function evaluates the ordinal probabilities
    from a list time series or a list of matrices.
    
    Parameters
    ----------
    data: a list of np.arrays containing the time series or matrices
    dx: the embedding dimension in the horizontal axis (Default: 3)
    dy: the embedding dimension in the vertical axis, shoud me 1 for a time series (Default: 1)
    tau_x: the embedding delay in the horizontal axis (Default: 1)
    tau_y: the embedding delay in the vertical axis (Default: 1)
    sort: 'ascending' to sort the pdf by values in ascending order,
          'descending' to sort the pdf by values in descending order,
          'bystates' to sort the pdf by states,
          'none' to not sort (Default: 'none')
    ----------
    Returns (state labels, probabilities)
    """
    pdfs = []
    for datum in data:
        states, missing_states, probabilities = get_ordinal_probabilities(datum,dx,dy,tau_x,tau_y)
        all_states_str = np.apply_along_axis(lambda x: np.array2string(x,separator=''), 1, 
                                             np.concatenate([states,missing_states]))
        if sort=='none':
            pdfs.append([all_states_str,probabilities])
        if sort=='ascending':
            index_sort = np.argsort(probabilities)
            pdfs.append([all_states_str[index_sort],probabilities[index_sort]])
        if sort=='descending':
            index_sort = np.argsort(probabilities)
            index_sort = index_sort[::-1]
            pdfs.append([all_states_str[index_sort],probabilities[index_sort]])
        if sort=='bystates':
            index_sort = np.argsort(all_states_str)
            pdfs.append([all_states_str[index_sort],probabilities[index_sort]])
    return(pdfs)


def complexity_entropy_plane(data, dx=3, dy=1, tau_x=1, tau_y=1):
    """
    This function evaluates the permutation entropy and the statistical complexity
    from a list time series or a list of matrices.
    
    Parameters
    ----------
    data: a list of np.arrays containing the time series or matrices
    dx: the embedding dimension in the horizontal axis (Default: 3)
    dy: the embedding dimension in the vertical axis, shoud me 1 for a time series (Default: 1)
    tau_x: the embedding delay in the horizontal axis (Default: 1)
    tau_y: the embedding delay in the vertical axis (Default: 1)
    ----------
    Returns (permutation entropy, statistical complexity)
    """
    CE = []
    for datum in data:
        _, _, probabilities = get_ordinal_probabilities(datum,dx,dy,tau_x,tau_y)
        S = Shannon_entropy(probabilities)
        JS_div = Jensen_Shannon_divergence(probabilities)
        CE.append([S,S*JS_div])
    return(CE)

    