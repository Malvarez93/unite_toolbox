import numpy as np

from functools import reduce

def estimate_ideal_bins(data, counts=True):
    """
    Estimates the ideal number of bins for each column of a 2D data array using
    three different methods: Scott, Freedman-Diaconis, and Sturges.

    Parameters
    ----------
    data : numpy.ndarray
        Array of shape (n_samples, n_features)
    counts : bool, optional
        Whether to return the number of bins (True) or the bin edges (False).

    Returns
    -------
    dict
        A dictionary with a key for each method, and the values are lists of 
        number of bins or bin edges for each feature of the data.
    """
    
    _, n_features = data.shape
    
    methods = ["scott", "fd", "sturges"]
    ideal_bins = []
    
    for m in methods:
        d_bins = []
        for d in range(n_features):
            num_bins = np.histogram_bin_edges(data[:, d], bins=m)
            num_bins = len(num_bins) if counts == True else num_bins              
            d_bins.append(num_bins)
        ideal_bins.append(d_bins)
            
    return dict(zip(methods, ideal_bins))

def calc_bin_entropy(data, edges):
    """
    Calculates the (joint) entropy of the input data after binning it along each
    dimension using specified bin edges or number of bins.

    Parameters
    ----------
    data : numpy.ndarray
        Array of shape (n_samples, n_features)
    edges : list or int
        A list of length n_features which contains arrays describing the bin edges
        along each dimension or a list of ints describing the number of bins to use
        in each dimension. Input can also be a single int and the histogram will be
        created with the same number of bins for each dimension.

    Returns
    -------
    h : float
        The (joint) entropy of the input data after binning.
    cf : float
        The correction factor due to bin spacing. See Cover &
        Thomas (2006) Eq. 8.30 ISBN: 978-0-471-24195-9
    """

    # binning
    pi, edges = np.histogramdd(data, bins=edges)
    pi = pi / np.sum(pi)

    # volume
    edges = [np.diff(e) for e in edges]
    volume = reduce(np.outer, edges)

    # entropy
    ids = np.nonzero(pi)
    delta = volume[ids]
    pi = pi[ids]

    h = np.sum(-1.0 * pi * np.log(pi))
    cf = np.sum(pi * np.log(delta))

    return h, cf