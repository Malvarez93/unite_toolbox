import numpy as np

from scipy.stats import gaussian_kde
from scipy.integrate import nquad

def calc_kde_entropy(data, bandwidth=None):
    """Calculates the (joint) entropy of the input n-d array.

    Calculates the (joint) entropy of the input data [in nats] by
    approximating the (joint) density of the distribution using a
    Gaussian kernel density estimator (KDE). By defaul the Scott
    estimate for the bandwith is used for the Gaussian kernel.

    Parameters
    ----------
    data : numpy.ndarray
        Array of shape (n_samples, d_features)
    bandwidth : float, optional
        bandwidth of the gaussian kernel
    
    Returns
    -------
    h : float
        Entropy of the data set [in nats]"""
    
    lims = np.vstack((data.min(axis=0), data.max(axis=0))).T
    kde = gaussian_kde(data.T, bw_method=bandwidth)

    def eval_entropy(*args):
        p = kde.evaluate(np.vstack(args))
        return -1*p*np.log(p)
    
    h = nquad(eval_entropy, ranges=lims)[0]
    return h

def calc_kde_kld(data_p, data_q, bandwidth=None):
    """Calculates the the Kullback-Leibler divergence (relative entropy) between
    two n-d arrays of data.
    
    Calculates the Kullback-Leibler divergence (relative entropy) between
    two data sets (p and q) [in nats] by approximating both distributions using
    a Gaussian kernel density estimate (KDE). The divergence is measured between
    both of the estimated densities. Both density estimates are independent, therefore
    a different number of total samples in p and q is valid.

    Parameters
    ----------
    data_p : numpy.ndarray
        Array of shape (n_samples, d_features)
    data_q : numpy.ndarray
        Array of shape (m_samples, d_features)
    bw : float, optional
        bandwith of the gaussian kernel

    Returns
    -------
    kld : float
        Kullback-Leibler divergence between p and q [in nats]"""
    
    lims = np.vstack((data_p.min(axis=0), data_p.max(axis=0))).T
    p_kde = gaussian_kde(data_p.T, bw_method=bandwidth)
    q_kde = gaussian_kde(data_q.T, bw_method=bandwidth)

    def eval_kld(*args):
        p = p_kde.evaluate(np.vstack(args))
        q = q_kde.evaluate(np.vstack(args))
        return p * np.log(p / q)
    
    kld = nquad(eval_kld, ranges=lims)[0]
    return kld