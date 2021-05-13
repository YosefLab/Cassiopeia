"""This file stores a variety of ways to calculate the similarity used for STDR with neighborjoining
"""

import numpy as np
import scipy
import spectraltree

def JC_similarity_matrix(vals):
    return spectraltree.JC_similarity_matrix(vals)

def hamming_similarity_matrix(vals):
    """Hamming Similarity"""
    classes = np.unique(vals)
    if classes.dtype == np.dtype('<U1'):
        # needed to use hamming distance with string arrays
        vord = np.vectorize(ord)
        vals = vord(vals)
    k = len(classes)
    hamming_matrix = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(vals, metric='hamming'))
    return 1 - hamming_matrix

def hamming_sim_ignore_missing_values(vals):
    missing_val = -1
    classnames, indices = np.unique(vals, return_inverse=True)
    num_arr = indices.reshape(vals.shape)
    hamming_matrix = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(num_arr, metric='hamming'))
    missing_array = (vals==missing_val)
    pdist_xor = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(missing_array, lambda u,v: np.sum(np.logical_xor(u,v))))
    
    return 1 - (hamming_matrix*vals.shape[1] - pdist_xor) / (np.ones_like(hamming_matrix) * vals.shape[1])

def hamming_sim_normalize_missing_values(vals):
    missing_val = -1
    classnames, indices = np.unique(vals, return_inverse=True)
    num_arr = indices.reshape(vals.shape)
    hamming_matrix = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(num_arr, metric='hamming'))
    missing_array = (vals==missing_val)
    pdist_xor = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(missing_array, lambda u,v: np.sum(np.logical_xor(u,v))))
    pdist_or = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(missing_array, lambda u,v: np.sum(np.logical_or(u,v))))
    
    return 1 - (hamming_matrix*vals.shape[1] - pdist_xor) / (np.ones_like(hamming_matrix) * vals.shape[1] - pdist_or)

def shared_mutations_normalize_missing_values(vals):
    missing_val = -1
    uncut_state = 0
    classnames, indices = np.unique(vals, return_inverse=True)
    num_arr = indices.reshape(vals.shape)
    hamming_matrix = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(num_arr, metric='hamming'))
    missing_array = (vals==missing_val)
    uncut_array = (vals==uncut_state)
    pdist_and = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(missing_array, lambda u,v: np.sum(np.logical_and(u,v))))
    pdist_or = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(missing_array, lambda u,v: np.sum(np.logical_or(u,v))))
    uncut_dist_and = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(uncut_array, lambda u,v: np.sum(np.logical_and(u,v))))
    return 1 - (((1 - hamming_matrix)*vals.shape[1] - pdist_and - uncut_dist_and) / (np.ones_like(hamming_matrix) * vals.shape[1] - pdist_or))