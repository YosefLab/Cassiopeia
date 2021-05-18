"""This file stores a variety of ways to calculate the similarity used for STDR with neighborjoining
"""

import numpy as np
import scipy
import spectraltree

def JC_hamming_sim(vals):
    return spectraltree.JC_similarity_matrix(vals)

def hamming_sim(vals): #
    hamming_matrix = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(vals, metric='hamming'))
    return 1 - hamming_matrix

def hamming_sim_ignore_missing(vals): #
    missing_val = -1
    hamming_matrix = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(vals, metric='hamming'))
    missing_array = (vals==missing_val)
    pdist = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(missing_array, lambda u,v: np.sum(np.logical_and(u,v))))
    
    return vals.shape[1] - hamming_matrix*vals.shape[1] - pdist

def hamming_sim_ignore_missing_normalize_over_nonmissing(vals): #
    missing_val = -1
    hamming_matrix = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(vals, metric='hamming'))
    missing_array = (vals==missing_val)
    pdist_xor = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(missing_array, lambda u,v: np.sum(np.logical_xor(u,v))))
    pdist_or = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(missing_array, lambda u,v: np.sum(np.logical_or(u,v))))
    
    ret_mat = 1 - (hamming_matrix*vals.shape[1] - pdist_xor) / (np.ones_like(hamming_matrix) * vals.shape[1] - pdist_or)
    ret_mat[np.isnan(ret_mat)] = 0
    ret_mat[np.isinf(ret_mat)] = 0
    return ret_mat
    
def hamming_sim_ignore_uncut(vals): #
    uncut_state = 0
    hamming_matrix = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(vals, metric='hamming'))
    uncut_array = (vals==uncut_state)
    pdist = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(uncut_array, lambda u,v: np.sum(np.logical_and(u,v))))
    return vals.shape[1] - hamming_matrix*vals.shape[1] - pdist

def hamming_sim_ignore_uncut_normalize_over_cut(vals): #
    uncut_state = 0
    hamming_matrix = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(vals, metric='hamming'))
    uncut_array = (vals==uncut_state)
    pdist_xor = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(uncut_array, lambda u,v: np.sum(np.logical_xor(u,v))))
    pdist_or = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(uncut_array, lambda u,v: np.sum(np.logical_or(u,v))))
    
    ret_mat = 1 - (hamming_matrix*vals.shape[1] - pdist_xor) / (np.ones_like(hamming_matrix) * vals.shape[1] - pdist_or)
    ret_mat[np.isnan(ret_mat)] = 0
    ret_mat[np.isinf(ret_mat)] = 0
    return ret_mat
    
def hamming_sim_ignore_both(vals): #
    missing_val = -1
    uncut_state = 0
    hamming_matrix = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(vals, metric='hamming'))
    missing_array = (vals==missing_val)
    uncut_array = (vals==uncut_state)
    missing_pdist = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(missing_array, lambda u,v: np.sum(np.logical_and(u,v))))
    uncut_pdist = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(uncut_array, lambda u,v: np.sum(np.logical_and(u,v))))
    return (1 - hamming_matrix)*vals.shape[1] - missing_pdist - uncut_pdist

def hamming_sim_ignore_both_normalize_over_nonmissing(vals): #
    missing_val = -1
    uncut_state = 0
    hamming_matrix = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(vals, metric='hamming'))
    missing_array = (vals==missing_val)
    uncut_array = (vals==uncut_state)
    missing_and = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(missing_array, lambda u,v: np.sum(np.logical_and(u,v))))
    missing_or = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(missing_array, lambda u,v: np.sum(np.logical_or(u,v))))
    uncut_and = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(uncut_array, lambda u,v: np.sum(np.logical_and(u,v))))
    
    ret_mat = ((1 - hamming_matrix)*vals.shape[1] - missing_and - uncut_and) / (np.ones_like(hamming_matrix) * vals.shape[1] - missing_or)
    ret_mat[np.isnan(ret_mat)] = 0
    ret_mat[np.isinf(ret_mat)] = 0
    return ret_mat
    
def hamming_sim_ignore_both_normalize_over_cut(vals): #
    missing_val = -1
    uncut_state = 0
    hamming_matrix = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(vals, metric='hamming'))
    missing_array = (vals==missing_val)
    uncut_array = (vals==uncut_state)
    missing_and = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(missing_array, lambda u,v: np.sum(np.logical_and(u,v))))
    uncut_or = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(uncut_array, lambda u,v: np.sum(np.logical_or(u,v))))
    uncut_and = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(uncut_array, lambda u,v: np.sum(np.logical_and(u,v))))
    
    ret_mat = ((1 - hamming_matrix)*vals.shape[1] - missing_and - uncut_and) / (np.ones_like(hamming_matrix) * vals.shape[1] - uncut_or)
    ret_mat[np.isnan(ret_mat)] = 0
    ret_mat[np.isinf(ret_mat)] = 0
    return ret_mat
    
def hamming_sim_ignore_both_normalize_over_both(vals): #
    missing_val = -1
    uncut_state = 0
    hamming_matrix = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(vals, metric='hamming'))
    missing_array = (vals==missing_val)
    uncut_array = (vals==uncut_state)
    either_array = missing_array + uncut_array
    missing_and = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(missing_array, lambda u,v: np.sum(np.logical_and(u,v))))
    uncut_and = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(uncut_array, lambda u,v: np.sum(np.logical_and(u,v))))
    either_or = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(either_array, lambda u,v: np.sum(np.logical_or(u,v))))
    
    ret_mat = np.divide((1 - hamming_matrix)*vals.shape[1] - missing_and - uncut_and, np.ones_like(hamming_matrix) * vals.shape[1] - either_or)
    ret_mat[np.isnan(ret_mat)] = 0
    ret_mat[np.isinf(ret_mat)] = 0
    return ret_mat