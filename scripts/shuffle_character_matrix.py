from __future__ import division

import numpy as np
import pandas as pd
import pickle as pic

import argparse

import sys
import os

from tqdm import tqdm

def series_setdiff(s1, s2):

    l = [] # list of disagreeing indices
    for i in s1.index:
        for j in s2.index:

            if i == j and s1.loc[i] != s2.loc[j]:
                l.append(i)

    return l

def shuffle_mat(char_mat, N=10000):
    """
    TODO: Write descriptoin of algorithm.
    """
    
    # create copy of char_mat to shuffle
    shuff = char_mat.copy(deep = True)

    samples = shuff.index

    pbar = tqdm(total=N, desc="Shuffling")
    num_success = 0
    while num_success < N:

        si, sj = np.random.choice(samples), np.random.choice(samples)
        
        # make sure that si != sj
        while si == sj:
            si, sj = np.random.choice(samples), np.random.choice(samples)

        cs_i, cs_j = shuff.loc[si], shuff.loc[sj] 
        nnz_i, nnz_j = cs_i[(cs_i != 0) & (cs_i.astype('str') != "-")], cs_j[(cs_j != 0) & (cs_j.astype('str') != "-")]

        sdiff = series_setdiff(nnz_i, nnz_j) # list of possible switching locations (where state of i != state j)

        ck, cl = np.random.choice(nnz_i.index), np.random.choice(nnz_j.index)
        
        if len(sdiff) < 1:
            continue 

        # want to make sure that we're actually switching states here
        # check against registry of characters where si and sj have different states
        while ck not in sdiff or cl not in sdiff:
    #        print(ck, cl, cs_i, cs_j, sdiff)
            ck, cl = np.random.choice(nnz_i.index), np.random.choice(nnz_j.index)

        shuff.loc[si, cl], shuff.loc[sj, ck] = shuff.loc[sj, cl], shuff.loc[si, ck] 

        num_success += 1
        pbar.update(1)

    pbar.close()

    return shuff


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("char_fp", type=str, help="character matrix to shuffle")
    parser.add_argument("out_fp", type=str, help="output file name for shuffled character matrix")
    parser.add_argument("-N", type=int, default=10000, help="Number of successful shuffles to complete")

    args = parser.parse_args()

    char_fp = args.char_fp
    out_fp = args.out_fp
    N = args.N

    charmat = pd.read_csv(char_fp, sep='\t', index_col = 0)

    shuffled_character_matrix = shuffle_mat(charmat, N=N)

    shuffled_character_matrix.to_csv(out_fp, sep='\t')
    
