from __future__ import division

import pandas as pd
import sys
import numpy as np
import math

def binarize(mat):

    num_states_dict = []
    bin_chars = 0
    for i in mat.columns:

        m = mat[i].max()
        if m == '-' or m == '0':
            m = 1
        else:
            m = int(m)

        m = math.ceil(np.log2(m)) if m > 1 else 1
        num_states_dict.append(int(m))
        # we're going to convert each multistate range into binary, therefore
        # increasing the number of characters by ceil(log2(m)))
        # e.g. 9 states for a character -> 4 new characters
        bin_chars += int(m)

    return num_states_dict

def multi_map(mat):

    num_states_dict = []
    bin_chars = 0
    ind_col = mat.columns[0]
    for i in mat.columns:

        if i == ind_col:
            num_states_dict.append(mat.shape[0])
            continue

        m = 0
        for j in mat[i]:
            if j != "-":
                m = max(int(j), m)
                
        if m == 0:
            m = 1
        else:
            m = m + 1

        num_states_dict.append(m)

    return num_states_dict

def convert_to_one_hot(char, num_bin):

    if char == "-":
        return '?' * (num_bin)

    s = [0] * num_bin
    s[int(char)] = 1

    return ''.join(map(lambda x: str(int(x)), s))

def construct_file(charmat, state_map, relaxed=False):

    strings = []

    for i in range(charmat.shape[0]):

        if not relaxed:
            name = "s" + str(charmat.iloc[i, 0])
            while len(name) < 10:
                name += " "

        else:
            name = str(charmat.iloc[i, 0]) + " " 

        curr_len = 0
        s = name
        for j in range(1, charmat.shape[1]):

            num_bin = state_map[j]
            char = charmat.iloc[i, j]
            to_write = convert_to_one_hot(char, num_bin)

            s += to_write
            curr_len += len(to_write)

        strings.append(s + "\n")

    return strings, curr_len

if __name__ == "__main__":

    charmat_fp = sys.argv[1]
    outputfile_fp = sys.argv[2]
    relaxed_cmd = ""
    if len(sys.argv) > 3:
        relaxed_cmd = sys.argv[3]

    relaxed = False
    if relaxed_cmd == "--relaxed":
        relaxed = True

    charmat = pd.read_csv(charmat_fp, sep='\t')

    state_map = multi_map(charmat)

    num_samples = charmat.shape[0]

    strings, m = construct_file(charmat, state_map, relaxed=relaxed)

    with open(outputfile_fp, "w") as f:

        f.write("\t" + str(num_samples) + " " + str(m) + "\n")

        for i in strings:
            f.write(i)
