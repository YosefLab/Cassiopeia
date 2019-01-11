import sys

import numpy as np
import pandas as pd
from collections import defaultdict, OrderedDict

from tqdm import tqdm
import math

import pickle as pic

def get_indel_props(at):

    uniq_alleles = np.union1d(at["r1"], np.union1d(at["r2"], at["r3"]))

    groups = at.groupby("intBC").agg({"r1": "unique", "r2": "unique", "r3": "unique"})

    count = defaultdict(int)

    for i in tqdm(groups.index, desc="Counting unique alleles"):
        alleles = np.union1d(groups.loc[i, "r1"], np.union1d(groups.loc[i, "r2"], groups.loc[i, "r3"]))
        for a in alleles:
            if a != a:
                continue
            if "None" not in a:
                count[a] += 1

    tot = len(groups.index)
    freqs = dict(zip(list(count.keys()), [ v / tot for v in count.values()]))

    return_df = pd.DataFrame([count, freqs]).T
    return_df.columns = ["count", "freq"]

    return_df.index.name = "indel"
    return return_df

def process_allele_table(cm, old_r = False, mutation_map=None):

    filtered_samples = defaultdict(OrderedDict)
    for sample in cm.index:
        cell = cm.loc[sample, "cellBC"]
        if old_r:
            filtered_samples[cell][cm.loc[sample, 'intBC'] + '_1'] = cm.loc[sample, 'r1.old']
            filtered_samples[cell][cm.loc[sample, 'intBC'] + '_2'] = cm.loc[sample, 'r2.old']
            filtered_samples[cell][cm.loc[sample, 'intBC'] + '_3'] = cm.loc[sample, 'r3.old']
        else:
            filtered_samples[cell][cm.loc[sample, 'intBC'] + '_1'] = cm.loc[sample, 'r1']
            filtered_samples[cell][cm.loc[sample, 'intBC'] + '_2'] = cm.loc[sample, 'r2']
            filtered_samples[cell][cm.loc[sample, 'intBC'] + '_3'] = cm.loc[sample, 'r3']

    samples_as_string = defaultdict(str)
    allele_counter = defaultdict(OrderedDict)

    intbc_uniq = []
    for s in filtered_samples:
        for key in filtered_samples[s]:
            if key not in intbc_uniq:
                intbc_uniq.append(key)

    prior_probs = defaultdict(dict)
    indel_to_charstate = defaultdict(dict)
    # for all characters
    for i in tqdm(range(len(list(intbc_uniq))), desc="Processing characters"):

        c = list(intbc_uniq)[i]

        # for all samples, construct a character string
        for sample in filtered_samples.keys():

            if c in filtered_samples[sample]:

                state = filtered_samples[sample][c]

                if type(state) != str and np.isnan(state):
                    samples_as_string[sample] += "-|"
                    continue

                if state == "NONE" or "None" in state:
                    samples_as_string[sample] += '0|'
                else:
                    if state in allele_counter[c]:
                        samples_as_string[sample] += str(allele_counter[c][state] + 1) + '|'
                    else:
                        # if this is the first time we're seeing the state for this character,
                        allele_counter[c][state] = len(allele_counter[c]) + 1
                        samples_as_string[sample] += str(allele_counter[c][state] + 1) + '|'

                        # add a new entry to the character's probability map
                        if mutation_map is not None:
                            prob = np.mean(mutation_map.loc[state]['freq'])
                            prior_probs[i][str(len(allele_counter[c]) + 1)] = float(prob)
                            indel_to_charstate[i][str(len(allele_counter[c]) + 1)] = state
            else:
                samples_as_string[sample] += '-|'
    for sample in samples_as_string:
        samples_as_string[sample] = samples_as_string[sample][:-1]

    return samples_as_string, prior_probs, indel_to_charstate

def string_to_cm(string_sample_values):

    m = len(string_sample_values[list(string_sample_values.keys())[0]].split("|"))
    n = len(string_sample_values.keys())

    cols = ["r" + str(i) for i in range(m)]
    cm = pd.DataFrame(np.zeros((n, m)))
    indices = []
    for i, k in zip(range(n), string_sample_values.keys()):
        indices.append(k)
        alleles = np.array(string_sample_values[k].split("|"))
        cm.iloc[i,:] = alleles

    cm.index = indices
    cm.index.name = "cellBC"
    cm.columns = cols

    return cm



def write_to_charmat(string_sample_values, out_fp):

    m = len(string_sample_values[list(string_sample_values.keys())[0]].split("|"))

    with open(out_fp, "w") as f:

        cols = ["cellBC"] + [("r" + str(i)) for i in range(m)]
        f.write('\t'.join(cols) + "\n")

        for k in string_sample_values.keys():

            f.write(k)
            alleles = string_sample_values[k].split("|")

            for a in alleles:
                f.write("\t" + str(a))

            f.write("\n")

def alleletable_to_character_matrix(at, out_fp=None, mutation_map = None, old_r = False, write=True):


    out_stem = ''.join(out_fp.split('.')[:-1])

    character_matrix_values, prior_probs, indel_to_charstate = process_allele_table(at, old_r = old_r, mutation_map=mutation_map)

    if mutation_map is not None:
        # write prior probability dictionary to pickle for convenience
        pic.dump(prior_probs, open(out_stem + "_priorprobs.pkl", "wb"))

        # write indel to character state mapping to pickle
        pic.dump(indel_to_charstate, open(out_stem + "_indel_character_map.pkl", "wb"))

    if write:
        if out_fp is None:
            raise Exception("Need to specify an output file if writing to file")

        write_to_charmat(character_matrix_values, out_fp)

    else:
        return string_to_cm(character_matrix_values)
