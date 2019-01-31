import sys

import numpy as np
import pandas as pd
from collections import defaultdict, OrderedDict

from tqdm import tqdm
import math

import pickle as pic

def get_indel_props(at):
	"""
	Given an alleletable file, this function will split the alleletable into independent
	lineage groups and estimate the indel formation probabilities. This is done by
	treating each intBC as an independent observation and counting how many intBC contain
	a specific mutation, irrespective of the cell.

	:param at:
		The allele table pandas DataFrame
	:return:
		An M x 2 pandas DataFrame mapping all M mutations to the frequency and raw counts
		of how many intBC this mutation appeared on.

	"""

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
	"""
	Given an alleletable, create a character strings and lineage-specific mutation maps.
	A character string for a cell consists of a summary of all mutations observed at each
	cut site, delimited by a '|' character. We codify these mutations into integers, where
	each unique integer in a given column (character cut site) corresponds to a unique
	indel observed at that site.

	:param cm:
		The allele table pandas DataFrame.
	:param old_r:
		Do not use sequence context when identifying unique indels. (Default = False)
	:param mutation_map:
		A specification of the indel formation probabilities, in the form of a pandas
		DataFrame as returned from the `get_indel_props` function. (Default = None)
	:return:
		A list of three items - 1) a conversion of all cells into "character strings",
		2) A dictionary mapping each character to its mutation map. This mutation map
		is another dictionary storing the probabilities of mutating to any given
		state at that character. (e.g. {"character 1": {1: 0.5, 2: 0.5}})
		(C) dictionary mapping each indel to the number that represents it per
		character.
	"""

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
	"""
	Create a character matrix from the character strings, as created in `process_allele_table`.

	:param string_sample_values:
	   Input character strings, one for each cell.
	:return:
	   A Character Matrix, returned as a pandas DataFrame of size N x C, where we have
	   N cells and C characters.
	"""


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
	"""
	Write the character strings out to file.

	:param string_sample_values:
		Input character strings, one for each cell.
	:param out_fp:
		File path to be written to.
	:return:
		None.
	"""


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
	"""
	Wrapper function for creating character matrices out of allele tables.

	:param at:
		Allele table as a pandas DataFrame.
	:param out_fp:
		Output file path, only necessary when write = True (Default = None)
	:param mutation_map:
		Mutation map as a pandas DataFrame. This can be created with the
		`get_indel_props` function. (Default = None)
	:param old_r:
		Do not use sequence context when calling character states (Default = False)
	:param write:
		Write out to file. This requires `out_fp` to be specified as well. (Default = True)
	:return:
		None if write is specified. If not, return an N x C character matrix as a pandas DataFrame, the
		mutation map, and the indel to character state mapping. If writing out to file,
		the mutation and indel to character state mappings are also saved as pickle
		files.
	"""


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
		return string_to_cm(character_matrix_values), prior_probs, idnel_to_charstate
