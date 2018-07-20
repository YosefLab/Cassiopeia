from collections import defaultdict

def read_and_process_data(filename, lineage_group=None, intBC_minimum_appearance = 0.1):
	"""
	Given an alleletable file, converts the corresponding cell samples to a string format
	to be passed into the solver


	:param filename:
		The name of the corresponding allele table file
	:param lineage_group:
		The lineage group as a string to draw samples from, None if all samples are wanted
	:param intBC_minimum_appearance:
		The minimum percentage appearance an integration barcode should have across the lineage group
	:return:
		A dictionary mapping samples to their corresponding string representation
		Please use .values() to get the corresponding

	"""


	# Reading in the samples
	samples = defaultdict(dict)
	int_bc_counts = defaultdict(int)
	for line in open(filename):

		if 'cellBC' in line:
			pass
		else:
			line = line.split()
			if lineage_group in line[6] or not lineage_group:
				samples[line[0]][line[1] + '_1'] = line[3]
				samples[line[0]][line[1] + '_2'] = line[4]
				samples[line[0]][line[1] + '_3'] = line[5]
				int_bc_counts[line[1]] += 1

	# Filtering samples for integration barcodes that appear in at least intBC_minimum_appearance of samples
	filtered_samples = defaultdict(dict)
	for sample in samples:
		for allele in samples[sample]:
			if int_bc_counts[allele.split('_')[0]] >= len(samples) * intBC_minimum_appearance:
				filtered_samples[sample][allele] = samples[sample][allele]

	# Finding all unique integration barcodes
	intbc_uniq = set()
	for s in filtered_samples:
		for key in filtered_samples[s]:
			intbc_uniq.add(key)

	# Converting the samples to string format
	samples_as_string = defaultdict(str)
	int_bc_counter = defaultdict(dict)
	for intbc in sorted(list(intbc_uniq)):
		for sample in filtered_samples:

			if intbc in filtered_samples[sample]:
				if filtered_samples[sample][intbc] == 'None':
					samples_as_string[sample] += '0|'
				else:
					if filtered_samples[sample][intbc] in int_bc_counter[intbc]:
						samples_as_string[sample] += str(
							int_bc_counter[intbc][filtered_samples[sample][intbc]] + 1) + '|'
					else:
						int_bc_counter[intbc][filtered_samples[sample][intbc]] = len(int_bc_counter[intbc]) + 1
						samples_as_string[sample] += str(
							int_bc_counter[intbc][filtered_samples[sample][intbc]] + 1) + '|'

			else:
				samples_as_string[sample] += '-|'
	# Dropping final | in string
	for sample in samples_as_string:
		samples_as_string[sample] = samples_as_string[sample][:-1]

	return samples_as_string

def convert_network_to_newick_format(graph):
	"""
	Given a networkx network, converts to proper Newick format.

	TODO: Add options for edge weights
	:param graph: Networkx graph object
	:return: String in newick format representing the above graph
	"""

	def _to_newick_str(g, node):
		is_leaf = g.out_degree(node) == 0
		return '%s' % (node,) if is_leaf else (
					'(' + ','.join(_to_newick_str(g, child) for child in g.successors(node)) + ')')

	def to_newick_str(g, root=0):  # 0 assumed to be the root
		return _to_newick_str(g, root) + ';'

	return to_newick_str(graph, [node for node in graph if graph.in_degree(node) == 0][0])