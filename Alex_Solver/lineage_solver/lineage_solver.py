import concurrent.futures
import functools
import multiprocessing
import networkx as nx
import traceback

from greedy_solver import root_finder, greedy_build
from ILP_solver import generate_mSteiner_model, solve_steiner_instance
from solver_utils import build_potential_graph_from_base_graph

def solve_lineage_instance(target_nodes, prior_probabilities = None, method='hybrid', threads=8, hybrid_subset_cutoff=200):
	"""
	Aggregated lineage solving method, which given a set of target nodes, will find the maximum parsimony tree
	accounting the given target nodes

	:param target_nodes:
		A list of target nodes, where each node is in the form 'Ch1|Ch2|....|Chn'
	:param prior_probabilities:
		A nested dictionary containing prior probabilities for [character][state] mappings
		where characters are in the form of integers, and states are in the form of strings,
		and values are the probability of mutation from the '0' state.
	:param method:
		The method used for solving the problem ['ilp, 'hybrid', 'greedy']
			- ilp: Attempts to solve the problem based on steiner tree on the potential graph
				   (Recommended for instances with several hundred samples at most)
			- greedy: Runs a greedy algorithm to find the maximum parsimony tree based on choosing the most occurring split in a
				   top down fasion (Algorithm scales to any number of samples)
			- hybrid: Runs the greedy algorithm until there are less than hybrid_subset_cutoff samples left in each leaf of the
				   tree, and then returns a series of small instance ilp is then run on these smaller instances, and the
				   resulting graph is created by merging the smaller instances with the greedy top-down tree
	:param threads:
		The number of threads to use in parallel for the hybrid algorithm
	:param hybrid_subset_cutoff:
		The maximum number of nodes allowed before the greedy algorithm terminates for a given leaf node
	:return:
		A reconstructed subgraph representing the nodes
	"""
	master_root = root_finder(target_nodes)
	if method == "ilp":
		potential_network = build_potential_graph_from_base_graph(target_nodes, priors=prior_probabilities)

		model, edge_variables = generate_mSteiner_model(potential_network, master_root, set(target_nodes))
		
		subgraph = solve_steiner_instance(model, potential_network, edge_variables, MIPGap=.01, detailed_output=False,
							   time_limit=300)[0]
		return subgraph

	if method == "hybrid":
		network, target_sets = greedy_build(target_nodes, priors=prior_probabilities, cutoff=200)

		executor = concurrent.futures.ProcessPoolExecutor(min(multiprocessing.cpu_count(), 10))
		futures = [executor.submit(find_good_gurobi_subgraph, root, targets, prior_probabilities) for root, targets in target_sets]
		concurrent.futures.wait(futures)
		for future in futures:
			network = nx.compose(network, future.result())
		return network

	if method == "greedy":
		graph = greedy_build(target_nodes, priors=prior_probabilities, cutoff=-1)[0]
		return graph

	else:
		raise Exception("Please specify one of the following methods: ilp, hybrid, greedy")

def reraise_with_stack(func):

    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            traceback_str = traceback.format_exc(e)
            raise StandardError("Error occurred. Original traceback "
                                "is\n%s\n" % traceback_str)

    return wrapped

@reraise_with_stack
def find_good_gurobi_subgraph(root, targets, prior_probabilities):
	"""
	Sub-Function used for multi-threading in hybrid method

	:param root:
		Sub-root of the subgraph that is attempted to be reconstructed
	:param targets:
		List of sub-targets for a given subroot where each node is in the form 'Ch1|Ch2|....|Chn'
	:param prior_probabilities:
		A nested dictionary containing prior probabilities for [character][state] mappings
		where characters are in the form of integers, and states are in the form of strings,
		and values are the probability of mutation from the '0' state.
	:return:
		Optimal ilp subgraph for a given subset of nodes
	"""

	print "Started new thread for: " + str(root)

	if len(set(targets)) == 1:
		graph = nx.DiGraph()
		graph.add_node(root)
		return graph

	potential_network_priors = build_potential_graph_from_base_graph(targets, priors=prior_probabilities)

	model, edge_variables = generate_mSteiner_model(potential_network_priors, root, set(targets))
	subgraph = solve_steiner_instance(model, potential_network_priors, edge_variables, MIPGap=.01, detailed_output=False,
						   time_limit=120)[0]
	return subgraph

