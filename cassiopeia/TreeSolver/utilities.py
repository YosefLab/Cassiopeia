from collections import defaultdict
import networkx as nx
import random
import numpy as np

from skbio.tree import TreeNode, majority_rule
from io import StringIO

from tqdm import tqdm
from cassiopeia.TreeSolver.Cassiopeia_Tree import Cassiopeia_Tree
from cassiopeia.TreeSolver.Node import Node
from cassiopeia.TreeSolver.data_pipeline import convert_network_to_newick_format
from cassiopeia.TreeSolver.lineage_solver.solver_utils import node_parent

def tree_collapse(tree):
	"""
	Given a networkx graph in the form of a tree, collapse two nodes together if there are no mutations seperating the two nodes

	:param graph: Networkx Graph as a tree
	:return: Collapsed tree as a Networkx object
	"""
	if isinstance(tree, Cassiopeia_Tree):
		graph = tree.get_network()
	else:
		graph = tree

	new = {}
	for node in graph.nodes():

		if isinstance(node, Node):
			new[node] = node.char_string
		else:
			new[node] = node.split("_")[0]

	new_graph = nx.relabel_nodes(graph, new)

	dct = defaultdict(str)
	while len(dct) != len(new_graph.nodes()):
		for node in new_graph:
			if '|' in node:
				dct[node] = node
			else:
				succ = list(new_graph.successors(node))
				if len(succ) == 1:
						if '|' in succ[0]:
							 dct[node] = succ[0]
						elif '|' in dct[succ[0]]:
							dct[node] = dct[succ[0]]
				else:
					if '|' in succ[0] and '|' in succ[1]:
							dct[node] = node_parent(succ[0], succ[1])
					elif '|' in dct[succ[0]] and '|' in succ[1]:
							dct[node] = node_parent(dct[succ[0]], succ[1])
					elif '|' in succ[0] and '|' in dct[succ[1]]:
							dct[node] = node_parent(succ[0], dct[succ[1]])
					elif '|' in dct[succ[0]] and '|' in dct[succ[1]]:
							dct[node] = node_parent(dct[succ[0]], dct[succ[1]])

	new_graph = nx.relabel_nodes(new_graph, dct)
	new_graph.remove_edges_from(new_graph.selfloop_edges())

	final_dct = {}
	for n in new_graph:

		final_dct[n] = Node('state-node', character_vec = n.split("|"))
	
	new_graph = nx.relabel_nodes(new_graph, final_dct)		

	# new2 = {}
	# for node in new_graph.nodes():
	# 	new2[node] = node+'_x'
	# new_graph = nx.relabel_nodes(new_graph, new2)

	return new_graph

def tree_collapse2(tree):
    """
    Given a networkx graph in the form of a tree, collapse two nodes together if there are no mutations seperating the two nodes

    :param graph: Networkx Graph as a tree
    :return: Collapsed tree as a Networkx object
    """
    if isinstance(tree, Cassiopeia_Tree):
        graph = tree.get_network()
    else:
        graph = tree
    
    
    leaves = [n for n in graph if graph.out_degree(n) == 0]
    for l in leaves:
        
        # traverse up beginning at leaf 
        parent = list(graph.predecessors(l))[0]
        pair = (parent, l)
        
        root = [n for n in graph if graph.in_degree(n) == 0][0] # need to reinstantiate root, in case we reassigned it
        
        is_root = False
        while not is_root:
            
            u, v = pair[0], pair[1]
            if u == root:
                is_root = True
                    
            if u.get_character_string() == v.get_character_string():
                
                # replace u with v
                children_of_parent = graph.successors(u)
                
                if not is_root:
                    new_parent = list(graph.predecessors(u))[0]
                    
                graph.remove_node(u) #removes parent node
                for c in children_of_parent:
                    if c != v:
                        graph.add_edge(v, c)
                
                if not is_root:
                    graph.add_edge(new_parent, v)
                    pair = (new_parent, v)
                
            else: 
                if not is_root:
                    pair = (list(graph.predecessors(u))[0], u)
                    
    return graph

def find_parent(node_list):
	parent = []
	if isinstance(node_list[0], Node):
		node_list = [n.get_character_string() for n in node_list]


	num_char = len(node_list[0].split("|"))
	for x in range(num_char):
		inherited = True

		state = node_list[0].split("|")[x]
		for n in node_list:
			if n.split("|")[x] != state:
				inherited = False
			
		if not inherited:
			parent.append("0")
			
		else: 
			parent.append(state)
	
	return Node('state-node', parent, is_target=False)

def fill_in_tree(tree, cm = None):
	
	# rename samples to character strings
	rndct = {}
	for n in tree.nodes:
		if cm is None:
			if '|' in n:
				rndct[n] = Node('state-node', n.split("_")[0].split('|'), is_target = False)
		else:
			if n.name in cm.index.values:
				rndct[n] = Node('state-node', [str(k) for k in cm.loc[n.name].values], is_target = True)
			else:
				rndct[n] = Node('state-node', '', is_target=False)

	tree = nx.relabel_nodes(tree, rndct)
	
	root = [n for n in tree if tree.in_degree(n) == 0][0]
	
	# run dfs and reconstruct states 
	anc_dct = {}
	for n in tqdm(nx.dfs_postorder_nodes(tree, root)):
		if '|' not in n.char_string or len(n.char_string) == 0:
			children = list(tree[n].keys())
			for c in range(len(children)):
				if children[c] in anc_dct:
					children[c] = anc_dct[children[c]]
			anc_dct[n] = find_parent(children)
			
	tree = nx.relabel_nodes(tree, anc_dct)

	tree.remove_edges_from(tree.selfloop_edges())

	return tree

def find_consensus_tree(trees, character_matrix, cutoff = 0.5):

	_trees = [TreeNode.read(StringIO(convert_network_to_newick_format(t.network))) for t in trees]
	np.random.shuffle(_trees)
	consensus = majority_rule(_trees, cutoff=cutoff)[0]
	G = nx.DiGraph()

	# Create dict from scikit bio TreeNode to cassiopeia.Node
	e2cass = {}
	for n in consensus.postorder():
		if n.name is not None:
			nn = Node(n.name, character_matrix.loc[n.name].values, is_target = True, support = n.support)
		else:
			nn = Node('state-node', [], support = n.support)

		e2cass[n] = nn
		G.add_node(nn)

	for p in consensus.traverse('postorder'):

		pn = e2cass[p]

		for c in p.children:
			cn = e2cass[c]

			G.add_edge(pn, cn)

	return G
