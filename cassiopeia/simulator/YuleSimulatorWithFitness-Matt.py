"""
Matt's implementation of a simple forwrad-time Yule-based tree simulator with
fitness.
"""

import abc

from cassiopeia.data import CassiopeiaTree
from cassiopeia.simulator import TreeSimulator


class YuleSimulatorWithFitness_Matt(TreeSimulator.TreeSimulator):

    def  __init__(self, base_birth_rate: float = 1.0, death_rate: float = 0.0,
                number_of_generations: int = 100,
                fitness_function: lambda x: np.random.exponential(1),
                mutation_rate: 1):

            self.birth_rate = base_birth_rate
            self.death_rate = death_rate
            self.number_of_generations = number_of_generations
            self.fitness_function = fitness_function
            self.mutation_rate = mutation_rate

    @abc.abstractmethod
    def simulate_tree(self) -> CassiopeiaTree:
        """
        Simulate a CassiopeiaTree.

        The returned tree will have at least its tree topology initialized.
        """
        
        tree = nx.DiGraph()
        tree.add_node(0, s = 0)
        
        size_of_tree = 1
        
        node_id = 1
        
        for _ in tqdm(range(self.number_of_generations)):
                
            extant = [n for n in tree if tree.out_degree(n) == 0]
            
            mean_birth_rate = np.random.exponential(np.mean([self.birth_rate*(1 + tree.nodes[e]['s']) for e in extant]))
            
            for n in extant:
                
                lambda_n = mean_birth_rate * (1 + tree.nodes[n]['s'])
                
                life_span = np.random.exponential(lambda_n)
                
                if life_span < np.random.exponential(self.death_rate):
                    
                    parent = tree.predecessors(n)
                    tree.remove_node(n)
                    parent_children = [c for c in tree.successors(parent)]
                    
                    while len(parent_children) == 0:
                        n = parent
                        parent = tree.predecessors(n)
                        tree.remove_node(n)
                        parent_children = [c for c in tree.successors(parent)]
                    
                    continue
                    
                
                if life_span < (mean_birth_rate):
                
                    for children in range(2):

                        tree.add_edge(n, node_id, length = life_span)

                        number_of_mutations = np.random.poisson(self.mutation_rate)

                        tree.nodes[node_id]['s'] = tree.nodes[n]['s'] + np.sum([self.fitness_function() for m in range(number_of_mutations)])
                        node_id += 1
                    
            size_of_tree = len([n for n in tree if tree.out_degree(n) == 0])
                
        nwk = to_newick(tree)
        tree = ete3.Tree(nwk)
        
        all_leaves = tree.get_leaf_names()
            
        leaves_to_keep = np.random.choice(all_leaves, number_of_cells, replace=False)
        
        tree.prune([str(n) for n in leaves_to_keep])
        tree = solver_utilities.collapse_unifurcations(tree)

        cassiopeia_tree = CassiopeiaTree(tree=tree)
        
        return cassiopeia_tree
