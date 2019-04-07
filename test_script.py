import networkx as nx
from Cassiopeia.TreeSolver import Node
import Cassiopeia.TreeSolver.lineage_solver as ls 

n1 = Node('a', [1,0,0,0,0])
n2 = Node('b', [1,0,0,1,0])
n3 = Node('c', [1,0,0,2,0])
n4 = Node('d', [1,2,0,1,0])
n5 = Node('e', [1,1,0,1,0])
n6 = Node('f', [1,0,3,2,0])
n7 = Node('g', [0,0,0,0,1])
n8 = Node('h', [0,1,0,0,1])
n9 = Node('i', [0,1,2,0,1])
n10 = Node('j', [0,1,1,0,1])

nodes = [n1, n2, n3, n4, n5, n6, n7, n8, n9, n10]

#tree = ls.solve_lineage_instance(nodes, method='greedy')
#net = tree.get_network()
#root = [n for n in net if net.in_degree(n) == 0][0]
#print([(r.name, r.char_string, r.pid) for r in net if net.in_degree(r) == 0])
#print(root.name, root.char_string, root.pid)
#for e in nx.dfs_edges(net, root):
#for e in net.edges():
#    print((e[0].name, e[0].char_string, e[0].pid), '->', (e[1].name, e[1].char_string, e[1].pid))

#leaves = tree.get_leaves()
#for l in leaves:
#    print(l.name, l.char_string)
#print("is connected: " + str(len([n for n in net if net.in_degree(n) == 0])== 1))
#print("Number of targets: ", len([n for n in net if n.is_target]))

#tree = ls.solve_lineage_instance(nodes, method='ilp')
#net = tree.get_network()
#root = [n for n in net if net.in_degree(n) == 0][0]
#for e in nx.dfs_edges(net, root):
#    print((e[0].name, e[0].char_string), (e[1].name, e[1].char_string))

#print("is connected: " + str(len([n for n in net if net.in_degree(n) == 0])== 1))
#print("Number of targets: ", len([n for n in net if n.is_target]))

tree = ls.solve_lineage_instance(nodes, method='hybrid', hybrid_subset_cutoff=3)
net = tree.get_network()
root = [n for n in net if net.in_degree(n) == 0][0]
for e in nx.dfs_edges(net, root):
    print((e[0].name, e[0].char_string), (e[1].name, e[1].char_string))


print("is connected: " + str(len([n for n in net if net.in_degree(n) == 0])== 1))
print("Number of targets: ", len([n for n in net if n.is_target]))

