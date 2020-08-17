import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import scipy as sp
import pickle as pic
from matplotlib.backends.backend_pdf import PdfPages
from collections import defaultdict
import time
start_time = time.time()

class Node:
    
    def __init__(self, char_list, num_states, parent=None, left=None, right=None):
        # num_states includes the 0 state. For example, if the possible states are 0 or 1, then num_states=2
        self.chars = char_list
        self.parent = parent
        self.left = left
        self.right = right
        self.num_chars = len(self.chars)
        self.num_states = num_states
        
    def is_leaf(self):
        return not (self.left or self.right)

    def duplicate(self, p, q=None, dropout_rate=0):
        assert len(p) == len(self.chars), "invalid p vector"
        if q:
            for i in range(len(p)):
                assert len(q[i]) + 1 == self.num_states, "invalid q[" + str(i) + "] vector"
                
        new_chars = []
        if not q:
            q = [None for i in self.chars]

        for l in range(len(self.chars)):
            if self.chars[l] != 0:
                new_chars.append(self.chars[l])
            else:
                if np.random.random() < p[l]:
                    new_chars.append(np.random.choice(np.arange(1, self.num_states), 1, p=q[l])[0])
                else:
                    new_chars.append(self.chars[l])
        
        return Node(new_chars, self.num_states)
    
    def __str__(self):
        s = ''
        for x in self.chars:
            s += str(x) + '|'
        return s[:-1]
                    
def simulation(p, num_states, time, q=None):
    root = Node([0 for i in p], num_states)
    curr_gen = [root]
    for t in range(time):
        new_gen = []
        for n in curr_gen:
            c1 = n.duplicate(p, q)
            c2 = n.duplicate(p, q)
            c1.parent = n
            c2.parent = n
            n.left = c1
            n.right = c2
            new_gen.append(c1)
            new_gen.append(c2)
        curr_gen = new_gen
    return curr_gen

def find_lineage(node):
    lineage = [node]
    while node.parent:
        node = node.parent
        lineage.insert(0, node)
    return lineage

def find_root(samples):
    return find_lineage(samples[0])[0]

def print_tree(root):
    
    def tree_str(node, level=0):
        ret = "\t"*level+str(node)+"\n"
        if node.left:
            ret += tree_str(node.left, level+1)
        if node.right:
            ret += tree_str(node.right, level+1)
        return ret
    
    print(tree_str(root))
    
def generate_frequency_matrix(samples, subset=None):
    k = samples[0].num_chars
    m = samples[0].num_states + 1
    F = np.zeros((k,m), dtype=int)
    if not subset:
        subset = list(range(len(samples)))
    for i in subset:
        for j in range(k):
            F[j][samples[i].chars[j]] += 1
    return F
            
def split_data(F):
    k,m = F.shape[0], F.shape[1]
    split_data = []
    for i in range(k):
        for j in range(1, m-1):
            split_data.append((i,j))
    split_data.sort(key=lambda tup: F[tup[0]][tup[1]], reverse=True)
    index = 0
    
    for i in range(5):
        s = ''
        for j in range(1, 5):
            a, b = split_data[index][0], split_data[index][1]
            s += str((a,b)) + " freq =" + str(F[a][b]) + " "
            index += 1
        print(s)
            
    
def construct_connectivity_graph(samples, subset=None):
    n = len(samples)
    k = samples[0].num_chars
    m = samples[0].num_states
    G = nx.Graph()
    if not subset:
        subset = range(n)
    for i in subset:
        G.add_node(i)
    F = generate_frequency_matrix(samples, subset)
    for i in subset:
        for j in subset:
            if j <= i:
                continue
            n1 = samples[i]
            n2 = samples[j]
            #compute simularity score
            score = 0
            for l in range(k):
                x = n1.chars[l]
                y = n2.chars[l]
                if min(x, y) >= 0 and max(x,y) > 0:
                    if x==y:
                        score -= 3*(len(subset) - F[l][x] - F[l][-1])
                    elif min(x,y) == 0:
                        score += F[l][max(x,y)] - 1
                    else:
                        score += (F[l][x] + F[l][y]) - 2
                        
                if score != 0:
                    G.add_edge(i,j, weight=score)
    return G

def max_cut_heuristic(G, sdimension, iterations, show_steps=False):
    #n = len(G.nodes())
    d = sdimension+1
    emb = {}        
    for i in G.nodes():
        x = np.random.normal(size=d)
        x = x/np.linalg.norm(x)
        emb[i] = x
        
    def show_relaxed_objective():
        score = 0
        for e in G.edges():
            u = e[0]
            v = e[1]
            score += G[u][v]['weight']*np.linalg.norm(emb[u]-emb[v])
        print(score)
        
    for k in range(iterations):
        new_emb = {}
        for i in G.nodes:
            cm = np.zeros(d, dtype=float)
            for j in G.neighbors(i):
                cm -= G[i][j]['weight']*np.linalg.norm(emb[i]-emb[j])*emb[j]
            cm = cm/np.linalg.norm(cm)
            new_emb[i] = cm
        emb = new_emb
        
    #print("final relaxed objective:")
    #show_relaxed_objective()
    return_set = set()
    best_score = 0
    for k in range(3*d):
        b = np.random.normal(size=d)
        b = b/np.linalg.norm(b)
        S = set()
        for i in G.nodes():
            if np.dot(emb[i], b) > 0:
                S.add(i)
        this_score = evaluate_cut(S, G)
        if this_score > best_score:
            return_set = S
            best_score = this_score
    #print("score before hill climb = ", best_score)
    improved_S = improve_cut(G, return_set)
    #final_score = evaluate_cut(improved_S, G)
    #print("final score = ", final_score)
    return improved_S

def improve_cut(G, S):
    #n = len(G.nodes())
    ip = {}
    new_S = S.copy()
    for i in G.nodes():
        improvement_potential = 0
        for j in G.neighbors(i):
            if cut(i,j,new_S):
                improvement_potential -= G[i][j]['weight']
            else:
                improvement_potential += G[i][j]['weight']
        ip[i] = improvement_potential
        
    all_neg = False
    iters = 0
    while (not all_neg) and (iters < 2*len(G.nodes)):
        best_potential = 0
        best_index = 0
        for i in G.nodes():
            if ip[i] > best_potential:
                best_potential = ip[i]
                best_index = i
        if best_potential > 0:
            for j in G.neighbors(best_index):
                if cut(best_index,j,new_S):
                    ip[j] += 2*G[best_index][j]['weight']
                else:
                    ip[j] -= 2*G[best_index][j]['weight']
            ip[best_index] = -ip[best_index]
            if best_index in new_S:
                new_S.remove(best_index)
            else:
                new_S.add(best_index)
        else:
            all_neg = True
        iters += 1
    #print("number of hill climbing interations: ", iters)
    return new_S

def evaluate_cut(S, G, B=None, show_total=False):
    cut_score = 0
    total_good = 0
    total_bad = 0
    for e in G.edges():
        u = e[0]
        v = e[1]
        w_uv = G[u][v]['weight']
        total_good += float(w_uv)
        if cut(u,v,S):
            cut_score += float(w_uv)

    if B:
        for e in B.edges():
            u = e[0]
            v = e[1]
            w_uv = B[u][v]['weight']
            total_bad += float(w_uv)
            if cut(u,v,S):
                cut_score -= float(w_uv)
            
    if show_total:
        print("total good = ", total_good)
        print("total bad = ", total_bad)
    return(cut_score)

def greedy_cut(samples, subset=None):
    F = generate_frequency_matrix(samples, subset)
    k,m = F.shape[0], F.shape[1]
    freq = 0
    char = 0
    state = 0
    if not subset:
        subset = list(range(len(samples)))
    for i in range(k):
        for j in range(1, m-1):
            if F[i][j] > freq and F[i][j] < len(subset) - F[i][-1] :
                char, state = i,j
                freq = F[i][j]
    if freq == 0:
        return random_nontrivial_cut(subset)
    S = set()
    Sc = set()
    missing = set()
    #print(char, state)
    for i in subset:
        if samples[i].chars[char] == state:
            S.add(i)
        elif samples[i].chars[char] == -1:
            missing.add(i)
        else:
            Sc.add(i)
            
    if not Sc:
        if len(S) == len(subset) or len(S) == 0:
            print(F)
            print(char, state, len(subset))
        return S
    
    for i in missing:
        s_score = 0
        sc_score = 0
        for j in S:
            for l in range(k):
                if samples[i].chars[l] > 0 and samples[i].chars[l] == samples[j].chars[l]:
                    s_score += 1
        for j in Sc:
            for l in range(k):
                if samples[i].chars[l] > 0  and samples[i].chars[l] == samples[j].chars[l]:
                    sc_score += 1
        if s_score/len(S) > sc_score/len(Sc):
            S.add(i)
        else:
            Sc.add(i)
        
    if len(S) == len(subset) or len(S) == 0:
            print(F)
            print(char, state, len(subset))
    return S
    
def random_cut(subset):
    S = set()
    for i in subset:
        if np.random.random() > 0.5:
            S.add(i)
    return S

def random_nontrivial_cut(subset):
    assert len(subset) > 1
    S = set()
    lst = list(subset)
    S.add(lst[0])
    for i in range(2,len(lst)):
        if np.random.random() > 0.5:
            S.add(lst[i])
    return S


def cut(u, v, S):
    return ((u in S) and (not v in S)) or ((v in S) and (not u in S))

def num_incorrect(S, h):
    num = 0
    for i in range(int(2**h/2)):
        if not i in S:
            num += 1

    for i in range(int(2**h/2), 2**h):
        if i in S:
            num += 1

    return min(num, 2**h - num)

def find_tree_lineage(i, T):
    p = list(T.predecessors(i))
    curr_node = i
    ancestor_list = [curr_node]
    while p:
        curr_node = p[0]
        ancestor_list.insert(0, curr_node)
        p = list(T.predecessors(curr_node))
    return(ancestor_list)

        
def outgroup(i, j, k, T):
    assert i != j and i != k and j != k, str(i) + ' ' + str(j) + ' ' + str(k) + ' not distinct'
    
    Li = find_tree_lineage(i, T)
    Lj = find_tree_lineage(j, T)
    Lk = find_tree_lineage(k, T)
    l = 0
    while Li[l] == Lj[l] and Lj[l] == Lk[l]:
        l += 1
    if Li[l] != Lj[l] and Li[l] != Lk[l] and Lj[l] != Lk[l]:
        return None
    if Li[l] == Lj[l]:
        return k
    if Li[l] == Lk[l]:
        return j
    if Lj[l] == Lk[l]:
        return i
    
    
    
def evaluate_split(S, subset, T, sample_size=1000):
    # assume S \subseteq T.leaves
    def S_outgroup(i,j,k):
        if (not cut(i,j,subset)) and (not cut(j,k,subset)):
            return None
        if not cut(i,j,subset):
            return k
        if not cut(i,k,subset):
            return j
        return i
    
    TC = 0
    TI = 0
    unresolved = 0
    superset = np.array(list(S))
    num_sampled = 0
    for a in range(sample_size):
        chosen = np.random.choice(superset, 3, replace=False)
        oS = S_outgroup(chosen[0], chosen[1], chosen[2])
        oT = outgroup(chosen[0], chosen[1], chosen[2], T)
        if oS == None or oT == None:
            unresolved += 1
        else:
            if oS == oT:
                TC += 1
            else:
                TI += 1
    return TC/sample_size, TI/sample_size, unresolved/sample_size
                  
def remove_duplicates(nodes, indices):
    indices = list(indices)
    indices.sort(key=lambda i: nodes[i].chars)
    final_set = set()
    i = 0
    j = 1
    while j < len(indices):
        if nodes[indices[i]].chars != nodes[indices[j]].chars:
            final_set.add(indices[i])
            i = j
        j += 1
    final_set.add(indices[i])
    return final_set
            
def mult_chain(a,b):
    f = 1
    for i in range(a, b+1):
        f*=i
    return f

def nCr(n, k):
    if k > n:
        return 0
    if k > n/2:
        return nCr(n, n-k)
    return int(mult_chain(n-k+1,n)/mult_chain(1,k))

def similarity(u, v, samples):
    k = samples[0].num_chars
    return sum([1 for i in range(k) if samples[u].chars[i] == samples[v].chars[i] and samples[u].chars[i] > 0])

def construct_similarity_graph(samples, subset=None, threshold=0):
    G = nx.Graph()
    if not subset:
        subset = range(len(samples))
    for i in subset:
        G.add_node(i)
    F = generate_frequency_matrix(samples, subset)
    k,m = F.shape[0], F.shape[1]
    for i in range(k):
        for j in range(1,m-1):
            if F[i][j] == len(subset) - F[i][-1]:
                threshold += 1
    for i in subset:
        for j in subset:
            if j <= i:
                continue
            s = similarity(i,j, samples) 
            if s > threshold:
                G.add_edge(i,j, weight=(s-threshold))
    return G

def spectral_split(G, k=2, method='Fiedler', return_eig=False, display=False):
    L = nx.normalized_laplacian_matrix(G).todense()
    diag = sp.linalg.eig(L)
    if k == 2 and method == 'Fiedler':
        v2 = diag[1][:, 1] 
        x = {}
        vertices = list(G.nodes())
        for i in range(len(vertices)):
            x[vertices[i]] = v2[i]
        vertices.sort(key=lambda v: x[v])
        total_weight = 2*sum([G[e[0]][e[1]]['weight'] for e in G.edges()])
        S = set()
        num = 0
        denom = 0
        best_score = 10000000
        best_index = 0
        for i in range(len(vertices) - 1):
            v = vertices[i]
            S.add(v)
            cut_edges = 0
            neighbor_weight = 0
            for w in G.neighbors(v):
                neighbor_weight += G[v][w]['weight']
                if w in S:
                    cut_edges += G[v][w]['weight']
            denom += neighbor_weight
            num += neighbor_weight - 2*cut_edges
            if num == 0:
                best_index = i
                break
            if num/min(denom, total_weight-denom) < best_score:
                best_score = num/min(denom, total_weight-denom)
                best_index = i
        if display:
            print("number of samples = ", len(v2))
            print("lambda2 = ", diag[0][1])
            plt.hist(v2, density=True, bins=30)
            plt.hist([x[v] for v in vertices[:best_index+1]], density=True, bins=30)
            plt.show()
        if return_eig:
            return vertices[:best_index+1], diag
        return vertices[:best_index+1]

def spectral_improve_cut(S, G, display=False):
    delta_n = {}
    delta_d = {}
    ip = {}
    new_S = set(S)
    total_weight = 2*sum([G[e[0]][e[1]]['weight'] for e in G.edges()])
    num =  sum([G[e[0]][e[1]]['weight'] for e in G.edges() if cut(e[0], e[1], new_S)])
    denom = sum([sum([G[u][v]['weight'] for v in G.neighbors(u)]) for u in new_S])
    if num == 0:
        return list(new_S)
    curr_score = num/min(denom, total_weight-denom)
    
    def set_ip(u):
        if min(denom + delta_d[u], total_weight - denom - delta_d[u]) == 0:
            ip[u] = 1000
        else:
            ip[u] = (num + delta_n[u])/min(denom + delta_d[u], total_weight - denom - delta_d[u]) - num/min(denom, total_weight - denom)
    
    for u in G.nodes():
        d = sum([G[u][v]['weight'] for v in G.neighbors(u)])
        if d == 0:
            return [u]
        c = sum([G[u][v]['weight'] for v in G.neighbors(u) if cut(u,v,new_S)])
        delta_n[u] = d-2*c
        if u in new_S:
            delta_d[u] = -d
        else:
            delta_d[u] = d
        set_ip(u)
    #TODO
    all_neg = False
    iters = 0
    
    while (not all_neg) and (iters < len(G.nodes)):
        best_potential = 0
        best_index = None
        for v in G.nodes():
            if ip[v] < best_potential:
                best_potential = ip[v]
                best_index = v
        if not best_index is None:
            num += delta_n[best_index]
            denom += delta_d[best_index]
            for j in G.neighbors(best_index):
                if cut(best_index,j,new_S):
                    delta_n[j] += 2*G[best_index][j]['weight']
                else:
                    delta_n[j] -= 2*G[best_index][j]['weight']
                set_ip(j)
            delta_n[best_index] = -delta_n[best_index]
            delta_d[best_index] = -delta_d[best_index]
            set_ip(best_index)
            if best_index in new_S:
                new_S.remove(best_index)
            else:
                new_S.add(best_index)
            #print("curr scores:", num/min(denom, total_weight - denom))
        else:
            all_neg = True
        iters += 1
    if display:
        print("sgreed+ score, ",  num/min(denom, total_weight - denom))
    return list(new_S)

def evaluate_sparsity(S, G):
    total_weight = 2*sum([G[e[0]][e[1]]['weight'] for e in G.edges()])
    num =  sum([G[e[0]][e[1]]['weight'] for e in G.edges() if cut(e[0], e[1], S)])
    denom = sum([sum([G[u][v]['weight'] for v in G.neighbors(u)]) for u in S])
    return num/min(denom, total_weight - denom)
    
def build_tree_sep(samples, method='greedy', subset = None):
    if not subset:
        subset = list(range(len(samples)))
    else:
        subset = list(subset)
    subset = remove_duplicates(samples, subset)
    T = nx.DiGraph()
    for i in subset:
        T.add_node(i)
    def build_helper(S):
        assert S, "error, S = "+ str(S)
        if len(S) == 1:
            return list(S)[0]
        left_set = set()
        if method == 'greedy':
            left_set = greedy_cut(samples, subset=S)
        elif method == 'SDP':
            G = construct_connectivity_graph(samples, subset=S)
            left_set = max_cut_heuristic(G, 3, 50)
        elif method == 'greedy+':
            G = construct_connectivity_graph(samples, subset=S)
            left_set = greedy_cut(samples, subset=S)
            left_set = improve_cut(G,left_set)

        if len(left_set) == 0 or len(left_set) == len(S):
            left_set = greedy_cut(samples, subset=S)
        right_set = set()
        for i in S:
            if not i in left_set:
                right_set.add(i)
        root = len(T.nodes) - len(subset) + len(samples)
        T.add_node(root)
        left_child = build_helper(left_set)
        right_child = build_helper(right_set)
        T.add_edge(root, left_child)
        T.add_edge(root, right_child)
        return root
    build_helper(subset)
    return T

def triplets_correct_sep(T, Tt, sample_size=5000):
    TC = 0
    sample_set = np.array([v for v in T.nodes() if T.in_degree(v) == 1 and T.out_degree(v) == 0])
    for a in range(sample_size):
        chosen = np.random.choice(sample_set, 3, replace=False)
        if outgroup2(chosen[0], chosen[1], chosen[2], T)[0] == outgroup2(chosen[0], chosen[1], chosen[2], Tt)[0]:
            TC += 1
    return TC/sample_size

def outgroup2(i, j, k, T):
    assert i != j and i != k and j != k, str(i) + ' ' + str(j) + ' ' + str(k) + ' not distinct'
    
#     Li = find_tree_lineage(i, T)
#     Lj = find_tree_lineage(j, T)
#     Lk = find_tree_lineage(k, T)

    Li = [node for node in nx.ancestors(T, i)]
    Lj = [node for node in nx.ancestors(T, j)]
    Lk = [node for node in nx.ancestors(T, k)]
    
    ij_common = len(set(Li) & set(Lj))
    ik_common = len(set(Li) & set(Lk))
    jk_common = len(set(Lj) & set(Lk))
    index = min(ij_common, ik_common, jk_common)

    if ij_common == ik_common and ik_common == jk_common:
        return None, index
    if ij_common > ik_common and ij_common > jk_common:
        return k, index
    elif jk_common > ik_common and jk_common > ij_common:
        return i, index
    elif ik_common > ij_common and ik_common > jk_common:
        return j, index

def triplets_correct_stratified(T, Tt, sample_size=5000, min_size_depth = 20):
    correct_class = defaultdict(int)
    freqs = defaultdict(int)
    sample_set = np.array([v for v in T.nodes() if T.in_degree(v) == 1 and T.out_degree(v) == 0])
    
    for a in range(sample_size):
        chosen = np.random.choice(sample_set, 3, replace=False)
        out1, index = outgroup2(chosen[0], chosen[1], chosen[2], T)
        out2, index2 = outgroup2(chosen[0], chosen[1], chosen[2], Tt)
        correct_class[index] += (out1 == out2)
        freqs[index] += 1
        
    tot_tp = 0
    num_consid = 0
    
    for k in correct_class.keys():
        if freqs[k] > min_size_depth:

            num_consid += 1
            tot_tp += correct_class[k] / freqs[k]

    tot_tp /= num_consid
    return tot_tp

def get_colless(network):
    root = [n for n in network if network.in_degree(n) == 0][0]
    colless = [0]
    colless_helper(network, root, colless)
    n = len([n for n in network if network.out_degree(n) == 0 and network.in_degree(n) == 1]) 
    return colless[0], (colless[0] - n * np.log(n) - n * (np.euler_gamma - 1 - np.log(2)))/n

def colless_helper(network, node, colless):
    if network.out_degree(node) == 0:
        return 1
    else:
        leaves = []
        for i in network.successors(node):
            leaves.append(colless_helper(network, i, colless))
        colless[0] += abs(leaves[0] - leaves[1])
        return sum(leaves)

def triplets_correct_at_time_sep(T, Tt, method='all', bin_size = 10, sample_size=5000, sampling_depths=None):
    sample_set = set([v for v in T.nodes() if T.in_degree(v) == 1 and T.out_degree(v) == 0])
    children = {}
    num_triplets = {}
    nodes_at_depth = {}

    def find_children(node, total_time):
        t = total_time + Tt.nodes[node]['parent_lifespan']
        children[node] = []
        if Tt.out_degree(node) == 0:
            if node in sample_set:
                children[node].append(node)
            return

        for n in Tt.neighbors(node):
            find_children(n, t)
            children[node] += children[n]

        L, R = list(Tt.neighbors(node))[0], list(Tt.neighbors(node))[1]
        num_triplets[node] = len(children[L])*nCr(len(children[R]), 2) + len(children[R])*nCr(len(children[L]), 2)
        if num_triplets[node] > 0:
            bin_num = t//bin_size
            
            if bin_num in nodes_at_depth:
                nodes_at_depth[bin_num].append(node)
            else:
                nodes_at_depth[bin_num] = [node]
                
    root = [n for n in Tt if Tt.in_degree(n) == 0][0]
    find_children(root, 0)

    def sample_at_depth(d):
        denom = sum([num_triplets[v] for v in nodes_at_depth[d]])
        node = np.random.choice(nodes_at_depth[d], 1, [num_triplets[v]/denom for v in nodes_at_depth[d]])[0]
        L, R = list(Tt.neighbors(node))[0], list(Tt.neighbors(node))[1]
        if np.random.random() < (len(children[R])-1)/(len(children[R])+len(children[L])-2):
            outgrp = np.random.choice(children[L], 1)[0]
            ingrp = np.random.choice(children[R], 2, replace=False)
        else:
            outgrp = np.random.choice(children[R], 1)[0]
            ingrp = np.random.choice(children[L], 2, replace=False)
        return outgroup(ingrp[0], ingrp[1], outgrp, T) == outgrp

    if not sampling_depths:
        sampling_depths = [d for d in range(len(nodes_at_depth))]
    if method == 'aggregate':
        score = 0
        freq = 0
        for d in sampling_depths:
            if d in nodes_at_depth:
                max_children = 0
                for i in nodes_at_depth[d]:
                    if len(children[i]) > max_children:
                        max_children = len(children[i])
                if max_children > 10:
                    freq += 1
                    for a in range(sample_size):
                        score += int(sample_at_depth(d))
        return score/(sample_size*freq)
    elif method == 'all':
        ret = ['NA'] * len(sampling_depths)
        for d in sampling_depths:
            if d in nodes_at_depth:
                max_children = 0
                for i in nodes_at_depth[d]:
                    if len(children[i]) > max_children:
                        max_children = len(children[i])
                if max_children > 10:
                    score = 0
                    for a in range(sample_size):
                        score += int(sample_at_depth(d))
                    ret[d] = score/sample_size
        return np.array(ret)

def triplets_correct_at_depth_sep(T, Tt, method='all', sample_size=5000, sampling_depths=None):
    sample_set = set([v for v in T.nodes() if T.in_degree(v) == 1 and T.out_degree(v) == 0])
    children = {}
    num_triplets = {}
    nodes_at_depth = {}

    def find_children(node, depth):
        children[node] = []
        if Tt.out_degree(node) == 0:
            if node in sample_set:
                children[node].append(node)
            return

        for n in Tt.neighbors(node):
            find_children(n, depth+1)
            children[node] += children[n]

        L, R = list(Tt.neighbors(node))[0], list(Tt.neighbors(node))[1]
        num_triplets[node] = len(children[L])*nCr(len(children[R]), 2) + len(children[R])*nCr(len(children[L]), 2)
        if num_triplets[node] > 0:
            if depth in nodes_at_depth:
                nodes_at_depth[depth].append(node)
            else:
                nodes_at_depth[depth] = [node]
                
    root = [n for n in Tt if Tt.in_degree(n) == 0][0]
    find_children(root, 0)

    def sample_at_depth(d):
        denom = sum([num_triplets[v] for v in nodes_at_depth[d]])
        node = np.random.choice(nodes_at_depth[d], 1, [num_triplets[v]/denom for v in nodes_at_depth[d]])[0]
        L, R = list(Tt.neighbors(node))[0], list(Tt.neighbors(node))[1]
        if np.random.random() < (len(children[R])-1)/(len(children[R])+len(children[L])-2):
            outgrp = np.random.choice(children[L], 1)[0]
            ingrp = np.random.choice(children[R], 2, replace=False)
        else:
            outgrp = np.random.choice(children[R], 1)[0]
            ingrp = np.random.choice(children[L], 2, replace=False)
        return outgroup(ingrp[0], ingrp[1], outgrp, T) == outgrp

    if not sampling_depths:
        sampling_depths = [d for d in range(len(nodes_at_depth))]
        
    if method == 'aggregate':
        score = 0
        freq = 0
        for d in sampling_depths:
            if d in nodes_at_depth:
                max_children = 0
                for i in nodes_at_depth[d]:
                    if len(children[i]) > max_children:
                        max_children = len(children[i])
                if max_children > 10:
                    freq += 1
                    for a in range(sample_size):
                        score += int(sample_at_depth(d))
        return score/(sample_size*freq)
    elif method == 'all':
        ret = ['NA'] * len(sampling_depths)
        for d in sampling_depths:
            if d in nodes_at_depth:
                max_children = 0
                for i in nodes_at_depth[d]:
                    if len(children[i]) > max_children:
                        max_children = len(children[i])
                if max_children > 10:
                    score = 0
                    for a in range(sample_size):
                        score += int(sample_at_depth(d))
                    ret[d] = score/sample_size
        return np.array(ret) 