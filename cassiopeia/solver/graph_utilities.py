import networkx as nx
import itertools

from cassiopeia.solver import solver_utilities as utils


def check_if_cut(u, v, S):
    return ((u in S) and (not v in S)) or ((v in S) and (not u in S))


def max_cut_improve_cut(G, S):
    ip = {}
    new_S = S.copy()
    for i in G.nodes():
        improvement_potential = 0
        for j in G.neighbors(i):
            if check_if_cut(i, j, new_S):
                improvement_potential -= G[i][j]["weight"]
            else:
                improvement_potential += G[i][j]["weight"]
        ip[i] = improvement_potential

    all_neg = False
    iters = 0
    while (not all_neg) and (iters < 2 * len(G.nodes)):
        best_potential = 0
        best_index = 0
        for i in G.nodes():
            if ip[i] > best_potential:
                best_potential = ip[i]
                best_index = i
        if best_potential > 0:
            for j in G.neighbors(best_index):
                if check_if_cut(best_index, j, new_S):
                    ip[j] += 2 * G[best_index][j]["weight"]
                else:
                    ip[j] -= 2 * G[best_index][j]["weight"]
            ip[best_index] = -ip[best_index]
            if best_index in new_S:
                new_S.remove(best_index)
            else:
                new_S.add(best_index)
        else:
            all_neg = True
        iters += 1
    # print("number of hill climbing interations: ", iters)
    return new_S


def construct_connectivity_graph(cm, missing_char, samples=None, w=None):
    G = nx.Graph()
    if not samples:
        samples = range(cm.shape[0])
    for i in samples:
        G.add_node(i)
    F = utils.compute_mutation_frequencies(cm, missing_char, samples)
    for i, j in itertools.combinations(samples, 2):
        # compute simularity scores
        score = 0
        for l in range(cm.shape[1]):
            x = cm.iloc[i, l]
            y = cm.iloc[j, l]
            if (x != missing_char and y != missing_char) and (x != "0" or y != "0"):
                if w is not None:
                    if x == y:
                        score -= (
                            3 * w[l][x] * (len(samples) - F[l][x] - F[l][missing_char])
                        )
                    elif x == "0" or y == "0":
                        score += w[l][max(x, y)] * (F[l][max(x, y)] - 1)
                    else:
                        score += w[l][x] * (F[l][x] - 1) + w[l][y] * (F[l][y] - 1)
                else:
                    if x == y:
                        score -= 3 * (len(samples) - F[l][x] - F[l][missing_char])
                    elif x == "0" or y == "0":
                        score += F[l][max(x, y)] - 1
                    else:
                        score += F[l][x] + F[l][y] - 2

            if score != 0:
                G.add_edge(i, j, weight=score)
    return G


def construct_similarity_graph(cm, missing_char, samples=None, threshold=0, w=None):
    G = nx.Graph()
    if not samples:
        samples = range(cm.shape[0])
    for i in samples:
        G.add_node(i)
    F = utils.compute_mutation_frequencies(cm, missing_char, samples)
    for i in F:
        for j in F[i]:
            if j != "0" and j != missing_char:
                if F[i][j] == len(samples) - F[i][missing_char]:
                    threshold += 1
    for i, j in itertools.combinations(samples, 2):
        s = similarity(i, j, cm, missing_char, w)
        if s > threshold:
            G.add_edge(i, j, weight=(s - threshold))
    return G


def spectral_improve_cut(S, G, display=False):
    delta_n = {}
    delta_d = {}
    ip = {}
    new_S = set(S)
    total_weight = 2 * sum([G[e[0]][e[1]]["weight"] for e in G.edges()])
    num = sum(
        [G[e[0]][e[1]]["weight"] for e in G.edges() if check_if_cut(e[0], e[1], new_S)]
    )
    denom = sum([sum([G[u][v]["weight"] for v in G.neighbors(u)]) for u in new_S])
    if num == 0:
        return list(new_S)
    # curr_score = num/min(denom, total_weight-denom)

    def set_ip(u):
        if min(denom + delta_d[u], total_weight - denom - delta_d[u]) == 0:
            ip[u] = 1000
        else:
            ip[u] = (num + delta_n[u]) / min(
                denom + delta_d[u], total_weight - denom - delta_d[u]
            ) - num / min(denom, total_weight - denom)

    for u in G.nodes():
        d = sum([G[u][v]["weight"] for v in G.neighbors(u)])
        if d == 0:
            return [u]
        c = sum(
            [G[u][v]["weight"] for v in G.neighbors(u) if check_if_cut(u, v, new_S)]
        )
        delta_n[u] = d - 2 * c
        if u in new_S:
            delta_d[u] = -d
        else:
            delta_d[u] = d
        set_ip(u)

    # TODO
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
                if check_if_cut(best_index, j, new_S):
                    delta_n[j] += 2 * G[best_index][j]["weight"]
                else:
                    delta_n[j] -= 2 * G[best_index][j]["weight"]
                set_ip(j)
            delta_n[best_index] = -delta_n[best_index]
            delta_d[best_index] = -delta_d[best_index]
            set_ip(best_index)
            if best_index in new_S:
                new_S.remove(best_index)
            else:
                new_S.add(best_index)
            # print("curr scores:", num/min(denom, total_weight - denom))
        else:
            all_neg = True
        iters += 1
    if display:
        print("sgreed+ score, ", num / min(denom, total_weight - denom))
    return list(new_S)


def similarity(u, v, cm, missing_char, w=None):
    # TODO Optimize this using masks
    k = cm.shape[0]
    if w is None:
        return sum(
            [
                1
                for i in range(k)
                if cm.iloc[u, i] == cm.iloc[v, i]
                and (cm.iloc[u, i] != "0" and cm.iloc[u, i] != missing_char)
            ]
        )
    else:
        return sum(
            [
                w[i][cm.iloc[u, i]]
                for i in range(k)
                if cm.iloc[u, i] == cm.iloc[v, i]
                and (cm.iloc[u, i] != "0" and cm.iloc[u, i] != missing_char)
            ]
        )
