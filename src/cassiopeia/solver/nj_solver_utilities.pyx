# nj_solver_utilities.pyx
# Dynamic Neighbor Joining (DNJ) and UPGMA via cached per-row minima.
# DNJ algorithm adapted from CCPhylo (P.T.L.C. Clausen, Apache 2.0, 2021).
# https://bitbucket.org/genomicepidemiology/ccphylo
#
# Attribution: Philip T.L.C. Clausen, Technical University of Denmark

import numpy as np
cimport numpy as np
cimport cython
from libc.float cimport DBL_MAX

np.import_array()


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def dnj(double[:, ::1] D_in):
    """Dynamic Neighbor-Joining with O(n²) per-row Q-cache.

    Adapted from CCPhylo dnj.c (Clausen 2021, Apache 2.0).  Each row i
    caches Q_min[i] = min_{j<i} Q(i,j).  Rows are skipped in the search
    when their cached minimum cannot beat the current best; otherwise the
    row is rescanned to get the exact value.

    Args:
        D_in: n×n float64 distance matrix, symmetric, zero diagonal.

    Returns:
        Tuple (parents, children, lengths, node_a, node_b):
          parents/children: int32 arrays of shape (2*(n-2),)
          lengths:          float64 array  of shape (2*(n-2),)
          node_a, node_b:   int IDs of the last two unjoined nodes.
        Leaf IDs are 0..n-1; internal node IDs are n..2n-2.
        The caller adds the final edge (node_a, node_b) when rooting.
    """
    cdef int n = D_in.shape[0]

    if n < 2:
        raise ValueError("dnj requires at least 2 taxa")

    if n == 2:
        return (
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.float64),
            0, 1,
        )

    # Working distance matrix (full symmetric)
    D = np.array(D_in, dtype=np.float64, copy=True)
    cdef double[:, ::1] d = D

    # Row sums over currently active nodes (diagonal = 0, included harmlessly)
    sD_arr = np.zeros(n, dtype=np.float64)
    cdef double[:] sd = sD_arr

    # Per-row cached Q minimum over lower triangle (jj < ii in active array)
    Q_cache = np.empty(n, dtype=np.float64)
    cdef double[:] qc = Q_cache

    # Cached argmin partner index (jj < ii) for each row ii
    P_cache = np.zeros(n, dtype=np.int32)
    cdef int[:] pc = P_cache

    # Active node indices: act[0:m] are currently active matrix rows/cols
    act_arr = np.arange(n, dtype=np.int32)
    cdef int[:] act = act_arr

    # Abstract tree node IDs: leaves 0..n-1, internal n..2n-2
    nid_arr = np.arange(n, dtype=np.int32)
    cdef int[:] nid = nid_arr

    # Output edge arrays (n-2 merges × 2 child edges)
    cdef int ne = 2 * (n - 2)
    par_arr = np.empty(ne, dtype=np.int32)
    chi_arr = np.empty(ne, dtype=np.int32)
    len_arr = np.empty(ne, dtype=np.float64)
    cdef int[:] par = par_arr
    cdef int[:] chi = chi_arr
    cdef double[:] lens = len_arr

    cdef int m = n
    cdef int ii, jj, k
    cdef int ai, aj, ak, bi, bj
    cdef int p_best, internal_id, merge_count
    cdef double q, q_best, min_q_i, Li, Lj, dij
    cdef double old_d, new_d, sd_j_new, fm2, fm2_new
    cdef int ai_moved

    # ── Initialise sD ────────────────────────────────────────────────────────
    for ii in range(m):
        ai = act[ii]
        s = 0.0
        for jj in range(m):
            s += d[ai, act[jj]]
        sd[ii] = s   # d[ai, ai] = 0, so this is the row sum over all active j ≠ ii

    # ── Initialise Q_cache (lower triangle: jj < ii) ─────────────────────────
    fm2 = <double>(m - 2)
    qc[0] = DBL_MAX
    pc[0] = 0
    for ii in range(1, m):
        qc[ii] = DBL_MAX
        pc[ii] = 0
        ai = act[ii]
        for jj in range(ii):
            q = fm2 * d[ai, act[jj]] - sd[ii] - sd[jj]
            if q < qc[ii]:
                qc[ii] = q
                pc[ii] = jj

    # ── Main loop ────────────────────────────────────────────────────────────
    internal_id = n
    merge_count = 0

    while m > 2:
        fm2 = <double>(m - 2)

        # Find best pair: scan rows, rescan only when cache < current best
        q_best = DBL_MAX
        bi = 1
        bj = 0
        for ii in range(1, m):
            if qc[ii] < q_best:
                ai = act[ii]
                min_q_i = DBL_MAX
                p_best = 0
                for jj in range(ii):
                    q = fm2 * d[ai, act[jj]] - sd[ii] - sd[jj]
                    if q < min_q_i:
                        min_q_i = q
                        p_best = jj
                qc[ii] = min_q_i
                pc[ii] = p_best
                if min_q_i < q_best:
                    q_best = min_q_i
                    bi = ii
                    bj = p_best

        # bi > bj guaranteed by lower-triangle construction
        ai = act[bi]
        aj = act[bj]
        dij = d[ai, aj]

        # Branch lengths (standard NJ)
        Li = 0.5 * (dij + (sd[bi] - sd[bj]) / fm2)
        Lj = dij - Li
        if Li < 0.0:
            Li = 0.0
            Lj = dij
        elif Lj < 0.0:
            Lj = 0.0
            Li = dij

        # Record two child edges to new internal node
        par[2 * merge_count]     = internal_id
        chi[2 * merge_count]     = nid[bi]
        lens[2 * merge_count]    = Li
        par[2 * merge_count + 1] = internal_id
        chi[2 * merge_count + 1] = nid[bj]
        lens[2 * merge_count + 1] = Lj

        # Update distances: new merged node stored at position bj (row/col aj)
        sd_j_new = 0.0
        for k in range(m):
            if k == bi or k == bj:
                continue
            ak = act[k]
            old_d = d[aj, ak]
            new_d = 0.5 * (d[ai, ak] + old_d - dij)
            if new_d < 0.0:
                new_d = 0.0
            d[aj, ak] = new_d
            d[ak, aj] = new_d
            sd[k] += new_d - old_d - d[ai, ak]
            sd_j_new += new_d
        d[aj, aj] = 0.0
        sd[bj] = sd_j_new

        # Multiplier for the next iteration (m will become m-1)
        fm2_new = <double>(m - 3)

        # Recompute Q_cache for the new merged node (at position bj)
        qc[bj] = DBL_MAX
        pc[bj] = 0
        for jj in range(bj):
            q = fm2_new * d[aj, act[jj]] - sd_j_new - sd[jj]
            if q < qc[bj]:
                qc[bj] = q
                pc[bj] = jj

        # Update Q_cache for rows k > bj: improve if new node bj is a better partner
        for k in range(bj + 1, m):
            if k == bi:
                continue
            q = fm2_new * d[act[k], aj] - sd[k] - sd_j_new
            if q < qc[k]:
                qc[k] = q
                pc[k] = bj

        # Mark merged node ID at bj
        nid[bj] = internal_id

        # Remove bi by swapping with the last active position (m-1)
        if bi != m - 1:
            ai_moved = act[m - 1]
            act[bi] = ai_moved
            sd[bi] = sd[m - 1]
            nid[bi] = nid[m - 1]

            # Fully recompute Q[bi] for the moved node (lower triangle: jj < bi)
            qc[bi] = DBL_MAX
            pc[bi] = 0
            for jj in range(bi):
                q = fm2_new * d[ai_moved, act[jj]] - sd[bi] - sd[jj]
                if q < qc[bi]:
                    qc[bi] = q
                    pc[bi] = jj

            # Update Q for rows k > bi: moved node may be a better partner
            for k in range(bi + 1, m - 1):
                q = fm2_new * d[act[k], ai_moved] - sd[k] - sd[bi]
                if q < qc[k]:
                    qc[k] = q
                    pc[k] = bi

        m -= 1
        merge_count += 1
        internal_id += 1

    return (par_arr, chi_arr, len_arr, int(nid[0]), int(nid[1]))


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def upgma(double[:, ::1] D_in, long[:] cluster_sizes_in=None):
    """UPGMA with O(n²) per-row minimum-distance cache.

    Args:
        D_in: n×n float64 distance matrix, symmetric, zero diagonal.
        cluster_sizes_in: optional int64 array of initial cluster sizes (default 1).

    Returns:
        Same (parents, children, lengths, node_a, node_b) as dnj().
    """
    cdef int n = D_in.shape[0]

    if n < 2:
        raise ValueError("upgma requires at least 2 taxa")

    if n == 2:
        return (
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.float64),
            0, 1,
        )

    D = np.array(D_in, dtype=np.float64, copy=True)
    cdef double[:, ::1] d = D

    # Cluster sizes (UPGMA weighted average)
    if cluster_sizes_in is None:
        sz_arr = np.ones(n, dtype=np.int64)
    else:
        sz_arr = np.array(cluster_sizes_in, dtype=np.int64, copy=True)
    cdef long[:] sz = sz_arr

    # Per-row minimum distance cache (lower triangle: jj < ii)
    Dmin_cache = np.empty(n, dtype=np.float64)
    cdef double[:] dmc = Dmin_cache

    P_cache = np.zeros(n, dtype=np.int32)
    cdef int[:] pc = P_cache

    act_arr = np.arange(n, dtype=np.int32)
    cdef int[:] act = act_arr

    nid_arr = np.arange(n, dtype=np.int32)
    cdef int[:] nid = nid_arr

    cdef int ne = 2 * (n - 2)
    par_arr = np.empty(ne, dtype=np.int32)
    chi_arr = np.empty(ne, dtype=np.int32)
    len_arr = np.empty(ne, dtype=np.float64)
    cdef int[:] par = par_arr
    cdef int[:] chi = chi_arr
    cdef double[:] lens = len_arr

    cdef int m = n
    cdef int ii, jj, k
    cdef int ai, aj, ak, bi, bj
    cdef int p_best, internal_id, merge_count
    cdef double d_best, min_d_i, Li, Lj, dij
    cdef double old_d, new_d
    cdef long si, sj
    cdef int ai_moved

    # Initialise Dmin_cache (lower triangle)
    dmc[0] = DBL_MAX
    pc[0] = 0
    for ii in range(1, n):
        dmc[ii] = DBL_MAX
        pc[ii] = 0
        ai = act[ii]
        for jj in range(ii):
            if d[ai, act[jj]] < dmc[ii]:
                dmc[ii] = d[ai, act[jj]]
                pc[ii] = jj

    internal_id = n
    merge_count = 0

    while m > 2:
        # Find closest pair
        d_best = DBL_MAX
        bi = 1
        bj = 0
        for ii in range(1, m):
            if dmc[ii] < d_best:
                ai = act[ii]
                min_d_i = DBL_MAX
                p_best = 0
                for jj in range(ii):
                    if d[ai, act[jj]] < min_d_i:
                        min_d_i = d[ai, act[jj]]
                        p_best = jj
                dmc[ii] = min_d_i
                pc[ii] = p_best
                if min_d_i < d_best:
                    d_best = min_d_i
                    bi = ii
                    bj = p_best

        ai = act[bi]
        aj = act[bj]
        dij = d[ai, aj]
        si = sz[bi]
        sj = sz[bj]

        # UPGMA: each half-distance is the merge height
        Li = dij * 0.5
        Lj = dij * 0.5

        par[2 * merge_count]     = internal_id
        chi[2 * merge_count]     = nid[bi]
        lens[2 * merge_count]    = Li
        par[2 * merge_count + 1] = internal_id
        chi[2 * merge_count + 1] = nid[bj]
        lens[2 * merge_count + 1] = Lj

        # Update distances: weighted average
        for k in range(m):
            if k == bi or k == bj:
                continue
            ak = act[k]
            new_d = (si * d[ai, ak] + sj * d[aj, ak]) / <double>(si + sj)
            d[aj, ak] = new_d
            d[ak, aj] = new_d
        d[aj, aj] = 0.0
        sz[bj] = si + sj

        # Recompute cache for merged node at bj
        dmc[bj] = DBL_MAX
        pc[bj] = 0
        for jj in range(bj):
            if d[aj, act[jj]] < dmc[bj]:
                dmc[bj] = d[aj, act[jj]]
                pc[bj] = jj

        # Update cache for rows k > bj
        for k in range(bj + 1, m):
            if k == bi:
                continue
            if d[act[k], aj] < dmc[k]:
                dmc[k] = d[act[k], aj]
                pc[k] = bj

        nid[bj] = internal_id

        if bi != m - 1:
            ai_moved = act[m - 1]
            act[bi] = ai_moved
            sz[bi] = sz[m - 1]
            nid[bi] = nid[m - 1]

            dmc[bi] = DBL_MAX
            pc[bi] = 0
            for jj in range(bi):
                if d[ai_moved, act[jj]] < dmc[bi]:
                    dmc[bi] = d[ai_moved, act[jj]]
                    pc[bi] = jj

            for k in range(bi + 1, m - 1):
                if d[act[k], ai_moved] < dmc[k]:
                    dmc[k] = d[act[k], ai_moved]
                    pc[k] = bi

        m -= 1
        merge_count += 1
        internal_id += 1

    return (par_arr, chi_arr, len_arr, int(nid[0]), int(nid[1]))
