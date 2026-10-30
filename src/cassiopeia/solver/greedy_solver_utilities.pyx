# greedy_solver_utilities.pyx
# Fast single-split kernel for vanilla Cassiopeia-Greedy.
#
# Operates on an integer-encoded character matrix (unmodified -> 0, missing ->
# missing_state, mutated -> positive ints).  Replicates the behaviour of the
# pure-Python ``_greedy_split`` + ``_assign_missing_average`` (greedy.py) so the
# reconstructed topology is identical, but moves the per-node frequency counting,
# best-mutation selection, partitioning, and "average" missing-data imputation
# into typed C loops.
#
# Ambiguous states are not handled here: after ``_encode_integer_matrix`` every
# cell is an atomic int, so the greedy/hybrid code paths never see ambiguous
# states.  Custom (non-"average") missing-data classifiers fall back to Python.

import numpy as np

cimport cython
cimport numpy as cnp

cnp.import_array()

ctypedef cnp.int64_t INT


@cython.boundscheck(False)
@cython.wraparound(False)
def perform_split(
    INT[:, ::1] cm,
    INT[::1] sample_idx,
    double[:, ::1] weights,
    int has_weights,
    INT missing_state,
    int nstates,
):
    """Partition ``sample_idx`` on the most frequent (character, state) pair.

    Args:
        cm: Integer character matrix (n_rows x n_chars), C-contiguous.
        sample_idx: Row indices into ``cm`` for the current subproblem.
        weights: (n_chars x nstates) float weight per (character, state); only
            read when ``has_weights`` is non-zero.
        has_weights: Whether to weight frequencies by ``weights``.
        missing_state: Integer code for missing data (negative by convention).
        nstates: One past the largest state value present in ``cm`` (so valid
            mutated states lie in ``1 .. nstates-1``).

    Returns:
        ``(left_idx, right_idx)``: int64 arrays of row indices into ``cm``.
        ``right_idx`` is empty when no informative split exists.
    """
    cdef Py_ssize_t m = sample_idx.shape[0]
    cdef Py_ssize_t nc = cm.shape[1]
    cdef Py_ssize_t i, c, st
    cdef INT s, v, q
    cdef Py_ssize_t nl = 0, nr = 0, nmiss = 0
    cdef double best_score = 0.0, score
    cdef Py_ssize_t chosen_char = 0, chosen_state = 0
    cdef double left_score, right_score, w, cur_nl, cur_nr
    cdef int assign_left
    cdef INT[:, ::1] lc, rc
    cdef INT[::1] lb, rb, mb

    # ── Mutation frequencies over the subset ─────────────────────────────────
    freq_arr = np.zeros((nc, nstates), dtype=np.int64)
    miss_arr = np.zeros(nc, dtype=np.int64)
    cdef INT[:, ::1] freq = freq_arr
    cdef INT[::1] miss = miss_arr

    for i in range(m):
        s = sample_idx[i]
        for c in range(nc):
            v = cm[s, c]
            if v == missing_state:
                miss[c] += 1
            elif v > 0:
                freq[c, v] += 1

    # ── Select the best (character, state) by (weighted) frequency ───────────
    for c in range(nc):
        for st in range(1, nstates):
            if freq[c, st] > 0 and freq[c, st] < m - miss[c]:
                if has_weights:
                    score = freq[c, st] * weights[c, st]
                else:
                    score = <double>freq[c, st]
                if score > best_score:
                    best_score = score
                    chosen_char = c
                    chosen_state = st

    # No informative split: everything goes left (polytomy upstream).
    if chosen_state == 0:
        return np.array(sample_idx, dtype=np.int64), np.empty(0, dtype=np.int64)

    # ── Partition on the chosen mutation ─────────────────────────────────────
    left_buf = np.empty(m, dtype=np.int64)
    right_buf = np.empty(m, dtype=np.int64)
    miss_buf = np.empty(m, dtype=np.int64)
    lb = left_buf
    rb = right_buf
    mb = miss_buf

    for i in range(m):
        s = sample_idx[i]
        v = cm[s, chosen_char]
        if v == chosen_state:
            lb[nl] = s
            nl += 1
        elif v == missing_state:
            mb[nmiss] = s
            nmiss += 1
        else:
            rb[nr] = s
            nr += 1

    # ── "Average" imputation of samples missing at the chosen character ──────
    if nmiss > 0:
        # Per-side counts of mutated states (built from the original members
        # only, matching the Python implementation's precomputed side states).
        lc_arr = np.zeros((nc, nstates), dtype=np.int64)
        rc_arr = np.zeros((nc, nstates), dtype=np.int64)
        lc = lc_arr
        rc = rc_arr

        for i in range(nl):
            s = lb[i]
            for c in range(nc):
                v = cm[s, c]
                if v > 0:
                    lc[c, v] += 1
        for i in range(nr):
            s = rb[i]
            for c in range(nc):
                v = cm[s, c]
                if v > 0:
                    rc[c, v] += 1

        # Running denominators grow as samples are assigned (matches Python,
        # which divides by the current len(left_set)/len(right_set)).
        cur_nl = <double>nl
        cur_nr = <double>nr

        for i in range(nmiss):
            s = mb[i]
            left_score = 0.0
            right_score = 0.0
            for c in range(nc):
                q = cm[s, c]
                if q > 0:
                    if has_weights:
                        w = weights[c, q]
                        left_score += w * lc[c, q]
                        right_score += w * rc[c, q]
                    else:
                        left_score += lc[c, q]
                        right_score += rc[c, q]
            if cur_nr == 0.0:
                assign_left = 1
            elif cur_nl == 0.0:
                assign_left = 0
            elif (left_score / cur_nl) > (right_score / cur_nr):
                assign_left = 1
            else:
                assign_left = 0
            if assign_left:
                lb[nl] = s
                nl += 1
                cur_nl += 1.0
            else:
                rb[nr] = s
                nr += 1
                cur_nr += 1.0

    return left_buf[:nl].copy(), right_buf[:nr].copy()
