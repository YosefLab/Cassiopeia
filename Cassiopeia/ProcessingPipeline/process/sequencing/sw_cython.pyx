import numpy as np
cimport cython

cdef int SOFT_CLIPPED_typed = -2
SOFT_CLIPPED = SOFT_CLIPPED_typed
cdef int GAP_typed = -1
GAP = GAP_typed

@cython.boundscheck(False)
def generate_matrices(char* query,
                      char* target,
                      int match_bonus,
                      int mismatch_penalty,
                      int indel_penalty,
                      force_query_start,
                      force_target_start,
                      force_either_start,
                     ):
    cdef unsigned int row, col, next_col, next_row
    cdef int match_or_mismatch, diagonal, from_left, from_above, new_score, unconstrained_start
    shape = (len(query) + 1, len(target) + 1)
    scores = np.zeros(shape, int)
    cdef long[:, ::1] scores_view = scores
    row_directions = np.zeros(shape, int)
    cdef long[:, ::1] row_directions_view = row_directions
    col_directions = np.zeros(shape, int)
    cdef long[:, ::1] col_directions_view = col_directions

    # If the alignment is constrained to include the start of the query,
    # indel penalties need to be applied to cells in the first row.
    if force_query_start: 
        for row in range(1, len(query) + 1):
            scores_view[row, 0] = scores_view[row - 1, 0] + indel_penalty
            row_directions_view[row, 0] = -1

    # If the alignment is constrained to include the start of the target,
    # indel penalties need to be applied to cells in the first column.
    if force_target_start:
        for col in range(1, len(target) + 1):
            scores_view[0, col] = scores_view[0, col - 1] + indel_penalty
            col_directions_view[0, col] = -1

    unconstrained_start = not (force_query_start or force_target_start or force_either_start)

    for row in range(1, len(query) + 1):
        for col in range(1, len(target) + 1):
            if query[row - 1] == 'N' or target[col - 1] == 'N':
                match_or_mismatch = match_bonus
            elif query[row - 1] == target[col - 1]:
                match_or_mismatch = match_bonus
            else:
                match_or_mismatch = mismatch_penalty
            diagonal = scores_view[row - 1, col - 1] + match_or_mismatch
            from_left = scores_view[row, col - 1] + indel_penalty
            from_above = scores_view[row - 1, col] + indel_penalty
            new_score = max(diagonal, from_left, from_above)
            if unconstrained_start:
                new_score = max(0, new_score)
            scores_view[row, col] = new_score
            if new_score > max_score:
                max_score = new_score
                max_row = row
                max_col = col
            if unconstrained_start and new_score == 0:
                pass
            elif new_score == diagonal:
                col_directions_view[row, col] = -1
                row_directions_view[row, col] = -1
            elif new_score == from_left:
                col_directions_view[row, col] = -1
            elif new_score == from_above:
                row_directions_view[row, col] = -1

    matrices = {'scores': scores,
                'row_directions': row_directions,
                'col_directions': col_directions,
               }
    return matrices

def backtrack_cython(char* query,
                     char* target,
                     matrices,
                     cells_seen,
                     int end_row,
                     int end_col,
                     int force_query_start,
                     int force_target_start,
                     int force_either_start,
                    ):
    cdef int row, col, next_row, next_col, target_index, query_index
    query_mappings = np.full(len(query), SOFT_CLIPPED_typed, int)
    cdef long [:] query_mappings_view = query_mappings
    target_mappings = np.full(len(target), SOFT_CLIPPED_typed, int)
    cdef long [:] target_mappings_view = target_mappings

    cdef long [:, :] col_directions = matrices['col_directions']
    cdef long [:, :] row_directions = matrices['row_directions']
    cdef long [:, :] scores = matrices['scores']

    path = []
    insertions = set()
    deletions = set()
    mismatches = set()
    
    unconstrained_start = not(force_query_start or force_target_start or force_either_start)

    row = end_row
    col = end_col

    if row == 0 or col == 0:
        # There are no query or target bases involved in a path that ends on
        # the top or the left edge.
        reached_end = True
    else:
        reached_end = False

    while not reached_end:
        if (row, col) in cells_seen:
            return None
        cells_seen.add((row, col))

        next_col = col + col_directions[row, col]
        next_row = row + row_directions[row, col]
        if next_col == col:
            target_index = GAP_typed
            insertions.add(row - 1)
        else:
            target_index = col - 1
        if next_row == row:
            query_index = GAP_typed
            deletions.add(col - 1)
        else:
            query_index = row - 1
        
        if target_index != GAP_typed:
            target_mappings_view[target_index] = query_index
        if query_index != GAP_typed:
            query_mappings_view[query_index] = target_index
        if target_index != GAP_typed and query_index != GAP_typed and query[query_index] != 'N' and target[target_index] != 'N' and query[query_index] != target[target_index]:
            mismatches.add((query_index, target_index))

        path.append((query_index, target_index))

        row = next_row
        col = next_col

        if unconstrained_start:
            if scores[row, col] <= 0:
                reached_end = True
        elif force_query_start and force_target_start:
            if row == 0 and col == 0:
                reached_end = True
        elif force_either_start:
            if row == 0 or col == 0:
                reached_end = True
        elif force_query_start:
            if row == 0:
                reached_end = True
        elif force_target_start:
            if col == 0:
                reached_end = True

    path = path[::-1]

    alignment = {'score': scores[end_row, end_col],
                 'path': path,
                 'query_mappings': query_mappings,
                 'target_mappings': target_mappings,
                 'insertions': insertions,
                 'deletions': deletions,
                 'mismatches': mismatches,
                 'XM': len(mismatches),
                 'XO': len(insertions) + len(deletions),
                }

    return alignment
