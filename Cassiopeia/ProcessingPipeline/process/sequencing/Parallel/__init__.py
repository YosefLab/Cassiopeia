import itertools
import numpy as np

def get_bounds(length, num_pieces):
    base_size = length // num_pieces
    num_with_extra = length - base_size * num_pieces
    sizes = np.ones(num_pieces, int) * base_size
    sizes[:num_with_extra] += 1
    bounds = np.append([0], sizes.cumsum())
    return bounds

def piece_of_list(full_list, num_pieces, which_piece, interleaved=False):
    if which_piece == -1:
        # Sentinel value that means 'the whole thing'.
        piece = full_list
    elif interleaved:
        piece = [x for i, x in enumerate(full_list) if i % num_pieces == which_piece]
    else:
        bounds = get_bounds(len(full_list), num_pieces)
        piece = full_list[bounds[which_piece]:bounds[which_piece + 1]]

    return piece

def piece_of_iter(full_iter, num_pieces, which_piece):
    if which_piece == -1:
        # Sentinel value that means 'the whole thing'.
        piece = full_iter
    else:
        piece = itertools.islice(full_iter, which_piece, None, num_pieces)

    return piece
