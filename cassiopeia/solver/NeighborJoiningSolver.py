"""
This file stores a subclass of DistanceSolver, NeighborJoining. The
inference procedure is the Neighbor-Joining algorithm proposed by Saitou and
Nei (1987) that iteratively joins together samples that minimize the Q-criterion
on the dissimilarity map.
"""
import abc
import numba
import numpy as np
import pandas as pd
from typing import Callable, Dict, Optional, Tuple, Union

from cassiopeia.solver import DistanceSolver


class NeighborJoiningSolver(DistanceSolver.DistanceSolver):
    def __init__(
        self,
        character_matrix: pd.DataFrame,
        meta_data: Optional[pd.DataFrame] = None,
        priors: Optional[Dict[int, str]] = None,
        dissimilarity_map: Optional[pd.DataFrame] = None,
        dissimilarity_function: Optional[Callable] = None,
    ):

        super().__init__(
            character_matrix,
            meta_data,
            priors,
            dissimilarity_map=dissimilarity_map,
            dissimilarity_function=dissimilarity_function,
        )

    def find_cherry(self, dissimilarity_matrix: np.array) -> Tuple[int, int]:
        """Finds a pair of samples to join into a cherry.

        Proceeds by minimizing the Q-criterion as in Saitou and Nei (1987) to
        select a pair of samples to join.

        Args:
            dissimilarity_matrix: A sample x sample dissimilarity matrix

        Returns:
            A tuple of intgers representing rows in the dissimilarity matrix
                to join.
        """

        q = self.compute_q(dissimilarity_matrix)
        np.fill_diagonal(q, np.inf)

        _min = np.argmin(q)

        i, j = _min % q.shape[0], _min // q.shape[0]

        return (i, j)

    @staticmethod
    @numba.jit(nopython=True, parallel=True)
    def compute_q(dissimilarity_map):
        """Computes the Q-criterion for every pair of samples.

        Computes the Q-criterion defined by Saitou and Nei (1987):

            Q(i,j) = d(i, j) - 1/(n-2) (sum(d(i, :)) + sum(d(j,:)))

        Args:
            dissimilarity_map: A sample x sample dissimilarity map
        
        Returns:
            A matrix storing the Q-criterion for every pair of samples.
        """

        q = np.zeros(dissimilarity_map.shape)
        n = dissimilarity_map.shape[0]
        for i in range(n):
            for j in range(i):
                q[i, j] = q[j, i] = (dissimilarity_map[i, j]) - (
                    1
                    / (n - 2)
                    * (
                        dissimilarity_map[i, :].sum()
                        + dissimilarity_map[j, :].sum()
                    )
                )
        return q

    def update_dissimilarity_map(
        self,
        dissimilarity_map: pd.DataFrame,
        cherry: Tuple[str, str],
        new_node: str,
    ) -> pd.DataFrame:

        updated_map = dissimilarity_map.drop(
            index=list(cherry), columns=list(cherry)
        )

        for v in dissimilarity_map.index:

            if v in cherry:
                continue

            updated_map.loc[v, new_node] = updated_map.loc[new_node, v] = (
                0.5
                * (
                    dissimilarity_map.loc[v, cherry[0]]
                    + dissimilarity_map.loc[v, cherry[1]]
                    - dissimilarity_map.loc[cherry[0], cherry[1]]
                )
            )

        updated_map.loc[new_node, new_node] = 0

        return updated_map
