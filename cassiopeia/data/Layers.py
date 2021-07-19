"""
This file stores the data structure behind layers in a CassiopeiaTree object.

Briefly, Layers are collection of character matrices that can be used to
store multiple versions of mutation data for each cell. This can be useful,
for example, during simulation when a user is experimenting with imputing
missing data.

The data structure behaves like a dictionary, allowing users to store, retrieve,
and delete entries using canonical commands.

This data structure is inspired by AnnData's layer functionality for scRNA-seq
count matrices. Much of the code and logic is derived from the AnnData project.
"""
from typing import Iterator, List, Mapping, Optional

import pandas as pd

from cassiopeia.data import CassiopeiaTree


class Layers(dict):

    attrname = "layers"

    parent_mapping: Mapping[str, pd.DataFrame]

    def __init__(
        self, parent: CassiopeiaTree, layers: Optional[Mapping] = None
    ):
        self._parent = parent
        self._data = dict()
        if layers is not None:
            self.update(layers)

    def __repr__(self):
        return f"{type(self).__name__} with keys: {', '.join(self.keys())}"

    def _ipython_key_completions_(self) -> List[str]:
        return list(self.keys())

    def copy(self):
        d = Layers(self._parent)
        for k, v in self.items():
            d[k] = v.copy()
        return d

    def __getitem__(self, key: str) -> pd.DataFrame:
        return self._data[key]

    def __setitem__(self, key: str, value: pd.DataFrame):
        value = self._validate_value(value, key)
        self._data[key] = value

    def __delitem__(self, key: str):
        del self._data[key]

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def _validate_value(self, val: pd.DataFrame, key: str) -> pd.DataFrame:
        """Checks passed value for correct structure."""

        if val.shape[0] != self._parent.n_cell:
            raise ValueError(
                f"Value passed for key {key!r} is of incorrect shape. "
                f"Values of {self.attrname} must have the same number of "
                f"samples as the tree. Value had {val.shape[0]} while it "
                f"should have had {self._parent.n_cell} samples."
            )
        return val
