"""Clone calling: group cells that share an integration (intBC) signature."""

from __future__ import annotations

import contextlib
import io
import logging
from pathlib import Path
from typing import Literal

import pandas as pd

from cassiopeia.mixins import PreprocessError, logger

from ._utils import finish_plot, passing, record_params, set_pass

_NOVA_HELP = (
    "pp.call_clones(method='nova_clone') requires NovaClone "
    "(https://github.com/YosefLab/nova-clone). Install it with:\n"
    "    git clone https://github.com/YosefLab/nova-clone\n"
    "    export PYTHONPATH=$PWD/nova-clone:$PYTHONPATH\n"
    "and its dependencies (editdistance, networkx, scikit-learn, seaborn). "
    "For data where all cells share the same integrations use method='single'."
)


def intbc_table(df: pd.DataFrame, use_pass: bool = True) -> pd.DataFrame:
    """UMI counts per (cell, intBC) from passing molecules."""
    d = passing(df) if use_pass else df
    keys = ["cellBC", "intBC"]
    if "sample" in d.columns:
        keys = ["sample"] + keys
    t = d.groupby(keys, observed=True).size().rename("UMI_count").reset_index()
    return t[t["UMI_count"] > 0]


def _signatures(
    table: pd.DataFrame, cell_clone: pd.Series, min_detection: float
) -> dict[str, list[str]]:
    t = table.assign(clone=table["cellBC"].astype(str).map(cell_clone))
    t = t[t["clone"].notna() & (t["clone"] != "multiplet")]
    sizes = cell_clone[cell_clone != "multiplet"].value_counts()
    det = t.groupby(["clone", "intBC"], observed=True)["cellBC"].nunique().reset_index(name="n")
    det["frac"] = det["n"] / det["clone"].map(sizes).astype(float)
    det = det[det["frac"] >= min_detection]
    sig = det.groupby("clone")["intBC"].apply(lambda s: sorted(map(str, s))).to_dict()
    return {str(k): v for k, v in sig.items()}


def call_clones(
    df: pd.DataFrame,
    method: Literal["nova_clone", "single"] = "nova_clone",
    *,
    clones: pd.Series | dict | None = None,
    min_detection: float = 0.4,
    return_raw: bool = False,
    plot: bool = False,
    show: bool = True,
    save: str | Path | None = None,
    **nova_kwargs,
) -> pd.DataFrame | tuple[pd.DataFrame, object]:
    """Assign cells to clones (groups of cells sharing an intBC signature).

    * ``method="nova_clone"``: data with many clones, each with its own set
      of intBCs (e.g. barcoded populations). Uses NovaClone; cells mapping to
      more than one clone are labelled ``"multiplet"``.
    * ``method="single"``: all cells share the same integrations (a single
      clone, e.g. a monoclonal line, an embryo from one founder, GESTALT).
    * ``clones=``: user-supplied cell -> clone assignment.

    The clone's intBC signature (``df.attrs['clone_intbcs']``) lists intBCs
    detected in at least ``min_detection`` of its cells (NovaClone signatures
    are used for ``nova_clone``).

    Args:
        df: Molecule table.
        method: ``"nova_clone"`` or ``"single"``.
        clones: Optional mapping of cellBC to clone label (overrides ``method``).
        min_detection: Detection rate for an intBC to join a clone signature.
        return_raw: Also return the raw NovaClone result.
        plot: Plot clone sizes and intBC detection per clone.
        show: Show the plot.
        save: Save the plot to this path.
        **nova_kwargs: Passed to ``nova_clone.clone_calling_algorithm`` (e.g.
            ``min_intbc_umi_count``, ``min_cells_assigned``, ``min_intbc_fraction``).

    Returns:
        The molecule table with a ``clone`` column and ``pass_clone``
        (cells assigned to a clone).
    """
    df = df.copy()
    table = intbc_table(df)
    raw = None
    cells = table["cellBC"].astype(str).unique()
    if clones is not None:
        cell_clone = pd.Series(clones, dtype=object).astype(str)
        cell_clone = cell_clone.reindex(cells)
        method_used = "user"
        sig = _signatures(table, cell_clone.dropna(), min_detection)
    elif method == "single":
        cell_clone = pd.Series("0", index=cells, dtype=object)
        method_used = "single"
        sig = _signatures(table, cell_clone, min_detection)
    elif method == "nova_clone":
        try:
            import nova_clone
        except ImportError as e:
            raise ImportError(_NOVA_HELP) from e
        samples = (
            table["sample"].astype(str)
            if "sample" in table.columns
            else pd.Series("s", index=table.index)
        )
        safe = {s: f"s{i}" for i, s in enumerate(sorted(samples.unique()))}
        per_sample = {}
        for s, sub in table.groupby(samples, sort=True):
            per_sample[safe[s]] = pd.DataFrame(
                {
                    "cellBC": sub["cellBC"].astype(str).to_numpy(),
                    "intBC": sub["intBC"].astype(str).to_numpy(),
                    "UMI_count": sub["UMI_count"].astype(int).to_numpy(),
                }
            )
        # NovaClone logs every step at INFO level; keep only warnings
        previous = logging.root.manager.disable
        logging.disable(logging.INFO)
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                raw = nova_clone.clone_calling_algorithm(per_sample, **nova_kwargs)
        finally:
            logging.disable(previous)
        call = raw.clone_call
        mapping = {}
        for cb, cl in call.cell_to_clones_mapping.items():
            _, _, cell = cb.partition("_")
            if len(cl) == 1:
                mapping[cell] = next(iter(cl)).replace("clone_", "")
            elif len(cl) > 1:
                mapping[cell] = "multiplet"
        cell_clone = pd.Series(mapping, dtype=object).reindex(cells)
        sig = {
            k.replace("clone_", ""): sorted(map(str, v)) for k, v in call.clone_intbc_sets.items()
        }
        method_used = "nova_clone"
    else:
        raise PreprocessError(f"Unknown clone calling method {method!r}.")

    clone_col = df["cellBC"].astype(str).map(cell_clone)
    order = sorted(
        [c for c in pd.unique(cell_clone.dropna()) if c != "multiplet"], key=lambda x: (len(x), x)
    )
    if (cell_clone == "multiplet").any():
        order.append("multiplet")
    df["clone"] = pd.Categorical(clone_col, categories=order)
    set_pass(df, "clone", df["clone"].notna().to_numpy())
    df.attrs["clone_intbcs"] = sig
    df.attrs.pop("integrations", None)
    record_params(df, clone_method=method_used, clone_min_detection=min_detection)
    sizes = cell_clone.value_counts()
    n_assigned = int(sizes.drop("multiplet", errors="ignore").sum())
    logger.info(
        f"call_clones ({method_used}): {len(sig)} clones, {n_assigned:,}/{len(cells):,} cells assigned"
        + (f", {int(sizes.get('multiplet', 0)):,} multiplets" if "multiplet" in sizes else "")
    )
    if len(sig) == 1 and method_used == "nova_clone" and n_assigned > 0.95 * len(cells):
        logger.warning("NovaClone found a single clone; consider method='single'.")
    if plot or save:
        from .plotting import plot_clones

        fig = plot_clones(df)
        finish_plot(fig, show and bool(plot), save)
    if return_raw:
        return df, raw
    return df


def clone_sizes(df: pd.DataFrame) -> pd.Series:
    """Number of cells per clone."""
    if "clone" not in df.columns:
        raise PreprocessError("Run pp.call_clones first.")
    return df.drop_duplicates("cellBC")["clone"].value_counts()


def _cell_clone(df: pd.DataFrame) -> pd.Series:
    """Clone label (str) of each cellBC, NaN when unassigned."""
    if "clone" not in df.columns:
        return pd.Series(dtype=object)
    s = df.drop_duplicates("cellBC").set_index("cellBC")["clone"]
    s.index = s.index.astype(str)
    return s.astype(object).where(s.notna(), None).astype(object)
