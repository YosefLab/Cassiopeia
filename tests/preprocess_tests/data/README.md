# Pre-3.0 preprocessing outputs

Frozen outputs of the pre-3.0 Cassiopeia preprocessing pipeline (removed in
3.0; it needed `pyseq-align`, `hits` and `ngs-tools`). The parity tests use them
to check that `pp.align` aligns and merges reads as it did. Each file was
computed from inputs that the tests regenerate deterministically.

| File | Legacy function | Inputs | Used by |
| --- | --- | --- | --- |
| `legacy_reads.tsv` | `align_local` / `align_global` (pyseq-align, legacy scoring) | `native_test._queries()` | `native_test.py` |
| `legacy_merge.txt` | UMI cluster merge in `form_collapsed_clusters` (`hits` global alignment: at most 2 gap bases and 3 mismatches between bases of quality > 20). One `0`/`1` per pair | `native_test._read_pairs()` | `native_test.py` |

If the simulator or these inputs change, regenerate the files with the pre-3.0
code (Cassiopeia < 3.0).
