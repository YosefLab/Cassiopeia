SingleCellLineageTracing.TreeSolver package
============================================

Lineage Solver module
----------------------------------------------------------

Lineage Solving Entry Points
__________________________________________________________________

.. automodule:: SingleCellLineageTracing.TreeSolver.lineage_solver.lineage_solver
    :members:
    :undoc-members:
    :show-inheritance:

Greedy Solver
__________________________________________________________________

.. automodule:: SingleCellLineageTracing.TreeSolver.lineage_solver.greedy_solver
    :members:
    :undoc-members:
    :show-inheritance:

ILP Solver
__________________________________________________________________

.. automodule:: SingleCellLineageTracing.TreeSolver.lineage_solver.ILP_solver
    :members:
    :undoc-members:
    :show-inheritance:

Solver Utilities
__________________________________________________________________

.. automodule:: SingleCellLineageTracing.TreeSolver.lineage_solver.solver_utils
    :members:
    :undoc-members:
    :show-inheritance:

Post-Processing Trees
_______________________

.. automodule:: SingleCellLineageTracing.TreeSolver.post_process_tree
    :members: prune_and_clean_leaves, assign_samples_to_charstrings, add_redundant_leaves, post_process_tree
    :undoc-members:
    :show-inheritance:

Benchmarking module
---------------------------------------------------------------

Dataset Generation
__________________________________________________________________________

.. automodule:: SingleCellLineageTracing.TreeSolver.simulation_tools.dataset_generation
    :members: 
    :undoc-members:
    :show-inheritance:

Validation
________________________________________________________________
.. automodule:: SingleCellLineageTracing.TreeSolver.simulation_tools.validation
   :members:
   :undoc-members:
   :show-inheritance:

Simulation Utilities
_______________________________________________________________________
.. automodule:: SingleCellLineageTracing.TreeSolver.simulation_tools.simulation_utils
  :members:
  :undoc-members:
  :show-inheritance:

Meta Analysis
_______________
.. automodule:: SingleCellLineageTracing.TreeSolver.compute_meta_purity
  :members: get_max_depth, extend_dummy_branches, set_progeny_size, get_progeny_size, get_children_of_clade, get_meta_counts, set_depth, cut_tree, calc_entropy, sample_chisq_test, compute_mean_membership, assign_meta, nearest_neighbor_dist,  
  :undoc-members:
  :show-inheritance:

Processing Allele Tables & Character Matrices
--------------------------------------------------

.. automodule:: SingleCellLineageTracing.TreeSolver.data_pipeline
    :members:
    :undoc-members:
    :show-inheritance:
