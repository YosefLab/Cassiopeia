# Lineage Tracing Solver

This project implements algorithms for solving the maximum parsimony problem in single-cell lineage tracing. It also contains tools for simulating interesting instances of the problem to test the feasibility and accuracy of the algorithms in practice.


#### Problem Statement

Formally, the lineage tracing problem is the following: we are given

1. A list of target cells/strings, where each cell is in the format 'Ch1|Ch2|....|Chn'.

2. Mutation probabilities, where p(i, j) = Probability character i transitions from '0', or the unmutated state, to j. Once mutated, a character is assumed to not mutate again.

3. Dropout probabilities, where p(i) = probability character i experiences a dropout event, and thus the character becomes NA

The task is to find a rooted subtree T of minimum total weight such that every target can be reached from the root

Our algorithm will be detailed in a forthcoming paper.


---
### Solving Lineage Tracing Instances



The main solver is invoked by calling the following function in `/lineage_solver/lineage_solver.py`:

```python
# See the docstring for details.
solve_lineage_instance(target_nodes, prior_probabilities = ..., method=...)
```
Note:
Three possible methods exist for solving the problem:
The methods used for solving the problem ['ilp, 'hybrid', 'greedy']

    - ilp: Attempts to solve the problem based on steiner tree on the potential graph
           (Recommended for instances with several hundred samples at most)

    - greedy: Runs a greedy algorithm to find the maximum parsimony tree based on choosing the most occurring split in a
           top down fasion (Algorithm scales to any number of samples)

    - hybrid: Runs the greedy algorithm until there are less than hybrid_subset_cutoff samples left in each leaf of the
           tree, and then returns a series of small instance ilp is then run on these smaller instances, and the
           resulting graph is created by merging the smaller instances with the greedy top-down tree


### Generating Artificial Instances

We implement the following procedure for generating sample lineage tracing instances

Given the following parameters, this method simulates the cell division and mutations over multiple lineages

    1) We begin with one progenitor cell with no mutations

    2) Each generation, all cells are duplicated, and each character is independently transformed
      with the probabilities of transformation defined in mutation_prob_map

    3) At the end of this process of duplication, there will be 2 ^ depth samples.

    4) We subsample a percentage of the final cells

    5) On the subsampled population, we simulate dropout on each individual character in each sample

This procedure is implemented in the following function in `/simulation_tools/dataset_generation.py`:

To see a sample workflow, please view:
`sample_workflow.py`
