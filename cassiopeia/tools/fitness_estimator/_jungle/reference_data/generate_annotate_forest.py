# Generate and annotate a Forest
# Usage: python generate_annotate_forest.py [n_leaves] [n_trees] [alpha] [output_dir]
# Saves Forest as a gzipped pickle archive.

import sys
import time
import uuid

sys.path.append("../../jungle/")
import jungle as jg

verbose = True

# Specify parameters
n_leaves = int(sys.argv[1])  # Number of leaves in tree
n_trees = int(sys.argv[2])  # Number of trees in forest
alpha = float(
    sys.argv[3]
)  # Shape parameter alpha (alpha = 2.0 for neutral Kingman trees, alpha = 1.0 for positive selection Bolthausen-Sznitman trees)
outfile_dir = sys.argv[4]  # Output directory

# Specify output file
outfile_vars = (n_leaves, n_trees, alpha, str(uuid.uuid4())[0:8])
outfile_basename = (
    "forest_nleaves{0}_ntrees{1}_alpha{2}_uuid{3}.pickle.gz".format(
        *outfile_vars
    )
)
outfile = outfile_dir + "/" + outfile_basename

# Report parameters
if verbose:
    print("Parameters")
    print(("n_leaves", n_leaves))
    print(("n_trees", n_trees))
    print(("alpha", alpha))
    print(("outfile_dir", outfile_dir))
    print(("outfile", outfile))

if verbose:
    print("Starting tree generation...")

# Track run time
start_time = time.time()

# Generate and annotate trees
F = jg.Forest.generate(
    n_trees=n_trees, params={"n_leaves": n_leaves, "alpha": alpha}
)
F.resolve_polytomy()
F.annotate_standard_node_features()
F.annotate_colless()

# Dump to file
F.dump(outfile)

# Track run time
elapsed_time = time.time() - start_time

# Report run time
if verbose:
    print("Done!!")
    print(("Elapsed time (s):", elapsed_time))
