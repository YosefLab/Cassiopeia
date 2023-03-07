### Inferring fitness from the shape of trees 

This repository contains the code associated with the manuscript

Neher, Russell, Shraiman: "Predicting evolution from the shape of genealogical trees". accepted for publication in eLife

---

The directory *prediction_src* contains the code base used for the fitness inference and prediction algorithms as well as classes to hold sequence data and trees adapted.

---

The directory *flu* contains the code specific to our analysis of historical influenza data, scripts that generate the figures, the influenza sequences and annotation, analysis results and figure files.

---

The directory *toy_data* contains the code to simulate adapting populations building on the FFPopSim library. In addition, it contains scripts to analyze this simulated data, the data itself and the resulting figures.

---

#### Ranking sequences by the local branching index (LBI)

The script *rank_sequences.py* is a simple wrapper for the prediction tool that takes a multiple sequence alignment and the name of the outgroup as input (this outgroup needs to be in the MSA). It produces a folder containing a ranking of sequences, the inferred ancestral sequences, the reconstructed tree, and optionally a pdf of the marked up tree. This script uses the local branching index (LBI), rather than the full fitness inference to rank sequences.  

build-in help and optional arguments:

    ./rank_sequences.py --help
    usage: rank_sequences.py [-h] --aln ALN --outgroup OUTGROUP
                             [--eps_branch EPS_BRANCH] [--tau TAU]
                             [--collapse [COLLAPSE]] [--plot [PLOT]]
    
    rank sequences in a multiple sequence aligment
    
    optional arguments:
      -h, --help            show this help message and exit
      --aln ALN             alignment of sequences to by ranked
      --outgroup OUTGROUP   name of outgroup sequence
      --eps_branch EPS_BRANCH
                            minimal branch length for inference
      --tau TAU             time scale for local tree length estimation (relative
                            to average pairwise distance)
      --collapse [COLLAPSE]
                            collapse internal branches with identical sequences
      --plot [PLOT]         plot trees
    
#### Inferring fitness distribution of nodes in the tree

The script *infer_fitness.py* also takes an alignment and outgroup as argument, but uses the full fitness inference to rank sequences and calculate the mean posterior and the variance of the posterior. Note that plausible posterior distributions require a that the parameter omega is well chosen. Also, the time conversion factor might need to be different from gamma=1 for optimal results.

    ./infer_fitness.py --help
    usage: infer_fitness.py [-h] --aln ALN --outgroup OUTGROUP
                            [--eps_branch EPS_BRANCH] [--diffusion DIFFUSION]
                            [--gamma GAMMA] [--omega OMEGA]
                            [--collapse [COLLAPSE]] [--plot [PLOT]]
    
    rank sequences in a multiple sequence aligment
    
    optional arguments:
      -h, --help            show this help message and exit
      --aln ALN             alignment of sequences to by ranked
      --outgroup OUTGROUP   name of outgroup sequence
      --eps_branch EPS_BRANCH
                            minimal branch length for inference
      --diffusion DIFFUSION
                            fitness diffusion coefficient
      --gamma GAMMA         scale factor for time scale, choose high (>2) for
                            prediction, 1 for fitness inference
      --omega OMEGA         approximate sampling fraction diveded by the fitness
                            standard deviation
      --collapse [COLLAPSE]
                            collapse internal branches with identical sequences
      --plot [PLOT]         plot trees
    




