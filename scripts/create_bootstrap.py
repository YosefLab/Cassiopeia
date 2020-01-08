from __future__ import division

import subprocess

import numpy as np
import pandas as pd
import pandascharm as pc

from Bio import AlignIO

import sys
import os

def msa_to_character_matrix(aln, sample_to_cells):
    
    df = pc.from_bioalignment(aln).T
    df.index = map(lambda x: samples_to_cells[x], df.index)

    return df

if __name__ == "__main__":

    charfp = sys.argv[1]
    bootfp = sys.argv[2]

    stem = ''.join(bootfp.split(".")[:-1])

    cm = pd.read_csv(charfp, sep='\t', index_col=0)

    cells = cm.index
    samples = [("s" + str(i)) for i in range(len(cells))]
    samples_to_cells = dict(zip(samples, cells))
        
    cm.index = list(range(len(cells)))
        
    infile = stem + 'infile.txt'
    fn = stem + "phylo.txt"
    weights_fn = stem + "weights.txt"
        
    cm.to_csv(fn, sep='\t')

    os.system("python2 /home/mattjones/projects/scLineages/SingleCellLineageTracing/scripts/binarize_multistate_charmat.py " + fn + " " + infile) 

    
    responses = "." + stem + ".temp.txt"
    FH = open(responses, 'w')
    current_dir = os.getcwd()
    FH.write(infile + "\n")
    FH.write("D\n")
    FH.write("Y\n")
    FH.write("F\n" + bootfp + "\n")
    FH.close()

    cmd = "~/software/phylip-3.697/exe/seqboot"
    cmd += " < " + responses + " > screenout" 
    p = subprocess.Popen(cmd, shell=True)
    pid, ecode = os.waitpid(p.pid, 0)

    # remove intermediate files
    os.system("rm " + responses)
    os.system("rm " + infile)

    boot_alns = list(AlignIO.parse("outfile", "phylip-sequential"))

    i = 0
    for a in boot_alns:
        boot_cm = msa_to_character_matrix(a, samples_to_cells)
        boot_cm.to_csv(stem + "_" + str(i) +  "_character_matrix.txt", sep='\t')
        i += 1

