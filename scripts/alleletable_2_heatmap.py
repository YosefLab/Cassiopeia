#' A small script to take an allele table and convert it to a character matrix ready to be plotted in
#' heatmap format. 

import sys

from pylab import * 
import pandas as pd
from collections import defaultdict
from tqdm import tqdm

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("at_fp", type = str, help="character_matrix")
parser.add_argument("out_fp", type=str, help="output file name")
parser.add_argument("--mutation_map", type=str, default="")
parser.add_argument("--old_r", action="store_true", default=False)
    
args = parser.parse_args() 

at_fp = args.at_fp
out_fp = args.out_fp
old_r = args.old_r

lg = pd.read_csv(at_fp, sep='\t')

if old_r:
    g = lg.groupby(["cellBC", "intBC"]).agg({"r1.old": "unique", "r2.old": "unique", "r3.old": "unique"})
else:
    g = lg.groupby(["cellBC", "intBC"]).agg({"r1": "unique", "r2": "unique", "r3": "unique"})

intbcs = lg["intBC"].unique()

# create mutltindex df by hand 
i1 = []
for i in intbcs:
    i1 += [i]*3

if old_r:
    i2 = ["r1.old", "r2.old", "r3.old"] * len(intbcs)
else:
    i2 = ["r1", "r2", "r3"] * len(intbcs)

indices = [i1, i2]

allele_piv = pd.DataFrame(index=g.index.levels[0], columns=indices)

for j in tqdm(g.index, desc="filling in multiindex table"):
    vals = map(lambda x: x[0], g.loc[j])
    if old_r:
        allele_piv.loc[j[0]][j[1], "r1.old"], allele_piv.loc[j[0]][j[1], "r2.old"], allele_piv.loc[j[0]][j[1], "r3.old"] = vals
    else:
        allele_piv.loc[j[0]][j[1], "r1"], allele_piv.loc[j[0]][j[1], "r2"], allele_piv.loc[j[0]][j[1], "r3"] = vals


allele_piv2 = pd.pivot_table(lg, index=["cellBC"], columns=["intBC"], values="UMI", aggfunc=size)
col_order = allele_piv2.dropna(axis=1, how="all").sum().sort_values(ascending=False, inplace=False).index

to_plot = allele_piv[col_order]

to_plot.to_csv(out_fp, sep='\t')

