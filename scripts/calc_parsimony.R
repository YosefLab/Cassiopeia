require(phangorn)

args = commandArgs(trailingOnly=T)

tree.fp = args[[1]]
char.fp = args[[2]]

cmd = paste0("python2 /home/mattjones/projects/scLineages/SingleCellLineageTracing/scripts/binarize_multistate_charmat.py ", char.fp, " infile --relaxed")

system(cmd)

aln = read.phyDat("infile", format = "phylip", type="USER", levels=c("0", "1"), ambiguity=c("?"))
tree = read.tree(tree.fp)

p = parsimony(tree, aln)

message(paste0("Tree Parsimony: ", p)) 
