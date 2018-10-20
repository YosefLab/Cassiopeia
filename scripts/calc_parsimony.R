require(phangorn)

args = commandArgs(trailingOnly=T)

tree.fp = args[[1]]
char.fp = args[[2]]
alg = args[[3]]
t = args[[4]]

infile = paste0(char.fp, "infile")

cmd = paste0("python2 /home/mattjones/projects/scLineages/SingleCellLineageTracing/scripts/binarize_multistate_charmat.py ", char.fp, " ", infile, " --relaxed")

system(cmd)

aln = read.phyDat(infile, format = "phylip", type="USER", levels=c("0", "1"), ambiguity=c("?"))
tree = read.tree(tree.fp)

p = parsimony(tree, aln)

spl1 = unlist(strsplit(tree.fp, "/", fixed=T))
name = spl1[[length(spl1)]]
spl2 = rev(unlist(strsplit(name, "_", fixed=T)))
param = spl2[[4]]
run = unlist(strsplit(spl2[[2]], ".", fixed=T))[[1]]

write(paste(param, run, p, alg, t, sep='\t'), stdout())

system(paste("rm", infile))
