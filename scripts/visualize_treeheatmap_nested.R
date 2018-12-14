require(phytools)
require(RColorBrewer)
require(ggplot2)
require(reshape2)
require(plotrix)

args = commandArgs(trailingOnly = T)
tree.fp = args[[1]]
tree.fp2 = args[[2]]
tree.fp3 = args[[3]]
X.fp1 = args[[4]]
X.fp2 = args[[5]]
X.fp3 = args[[6]]

tree1 = read.newick(tree.fp)
print(paste0("Number of samples: ", length(tree1$tip.label)))
tree1$edge.length = rep(1, length(tree1$edge))

spl = unlist(strsplit(tree.fp, ".txt", fixed=T))
out1 = paste0(spl[[1]], ".pdf")

tree2 = read.newick(tree.fp2)
print(paste0("Number of samples: ", length(tree2$tip.label)))
tree2$edge.length = rep(1, length(tree2$edge))

spl = unlist(strsplit(tree.fp2, ".txt", fixed=T))
out2 = paste0(spl[[1]], ".pdf")

tree3 = read.newick(tree.fp3)
print(paste0("Number of samples: ", length(tree3$tip.label)))
tree3$edge.length = rep(1, length(tree3$edge))

spl = unlist(strsplit(tree.fp3, ".txt", fixed=T))
out3 = paste0(spl[[1]], ".pdf")

get_unique_indels <- function(X.fp, tree) { 

    X = as.matrix(read.table(X.fp, sep='\t', row.names=1, header=T))
    rn = rownames(X)

    X = X[tree$tip.label,]

    unique_alleles = unique(as.character(X))

    return(list(X, unique_alleles))

}

process_X <- function(X, allele_map) { 

    # read in allele heatmap
    message("generating allele heatmap...")

    X.sub = apply(X, c(1,2), function(x) allele_map[x])

    rownames(X.sub) <- rownames(X)
    colnames(X.sub) <- colnames(X)

    return(X.sub)
}

X1.out = get_unique_indels(X.fp1, tree1)
X2.out = get_unique_indels(X.fp2, tree2)
X3.out = get_unique_indels(X.fp3, tree3)

unique_alleles = unique(c(X1.out[[2]], X2.out[[2]], X3.out[[2]]))
allele_map = 1:length(unique_alleles)
names(allele_map) <- unique_alleles

X1.sub = process_X(X1.out[[1]], allele_map)
X2.sub = process_X(X2.out[[1]], allele_map)
X3.sub = process_X(X3.out[[1]], allele_map)

redmag = c(0.5, 1, 0, 0.5, 0, 1)
grnyel = c(0, 1, 0.5, 1, 0, 0.5)
cynblu = c(0, 0.5, 0, 1, 0.5, 1)
color_list = list("red" = redmag, "grn" = grnyel, "blue" = cynblu)

random_color <- function(rgb) { 
  red = runif(1, rgb[[1]], rgb[[2]])
  grn = runif(1, rgb[[3]], rgb[[4]])
  blu = runif(1, rgb[[5]], rgb[[6]])
  return(rgb(red, grn, blu, maxColorValue=1))
}

# randomly assign colors
heatmap.cbpalette = sapply(names(allele_map), function(a) {
  x = as.character(a)
  if (nchar(x) == 0) {
    return("#FFFFFF")
  } else if (x == "NC") { 
    return("#000000")  
  } else if (grepl("None", x)) { 
    return(rgb(0.75, 0.75, 0.75, maxColorValue=1))
  }
  
  #if (grepl("I", x)) {
  #  rgb_i = color_list[["red"]]
  #} else if (grepl("D", x)) {
  #  rgb_i = color_list[["blue"]]
  #}
  rgb_i = color_list[[sample(names(color_list), 1)]]
  return(random_color(rgb_i))
})

#names(heatmap.cbpalette) <- as.integer(factor(unique_alleles))
names(heatmap.cbpalette) <- allele_map
sorted_names = sort(sapply(names(heatmap.cbpalette), as.integer))
heatmap.cbpalette <- heatmap.cbpalette[as.character(sorted_names)]

message("creating color scheme...")
#allele_cols = colorRampPalette(brewer.pal(11, "Paired"))(max(nalleles))
#allele_cols = sample(allele_cols, length(allele_cols))

#heatmap.cbpalette = c("#C0C0C0", allele_cols)

names(heatmap.cbpalette) <- sapply(names(heatmap.cbpalette), as.numeric)

print(heatmap.cbpalette)
print(allele_map[1:10])

# now plot
message("plotting...")
X.subs = list(X1.sub, X2.sub, X3.sub)
trees = list(tree1, tree2, tree3)
outs = list(out1, out2, out3)
for (i in 1:3) { 
    sorted_names = sort(sapply(names(heatmap.cbpalette), as.integer))
    heatmap.cbpalette <- heatmap.cbpalette[as.character(sorted_names)]
    pdf(outs[[i]], height = 7, width = 7, compress=F)
    phylo.heatmap(trees[[i]], X.subs[[i]], labels=F, legend = F, ftype="off", colors=heatmap.cbpalette, new=T)
    dev.off()
}

