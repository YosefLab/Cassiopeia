require(phytools)
require(RColorBrewer)
require(ggplot2)
require(reshape2)

args = commandArgs(trailingOnly = T)
tree.fp = args[[1]]
X.fp = args[[2]]
meta.fp = args[[3]]
target_meta = args[[4]]
out_fp = args[[5]]

color.bar <- function(lut, min, max=-min, nticks=11, ticks=seq(min, max, len=nticks), title='') {
  scale = (length(lut))/(max-min)
  
  plot(c(0,10), c(min,max), type='n', bty='n', xaxt='n', xlab='', yaxt='n', ylab='')
  for (i in 1:(length(lut)-1)) {
    y = (i)/scale + min
    rect(0,y,10,y+1/scale, col=lut[i], border=NA)
  }
  
}

tree = read.newick(tree.fp)
print(paste0("Number of samples: ", length(tree$tip.label)))
tree$edge.length = rep(1, length(tree$edge))

# read in allele table to extract sample id
all.meta = read.table(meta.fp, sep='\t', header=T, row.names=1)
meta = all.meta[, target_meta,drop=F]

meta = meta[tree$tip.label,]

# read in allele heatmap
message("generating allele heatmap...")
X = read.table(X.fp, sep='\t', row.names=1, header=T)
X = as.matrix(X[-c(1,2),])
rn = rownames(X)

all.X <- X

X = X[tree$tip.label, ]

emptylocs = sapply(as.vector(X), function(x) nchar(x) == 0)
nonelocs = sapply(as.vector(X), function(x) grepl("None", x))
#X2 = t(apply(X, 1, function(x) mapply(paste0, x, intbc)))

X = apply(X, 2, function(x) as.numeric(factor(x, levels=unique(x))))
X2 = as.numeric(factor(X), levels=unique(X))
nalleles = length(unique(X))
unique_alleles = unique(X)
#X[X == 0] = NA
#X[X == "-"] = -1
X2[emptylocs] = NA
X2[nonelocs] = -1
X.sub = matrix(X2, nrow(X), ncol(X))
rownames(X.sub) <- tree$tip.label


message("creating color scheme...")


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
heatmap.cbpalette = sapply(unique_alleles, function(a) {
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


names(heatmap.cbpalette) <- sapply(names(heatmap.cbpalette), as.numeric)


sample.cbpalette = colorRampPalette(brewer.pal(9, "Set1"))(length(unique(meta)))
names(sample.cbpalette) = rev(sapply(unique(meta), as.character))

#sample.cbpalette = c("#ED1C24", "#FCEE21", "#green", "#blue")
#names(sample.cbpalette) = c("IVLT-2B_00", "IVLT-2B_01", "IVLT-2B_10", "IVLT-2B_11")

cvec = sapply(meta, function(x) sample.cbpalette[[as.character(x)]])
#cvec = rev(cvec) # tree is built from bottom up, so need to flip cvec

print(length(meta))
print(length(tree$tip.label))

# now plot
message("plotting...")
pdf(out_fp, height = 50, width = 50, compress=F)
par(fig=c(0, 0.75, 0.1, 1.0))
phylo.heatmap(tree, X.sub, labels=F, legend = F, ftype="off", colors=heatmap.cbpalette, new=T)
par(fig=c(0.75, 0.85, 0.1, 1.0), new=T)
color.bar(cvec, -1)
par(fig=c(0, 1, 0, 0.2), new=T)
plot(NULL, xaxt='n', yaxt='n', bty='n', ylab='', xlab='', xlim=0:1, ylim=0:1)
legend(x="bottom", legend = names(sample.cbpalette), col = sample.cbpalette, pch=16, pt.cex=10, cex=10, bty='n', horiz=T)
dev.off()
