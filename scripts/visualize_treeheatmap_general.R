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
  scale = (length(lut)-1)/(max-min)

  plot(c(0,10), c(min,max), type='n', bty='n', xaxt='n', xlab='', yaxt='n', ylab='')
  for (i in 1:(length(lut)-1)) {
    y = (i-1)/scale + min
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
meta = sapply(meta, function(x) ifelse(x == "11" || x == "10" || x == "1", "1", "0"))

# read in allele heatmap
message("generating allele heatmap...")
X = read.table(X.fp, sep='\t', row.names=1, header=T)
rn = rownames(X)

all.X <- X

X = X[tree$tip.label, ]

emptylocs = sapply(as.vector(X), function(x) nchar(as.character(x)) == 0)
nonelocs = sapply(as.vector(X), function(x) x == "-")
#X2 = t(apply(X, 1, function(x) mapply(paste0, x, intbc)))

X = apply(X, 2, function(x) as.numeric(factor(x, levels=unique(x))))
X2 = as.numeric(factor(X), levels=unique(X))
nalleles = length(unique(X))
print(nalleles)
X2[emptylocs] = NA
X2[nonelocs] = -1
X.sub = matrix(X2, nrow(X), ncol(X))
rownames(X.sub) <- tree$tip.label

message("creating color scheme...")
allele_cols = colorRampPalette(brewer.pal(11, "Paired"))(max(nalleles))
allele_cols = sample(allele_cols, length(allele_cols))

heatmap.cbpalette = c("#C0C0C0", allele_cols)
names(heatmap.cbpalette) <- sapply(names(heatmap.cbpalette), as.numeric)


sample.cbpalette = colorRampPalette(brewer.pal(10, "Set3"))(length(unique(meta)))
names(sample.cbpalette) = sapply(unique(meta), as.character)
cvec = sapply(meta, function(x) sample.cbpalette[[as.character(x)]])
cvec = rev(cvec) # tree is built from bottom up, so need to flip cvec

# now plot
message("plotting...")
pdf(paste0(out_fp), height = 50, width = 50, compress=F)
par(fig=c(0, 0.75, 0, 0.9))
phylo.heatmap(tree, X.sub, labels=F, legend = F, ftype="off", colors=heatmap.cbpalette, new=T)
par(fig=c(0.75, 0.85, 0, 0.9), new=T)
# p2 = ggplot(samp.melt, aes(Var2, Var1)) + geom_tile(aes(fill = value)) +
#       scale_fill_gradient(low="white", high="red") +
#       theme(axis.text.y = element_blank(), axis.ticks.y = element_blank(), axis.text.x = element_text(angle = 45, hjust = 1)) +
#       labs(x="", y="") + guides(fill=F)
#image(t(samp.heatmap), col=colorRampPalette(c("white", "red"))(1000), xlab="", ylab="", xaxt='n', ylab=colnames(samp.heatmap))
color.bar(cvec, -1)
par(fig=c(0.85, 1, 0, 0.9), new=T)
dev.off()
