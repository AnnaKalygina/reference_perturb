#!/usr/bin/env Rscript
# Rscript extract_seurat.R <in.rds> <out_dir> <tag>: raw RNA counts of a Seurat object (reached through attr(), no Seurat needed)
# to p.bin/i.bin/x.bin (dgCMatrix genes x cells, so (p, i, x) is a CSR over cells), genes.txt, cells.txt, <tag>.meta.csv, info.txt.
suppressPackageStartupMessages(library(Matrix))
args <- commandArgs(trailingOnly = TRUE)
rds <- args[1]; outdir <- args[2]; tag <- args[3]
dir.create(outdir, showWarnings = FALSE, recursive = TRUE)
obj <- readRDS(rds)
md <- attr(obj, "meta.data")
assays <- attr(obj, "assays")
a <- assays[["RNA"]]
if (is.null(a)) a <- assays[[1]]
lay <- attr(a, "layers")
cn <- if (!is.null(lay) && !is.null(lay[["counts"]])) lay[["counts"]] else attr(a, "counts")
if (is.null(cn)) stop("no counts layer")
if (!inherits(cn, "dgCMatrix")) cn <- as(cn, "CsparseMatrix")
if (any(cn@x != round(cn@x))) stop("non-integer counts")
feat <- rownames(cn); if (is.null(feat)) feat <- rownames(attr(a, "features")) ; if (is.null(feat)) feat <- attr(a, "features")
cells <- colnames(cn); if (is.null(cells)) cells <- rownames(md)
writeBin(as.integer(cn@p), file.path(outdir, "p.bin"), size = 4)
writeBin(as.integer(cn@i), file.path(outdir, "i.bin"), size = 4)
writeBin(as.integer(cn@x), file.path(outdir, "x.bin"), size = 4)
writeLines(as.character(feat), file.path(outdir, "genes.txt"))
writeLines(as.character(cells), file.path(outdir, "cells.txt"))
md$cell <- rownames(md)
write.csv(md[, c("cell", setdiff(names(md), "cell"))], file.path(outdir, paste0(tag, ".meta.csv")), row.names = FALSE)
writeLines(c(paste("n_genes", nrow(cn)), paste("n_cells", ncol(cn)), paste("nnz", length(cn@x)), paste("source", rds)), file.path(outdir, "info.txt"))
cat(tag, nrow(cn), "genes", ncol(cn), "cells", length(cn@x), "nnz\n")
