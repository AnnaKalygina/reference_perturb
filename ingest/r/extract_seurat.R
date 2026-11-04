#!/usr/bin/env Rscript
# Rscript extract_seurat.R <in.rds> <out_dir> <tag>: raw RNA counts of a Seurat object, reached through attr() without SeuratObject,
# to p.bin/i.bin/x.bin (dgCMatrix genes x cells, so (p, i, x) is a CSR over cells), genes.txt, cells.txt, <tag>.meta.csv, info.txt.
suppressPackageStartupMessages(library(Matrix))
args <- commandArgs(trailingOnly = TRUE)
rds <- args[1]; outdir <- args[2]; tag <- args[3]
dir.create(outdir, showWarnings = FALSE, recursive = TRUE)
obj <- readRDS(rds)
md <- attr(obj, "meta.data")
assays <- attr(obj, "assays")
an <- if ("RNA" %in% names(assays)) "RNA" else attr(obj, "active.assay")
rna <- assays[[an]]
as_lm <- function(x) { y <- unclass(x); attr(y, "package") <- NULL; y }
lay <- attr(rna, "layers")
if (!is.null(lay)) {
  keys <- grep("^counts", names(lay), value = TRUE)
  if (length(keys) == 0) stop("no counts layer among: ", paste(names(lay), collapse = ", "))
  fm <- as_lm(attr(rna, "features")); cm <- as_lm(attr(rna, "cells"))
  allg <- rownames(fm); allc <- rownames(cm)
  parts <- list(); cells <- character(0); genes <- NULL
  for (k in keys) {
    m <- lay[[k]]
    g <- allg[fm[, k]]; cc <- allc[cm[, k]]
    if (length(g) != nrow(m) || length(cc) != ncol(m)) stop("layer ", k, ": map sizes disagree with the matrix")
    if (is.null(genes)) genes <- g
    if (!identical(g, genes)) stop("counts layers carry different gene sets")
    parts[[k]] <- m; cells <- c(cells, cc)
  }
  cnt <- if (length(parts) == 1) parts[[1]] else do.call(cbind, parts)
} else {
  cnt <- attr(rna, "counts")
  genes <- rownames(cnt); cells <- colnames(cnt)
}
rm(obj, assays, rna, lay); invisible(gc())
if (!is(cnt, "dgCMatrix")) cnt <- as(cnt, "CsparseMatrix")
if (length(cnt@x) >= 2^31) stop("nnz does not fit int32 pointers")
if (any(cnt@x != round(cnt@x))) stop("non-integer counts")
wr <- function(v, path) {
  con <- file(path, "wb"); n <- length(v); step <- 1e8
  for (lo in seq(1, max(n, 1), by = step)) { hi <- min(n, lo + step - 1); if (n > 0) writeBin(as.integer(v[lo:hi]), con, size = 4) }
  close(con)
}
wr(cnt@p, file.path(outdir, "p.bin")); wr(cnt@i, file.path(outdir, "i.bin")); wr(cnt@x, file.path(outdir, "x.bin"))
writeLines(as.character(genes), file.path(outdir, "genes.txt"))
writeLines(as.character(cells), file.path(outdir, "cells.txt"))
if (!identical(rownames(md), cells)) { if (!all(cells %in% rownames(md))) stop("count columns without meta.data rows"); md <- md[cells, , drop = FALSE] }
write.csv(cbind(data.frame(cell = rownames(md), stringsAsFactors = FALSE), md), file.path(outdir, paste0(tag, ".meta.csv")), row.names = FALSE)
writeLines(c(paste("n_genes", nrow(cnt)), paste("n_cells", ncol(cnt)), paste("nnz", length(cnt@x)), paste("source", rds)), file.path(outdir, "info.txt"))
cat(tag, nrow(cnt), "genes", ncol(cnt), "cells", length(cnt@x), "nnz\n")
