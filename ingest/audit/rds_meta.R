suppressPackageStartupMessages(library(Matrix))
args <- commandArgs(trailingOnly = TRUE)
rds <- args[1]; outdir <- args[2]; tag <- args[3]
dir.create(outdir, showWarnings = FALSE, recursive = TRUE)
obj <- readRDS(rds)
md <- attr(obj, "meta.data")
assays <- attr(obj, "assays")
an <- if ("RNA" %in% names(assays)) "RNA" else attr(obj, "active.assay")
rna <- assays[[an]]
lay <- attr(rna, "layers")
cnt <- if (!is.null(lay)) lay[[grep("^counts", names(lay), value = TRUE)[1]]] else attr(rna, "counts")
x <- head(cnt@x, 2e6)
info <- sprintf('{"assay": "%s", "n_genes": %d, "n_cells": %d, "nnz": %.0f, "non_integer_in_sample": %d, "sample_values": [%s]}',
                an, nrow(cnt), ncol(cnt), length(cnt@x), sum(x != round(x)), paste(head(x, 20000), collapse = ","))
writeLines(info, file.path(outdir, paste0(tag, ".info.json")))
write.csv(cbind(data.frame(cell = rownames(md), stringsAsFactors = FALSE), md),
          file.path(outdir, paste0(tag, ".meta.csv")), row.names = FALSE)
