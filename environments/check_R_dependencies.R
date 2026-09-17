# Environment check only; does not install packages or run DIABLO.
pkgs <- c("dplyr", "mixOmics", "pROC", "data.table")
missing <- pkgs[!vapply(pkgs, requireNamespace, logical(1), quietly=TRUE)]
if (length(missing)) stop(paste("Missing R packages:", paste(missing, collapse=", ")))
print(R.version.string)
print(vapply(pkgs, function(x) as.character(packageVersion(x)), character(1)))
