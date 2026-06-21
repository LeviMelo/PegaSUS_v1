rm(list = ls())

setwd("C:/Users/Galaxy/LEVI/PegaSUS")

.libPaths(c(
  normalizePath(".r-library", winslash = "/", mustWork = TRUE),
  .libPaths()
))

cat("\n=== RUNTIME ===\n")
cat("R:", R.version.string, "\n")
cat(".libPaths:\n")
print(.libPaths())

cat("\n=== PACKAGE LOAD ===\n")
library(microdatasus)
library(read.dbc)

cat("microdatasus version:", as.character(packageVersion("microdatasus")), "\n")
cat("read.dbc version:", as.character(packageVersion("read.dbc")), "\n")
cat("attached microdatasus:", "package:microdatasus" %in% search(), "\n")

cat("\n=== EXPORTS ===\n")
exports <- getNamespaceExports("microdatasus")
print(sort(exports[grepl("^fetch|^process_", exports)]))
cat("has process_datasus:", "process_datasus" %in% exports, "\n")
cat("has process_sim:", "process_sim" %in% exports, "\n")
cat("has process_sinasc:", "process_sinasc" %in% exports, "\n")

cat("\n=== PACKAGE DATA VISIBILITY ===\n")
pkg_data <- utils::data(package = "microdatasus")
print(str(pkg_data))

if (!is.null(pkg_data$results)) {
  print(head(pkg_data$results, 30))
  cat("tabNaturalidade in data(package):", "tabNaturalidade" %in% pkg_data$results[, "Item"], "\n")
}

cat("exists tabNaturalidade, default search:", exists("tabNaturalidade"), "\n")
cat(
  "exists tabNaturalidade in package env:",
  exists("tabNaturalidade", envir = as.environment("package:microdatasus"), inherits = FALSE),
  "\n"
)
cat(
  "exists tabNaturalidade in namespace:",
  exists("tabNaturalidade", envir = asNamespace("microdatasus"), inherits = FALSE),
  "\n"
)

if (exists("tabNaturalidade")) {
  cat("tabNaturalidade class/dim:\n")
  print(class(tabNaturalidade))
  print(dim(tabNaturalidade))
  print(head(tabNaturalidade))
}

cat("\n=== PROCESS_SIM SOURCE ENVIRONMENT ===\n")
print(environment(process_sim))
print(parent.env(environment(process_sim)))

cat("\n=== MANUAL DBC READ TEST ===\n")
dbc <- "data/diagnostics/microdatasus_probe/sim_al_2022/direct_SIM_DO_AL_2022_02.dbc"

x_default <- read.dbc::read.dbc(dbc)
x_asis <- read.dbc::read.dbc(dbc, as.is = TRUE)

cat("default rows/cols:", nrow(x_default), ncol(x_default), "\n")
cat("as.is rows/cols:", nrow(x_asis), ncol(x_asis), "\n")
cat("default TIPOBITO class:", paste(class(x_default$TIPOBITO), collapse = ","), "\n")
cat("as.is TIPOBITO class:", paste(class(x_asis$TIPOBITO), collapse = ","), "\n")

cat("\n=== PROCESS_SIM ON DEFAULT READ.DBC ===\n")
y_default <- tryCatch(
  process_sim(x_default, municipality_data = FALSE),
  error = function(e) e
)
print(class(y_default))
if (inherits(y_default, "error")) {
  cat("ERROR:\n")
  cat(conditionMessage(y_default), "\n")
} else {
  cat("SUCCESS rows/cols:", nrow(y_default), ncol(y_default), "\n")
  print(names(y_default))
}

cat("\n=== PROCESS_SIM ON READ.DBC AS.IS=TRUE ===\n")
y_asis <- tryCatch(
  process_sim(x_asis, municipality_data = FALSE),
  error = function(e) e
)
print(class(y_asis))
if (inherits(y_asis, "error")) {
  cat("ERROR:\n")
  cat(conditionMessage(y_asis), "\n")
} else {
  cat("SUCCESS rows/cols:", nrow(y_asis), ncol(y_asis), "\n")
  print(names(y_asis))
  print(head(y_asis))
}

cat("\n=== MICRODATASUS FETCH_NATIVE TEST ===\n")
fetched <- tryCatch(
  fetch_datasus(
    year_start = 2022,
    year_end = 2022,
    uf = "AL",
    information_system = "SIM-DO",
    timeout = 900,
    track_source = TRUE
  ),
  error = function(e) e
)

print(class(fetched))
if (inherits(fetched, "error")) {
  cat("fetch_datasus ERROR:\n")
  cat(conditionMessage(fetched), "\n")
} else if (is.null(fetched)) {
  cat("fetch_datasus returned NULL\n")
} else {
  cat("fetch_datasus SUCCESS rows/cols:", nrow(fetched), ncol(fetched), "\n")
  cat("TIPOBITO class:", paste(class(fetched$TIPOBITO), collapse = ","), "\n")
  print(head(fetched))
  
  cat("\n=== PROCESS_SIM ON FETCH_DATASUS OUTPUT ===\n")
  y_fetch <- tryCatch(
    process_sim(fetched, municipality_data = FALSE),
    error = function(e) e
  )
  print(class(y_fetch))
  if (inherits(y_fetch, "error")) {
    cat("ERROR:\n")
    cat(conditionMessage(y_fetch), "\n")
  } else {
    cat("SUCCESS rows/cols:", nrow(y_fetch), ncol(y_fetch), "\n")
    print(names(y_fetch))
    print(head(y_fetch))
  }
}

cat("\n=== DONE ===\n")