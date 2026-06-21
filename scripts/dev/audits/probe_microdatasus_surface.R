args <- commandArgs(trailingOnly = TRUE)

get_arg <- function(flag, default = NULL) {
  idx <- match(flag, args)
  if (is.na(idx) || idx == length(args)) return(default)
  args[[idx + 1]]
}

out_dir <- get_arg("--out-dir", "data/diagnostics/microdatasus_probe")
lib <- get_arg("--lib", ".r-library")
uf <- get_arg("--uf", "AL")
year <- as.integer(get_arg("--year", "2022"))
system_id <- get_arg("--system", "SIM-DO")
iterations <- as.integer(get_arg("--repeats", "3"))
timeout <- as.integer(get_arg("--timeout", "600"))

dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

if (!is.null(lib) && dir.exists(lib)) {
  .libPaths(c(normalizePath(lib, winslash = "/", mustWork = TRUE), .libPaths()))
}

json_line <- function(payload) {
  payload$timestamp <- format(Sys.time(), "%Y-%m-%dT%H:%M:%OS%z")
  cat(jsonlite::toJSON(payload, auto_unbox = TRUE, null = "null"), "\n")
  flush.console()
}

safe <- function(expr) {
  tryCatch(
    list(ok = TRUE, value = expr, error = NULL),
    error = function(e) list(ok = FALSE, value = NULL, error = conditionMessage(e))
  )
}

pkg_info <- function(pkg) {
  available <- requireNamespace(pkg, quietly = TRUE)
  list(
    package = pkg,
    available = available,
    version = if (available) as.character(utils::packageVersion(pkg)) else NULL,
    lib_path = if (available) normalizePath(find.package(pkg), winslash = "/", mustWork = TRUE) else NULL
  )
}

json_line(list(
  stage = "runtime",
  R_version = R.version.string,
  platform = R.version$platform,
  executable = R.home("bin"),
  libPaths = .libPaths(),
  packages = lapply(c(
    "microdatasus", "read.dbc", "arrow", "jsonlite",
    "dplyr", "dtplyr", "data.table", "vctrs", "lubridate",
    "timechange", "RCurl", "curl"
  ), pkg_info)
))

required <- c("microdatasus", "read.dbc", "jsonlite")
missing <- required[!vapply(required, requireNamespace, logical(1), quietly = TRUE)]
if (length(missing) > 0) {
  json_line(list(stage = "fatal_missing_required", missing = missing))
  quit(status = 30)
}

exports <- getNamespaceExports("microdatasus")
json_line(list(
  stage = "microdatasus_exports",
  has_fetch_datasus = "fetch_datasus" %in% exports,
  has_process_datasus = "process_datasus" %in% exports,
  has_process_sim = "process_sim" %in% exports,
  has_process_sinasc = "process_sinasc" %in% exports,
  has_process_sih = "process_sih" %in% exports,
  has_process_cnes = "process_cnes" %in% exports,
  process_exports = sort(exports[grepl("^process_", exports)])
))

if ("fetch_datasus" %in% exports) {
  json_line(list(
    stage = "fetch_formals",
    formals = paste(names(formals(microdatasus::fetch_datasus)), collapse = ",")
  ))
}

url_for <- function(system, uf, year) {
  if (system == "SIM-DO") {
    return(sprintf("ftp://ftp.datasus.gov.br/dissemin/publicos/SIM/CID10/DORES/DO%s%s.dbc", uf, year))
  }
  if (system == "SINASC") {
    return(sprintf("ftp://ftp.datasus.gov.br/dissemin/publicos/SINASC/1996_/Dados/DNRES/DN%s%s.dbc", uf, year))
  }
  if (system == "SIH-RD") {
    return(sprintf("ftp://ftp.datasus.gov.br/dissemin/publicos/SIHSUS/200801_/Dados/RD%s%02d%02d.dbc", uf, year %% 100, 1))
  }
  NA_character_
}

url <- url_for(system_id, uf, year)
json_line(list(stage = "target", system = system_id, uf = uf, year = year, direct_url = url))

for (i in seq_len(iterations)) {
  json_line(list(stage = "iteration_start", iteration = i))

  if (requireNamespace("RCurl", quietly = TRUE) && !is.na(url)) {
    rcurl_probe <- safe(RCurl::url.exists("ftp://ftp.datasus.gov.br"))
    json_line(list(
      stage = "rcurl_url_exists_bare_ftp_host",
      iteration = i,
      ok = rcurl_probe$ok,
      value = rcurl_probe$value,
      error = rcurl_probe$error
    ))

    rcurl_file_probe <- safe(RCurl::url.exists(url))
    json_line(list(
      stage = "rcurl_url_exists_exact_file",
      iteration = i,
      ok = rcurl_file_probe$ok,
      value = rcurl_file_probe$value,
      error = rcurl_file_probe$error
    ))
  }

  if (!is.na(url)) {
    direct_path <- file.path(out_dir, sprintf("direct_%s_%s_%s_%02d.dbc", gsub("[^A-Za-z0-9]", "_", system_id), uf, year, i))
    options(timeout = timeout)

    direct_download <- safe(utils::download.file(url, direct_path, mode = "wb", method = "libcurl", quiet = TRUE))
    direct_size <- if (file.exists(direct_path)) file.info(direct_path)[["size"]] else NA

    json_line(list(
      stage = "direct_download",
      iteration = i,
      ok = direct_download$ok,
      error = direct_download$error,
      path = direct_path,
      bytes = direct_size
    ))

    if (isTRUE(direct_download$ok) && file.exists(direct_path)) {
      direct_read <- safe(read.dbc::read.dbc(direct_path))
      direct_dim <- if (direct_read$ok && is.data.frame(direct_read$value)) dim(direct_read$value) else c(NA, NA)
      direct_cols <- if (direct_read$ok && is.data.frame(direct_read$value)) names(direct_read$value) else character()

      json_line(list(
        stage = "direct_read_dbc",
        iteration = i,
        ok = direct_read$ok,
        error = direct_read$error,
        rows = direct_dim[[1]],
        cols = direct_dim[[2]],
        colnames = direct_cols
      ))

      if (direct_read$ok && is.data.frame(direct_read$value)) {
        processor <- switch(system_id,
          "SIM-DO" = "process_sim",
          "SINASC" = "process_sinasc",
          "SIH-RD" = "process_sih",
          "CNES-ST" = "process_cnes",
          NULL
        )

        if (!is.null(processor) && processor %in% exports) {
          processed <- safe(do.call(getExportedValue("microdatasus", processor), list(direct_read$value)))
          processed_dim <- if (processed$ok && is.data.frame(processed$value)) dim(processed$value) else c(NA, NA)
          processed_cols <- if (processed$ok && is.data.frame(processed$value)) names(processed$value) else character()

          json_line(list(
            stage = "process_direct_decoded",
            iteration = i,
            processor = processor,
            ok = processed$ok,
            error = processed$error,
            rows = processed_dim[[1]],
            cols = processed_dim[[2]],
            colnames_head = head(processed_cols, 80)
          ))
        }
      }
    }
  }

  fetch_args <- list(
    year_start = year,
    year_end = year,
    uf = uf,
    information_system = system_id,
    timeout = timeout,
    track_source = TRUE
  )

  if (system_id %in% c("SIH-RD", "CNES-ST")) {
    fetch_args$month_start <- 1
    fetch_args$month_end <- 1
  }

  fetched <- safe(do.call(microdatasus::fetch_datasus, fetch_args))
  fetched_dim <- if (fetched$ok && is.data.frame(fetched$value)) dim(fetched$value) else c(NA, NA)
  fetched_cols <- if (fetched$ok && is.data.frame(fetched$value)) names(fetched$value) else character()

  json_line(list(
    stage = "microdatasus_fetch",
    iteration = i,
    ok = fetched$ok,
    is_null = is.null(fetched$value),
    error = fetched$error,
    rows = fetched_dim[[1]],
    cols = fetched_dim[[2]],
    colnames = fetched_cols,
    source_values = if (fetched$ok && is.data.frame(fetched$value) && "source" %in% names(fetched$value)) unique(fetched$value$source) else NULL
  ))

  if (fetched$ok && is.data.frame(fetched$value)) {
    processor <- switch(system_id,
      "SIM-DO" = "process_sim",
      "SINASC" = "process_sinasc",
      "SIH-RD" = "process_sih",
      "CNES-ST" = "process_cnes",
      NULL
    )

    if (!is.null(processor) && processor %in% exports) {
      processed <- safe(do.call(getExportedValue("microdatasus", processor), list(fetched$value)))
      processed_dim <- if (processed$ok && is.data.frame(processed$value)) dim(processed$value) else c(NA, NA)
      processed_cols <- if (processed$ok && is.data.frame(processed$value)) names(processed$value) else character()

      json_line(list(
        stage = "process_microdatasus_fetched",
        iteration = i,
        processor = processor,
        ok = processed$ok,
        error = processed$error,
        rows = processed_dim[[1]],
        cols = processed_dim[[2]],
        colnames_head = head(processed_cols, 80)
      ))
    }
  }

  json_line(list(stage = "iteration_end", iteration = i))
}