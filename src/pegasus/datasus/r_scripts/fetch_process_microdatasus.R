args <- commandArgs(trailingOnly = TRUE)

get_arg <- function(flag, default = NULL) {
  idx <- match(flag, args)
  if (is.na(idx) || idx == length(args)) return(default)
  args[[idx + 1]]
}

system_id <- get_arg("--system")
uf <- get_arg("--uf")
year_start <- as.integer(get_arg("--year-start"))
year_end <- as.integer(get_arg("--year-end"))
month_start <- get_arg("--month-start")
month_end <- get_arg("--month-end")
timeout_seconds <- as.integer(get_arg("--timeout-seconds", "900"))
raw_path <- get_arg("--raw-path")
processed_path <- get_arg("--processed-path")
out_dir <- get_arg("--out-dir", ".")

dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(dirname(raw_path), recursive = TRUE, showWarnings = FALSE)
dir.create(dirname(processed_path), recursive = TRUE, showWarnings = FALSE)

heartbeat_path <- file.path(out_dir, "heartbeat.json")

write_heartbeat <- function(stage, rows, message) {
  jsonlite::write_json(
    list(
      stage = stage,
      timestamp = format(Sys.time(), "%Y-%m-%dT%H:%M:%OS%z"),
      rows_so_far = rows,
      message = message
    ),
    heartbeat_path,
    auto_unbox = TRUE,
    pretty = TRUE
  )
}

required <- c("microdatasus", "read.dbc", "arrow", "jsonlite", "dplyr")
missing <- required[!vapply(required, requireNamespace, logical(1), quietly = TRUE)]
if (length(missing) > 0) {
  cat(paste("Missing R packages:", paste(missing, collapse = ", ")), file = stderr())
  quit(status = 30)
}

suppressPackageStartupMessages({
  library(microdatasus)
  library(read.dbc)
  library(arrow)
  library(jsonlite)
  library(dplyr)
})


sanitize_utf8_scalar <- function(x) {
  if (is.na(x)) return(NA_character_)
  y <- as.character(x)
  # DATASUS DBF text occasionally contains embedded NUL bytes. iconv aborts on
  # those before it can apply sub="byte", so strip them at the byte boundary.
  y <- gsub("\\x00", "", y, perl = TRUE, useBytes = TRUE)
  y <- tryCatch({
    bytes <- charToRaw(y)
    if (any(bytes == as.raw(0))) {
      rawToChar(bytes[bytes != as.raw(0)], multiple = FALSE)
    } else {
      y
    }
  }, error = function(e) y)
  Encoding(y) <- "unknown"
  z <- iconv(y, from = "", to = "UTF-8", sub = "byte")
  if (is.na(z)) {
    z <- iconv(y, from = "latin1", to = "UTF-8", sub = "byte")
  }
  if (is.na(z)) {
    z <- ""
  }
  z
}

sanitize_utf8_dataframe <- function(df) {
  if (is.null(df) || !is.data.frame(df)) return(df)
  out <- df
  sanitized_columns <- character()
  for (name in names(out)) {
    col <- out[[name]]
    if (is.character(col) || is.factor(col) || is.object(col)) {
      before <- as.character(col)
      after <- vapply(before, sanitize_utf8_scalar, character(1), USE.NAMES = FALSE)
      after[is.na(before)] <- NA_character_
      out[[name]] <- after
      sanitized_columns <- c(sanitized_columns, name)
    }
  }
  attr(out, "utf8_sanitized_columns") <- unique(sanitized_columns)
  out
}

write_utf8_parquet <- function(df, path) {
  safe <- sanitize_utf8_dataframe(df)
  arrow::write_parquet(safe, path)
  safe
}


information_system <- switch(
  system_id,
  "SIM-DO" = "SIM-DO",
  "SINASC" = "SINASC",
  "SIH-RD" = "SIH-RD",
  "CNES-ST" = "CNES-ST",
  NA
)

if (is.na(information_system)) {
  cat(paste("Unsupported DATASUS system:", system_id), file = stderr())
  quit(status = 20)
}

process_datasus_dispatch <- function(data, system_id) {
  if (!is.data.frame(data) || nrow(data) == 0) {
    stop("Cannot process empty DATASUS data.")
  }

  if (system_id == "SIM-DO") {
    return(process_sim(data, municipality_data = FALSE))
  }

  if (system_id == "SINASC") {
    return(process_sinasc(data, municipality_data = FALSE))
  }

  if (system_id == "SIH-RD") {
    return(process_sih(data, municipality_data = FALSE))
  }

  if (system_id == "CNES-ST") {
    return(process_cnes(data))
  }

  stop(sprintf("No microdatasus processor dispatch for system: %s", system_id))
}

official_dbc_url <- function(system_id, uf, year, month = NULL) {
  if (system_id == "SIM-DO") {
    return(sprintf(
      "ftp://ftp.datasus.gov.br/dissemin/publicos/SIM/CID10/DORES/DO%s%s.dbc",
      uf,
      year
    ))
  }

  if (system_id == "SINASC") {
    return(sprintf(
      "ftp://ftp.datasus.gov.br/dissemin/publicos/SINASC/1996_/Dados/DNRES/DN%s%s.dbc",
      uf,
      year
    ))
  }

  if (system_id == "SIH-RD") {
    if (is.null(month)) stop("SIH-RD direct fallback requires month.")
    return(sprintf(
      "ftp://ftp.datasus.gov.br/dissemin/publicos/SIHSUS/200801_/Dados/RD%s%02d%02d.dbc",
      uf,
      year %% 100,
      month
    ))
  }

  if (system_id == "CNES-ST") {
    if (is.null(month)) stop("CNES-ST direct fallback requires month.")
    return(sprintf(
      "ftp://ftp.datasus.gov.br/dissemin/publicos/CNES/200508_/Dados/ST/ST%s%02d%02d.dbc",
      uf,
      year %% 100,
      month
    ))
  }

  stop(sprintf("No direct official DBC fallback URL for system: %s", system_id))
}

read_direct_dbc <- function(url) {
  dbc_path <- file.path(out_dir, basename(url))
  utils::download.file(
    url,
    dbc_path,
    mode = "wb",
    method = "libcurl",
    quiet = TRUE
  )

  data <- read.dbc::read.dbc(dbc_path, as.is = TRUE)

  if (!("source" %in% names(data))) {
    data$source <- basename(url)
  }

  data
}

direct_fallback_fetch <- function(system_id, uf, year_start, year_end, month_start, month_end) {
  options(timeout = max(timeout_seconds, getOption("timeout")))

  if (system_id %in% c("SIM-DO", "SINASC")) {
    decoded <- lapply(seq.int(year_start, year_end), function(year) {
      url <- official_dbc_url(system_id, uf, year)
      read_direct_dbc(url)
    })
    return(dplyr::bind_rows(decoded))
  }

  if (system_id %in% c("SIH-RD", "CNES-ST")) {
    if (is.null(month_start) || is.null(month_end)) {
      stop(sprintf("%s fallback requires month_start and month_end.", system_id))
    }

    decoded <- list()
    idx <- 1L

    for (year in seq.int(year_start, year_end)) {
      start_m <- if (year == year_start) as.integer(month_start) else 1L
      end_m <- if (year == year_end) as.integer(month_end) else 12L

      for (month in seq.int(start_m, end_m)) {
        url <- official_dbc_url(system_id, uf, year, month)
        decoded[[idx]] <- read_direct_dbc(url)
        idx <- idx + 1L
      }
    }

    return(dplyr::bind_rows(decoded))
  }

  stop(sprintf("Direct fallback not implemented for system: %s", system_id))
}

write_heartbeat("fetching", 0, "microdatasus acquisition started")

fetch_args <- list(
  year_start = year_start,
  year_end = year_end,
  uf = uf,
  information_system = information_system,
  timeout = timeout_seconds,
  track_source = TRUE
)

if (!is.null(month_start)) fetch_args$month_start <- as.integer(month_start)
if (!is.null(month_end)) fetch_args$month_end <- as.integer(month_end)

raw <- NULL
acquisition_transport <- "microdatasus_fetch_datasus"
fetch_error <- NULL

raw <- tryCatch(
  do.call(fetch_datasus, fetch_args),
  error = function(e) {
    fetch_error <<- conditionMessage(e)
    NULL
  }
)

if (is.null(raw) || !is.data.frame(raw) || nrow(raw) == 0) {
  if (system_id %in% c("SIM-DO", "SINASC", "SIH-RD", "CNES-ST")) {
    write_heartbeat("fallback", 0, "microdatasus fetch failed or returned empty; trying direct official DBC fallback")
    acquisition_transport <- "direct_official_ftp_dbc_fallback"

    raw <- tryCatch(
      direct_fallback_fetch(system_id, uf, year_start, year_end, month_start, month_end),
      error = function(e) {
        cat(
          paste(
            "DATASUS direct fallback failed after microdatasus fetch failure.",
            "microdatasus_error=",
            ifelse(is.null(fetch_error), "NULL_or_empty", fetch_error),
            "fallback_error=",
            conditionMessage(e)
          ),
          file = stderr()
        )
        quit(status = 40)
      }
    )
  } else {
    cat(
      paste(
        "microdatasus fetch returned no records and no direct fallback exists.",
        "microdatasus_error=",
        ifelse(is.null(fetch_error), "NULL_or_empty", fetch_error)
      ),
      file = stderr()
    )
    quit(status = 40)
  }
}

if (is.null(raw) || !is.data.frame(raw) || nrow(raw) == 0) {
  cat("DATASUS fetch returned no records", file = stderr())
  quit(status = 40)
}

saveRDS(raw, raw_path)

write_heartbeat("processing", nrow(raw), "raw coded acquisition complete; writing canonical raw-coded Parquet")

canonical_raw <- raw
canonical_raw <- write_utf8_parquet(canonical_raw, processed_path)

microdatasus_processed_path <- file.path(out_dir, "microdatasus_processed.parquet")
microdatasus_processing_status <- "not_attempted"
microdatasus_processing_error <- NULL
microdatasus_processed_rows <- NA_integer_
microdatasus_processed_columns <- character()

processed_semantic <- tryCatch(
  process_datasus_dispatch(raw, system_id),
  error = function(e) {
    microdatasus_processing_error <<- conditionMessage(e)
    NULL
  }
)

if (!is.null(processed_semantic) && is.data.frame(processed_semantic) && nrow(processed_semantic) > 0) {
  processed_semantic <- write_utf8_parquet(processed_semantic, microdatasus_processed_path)
  microdatasus_processing_status <- "success"
  microdatasus_processed_rows <- nrow(processed_semantic)
  microdatasus_processed_columns <- names(processed_semantic)
} else {
  microdatasus_processing_status <- "failed"
}

rows <- nrow(canonical_raw)
columns <- names(canonical_raw)

manifest <- list(
  status = "success",
  system = system_id,
  uf = uf,
  raw_path = normalizePath(raw_path, winslash = "/", mustWork = TRUE),
  processed_path = normalizePath(processed_path, winslash = "/", mustWork = TRUE),
  microdatasus_processed_path = if (file.exists(microdatasus_processed_path)) {
    normalizePath(microdatasus_processed_path, winslash = "/", mustWork = TRUE)
  } else {
    NULL
  },
  row_counts = list(
    raw = nrow(raw),
    processed = rows,
    canonical_raw = rows,
    microdatasus_processed = ifelse(is.na(microdatasus_processed_rows), 0L, microdatasus_processed_rows)
  ),
  column_lists = list(
    raw = names(raw),
    processed = columns,
    canonical_raw = columns,
    microdatasus_processed = microdatasus_processed_columns
  ),
  r_version = R.version.string,
  microdatasus_version = as.character(utils::packageVersion("microdatasus")),
  read_dbc_version = as.character(utils::packageVersion("read.dbc")),
  acquisition_transport = acquisition_transport,
  fetch_error = fetch_error,
  processing_contract_version = "datasus_r_bridge_v3_utf8_sanitized_raw_canonical_plus_microdatasus_sidecar",
  processing_mode = "canonical_raw_codes_with_microdatasus_processed_sidecar",
  canonical_processed_role = "raw_codes_for_python_normalizer",
  utf8_sanitization = list(
    applied = TRUE,
    canonical_raw_columns = as.list(attr(canonical_raw, "utf8_sanitized_columns")),
    microdatasus_processed_columns = if (!is.null(processed_semantic) && is.data.frame(processed_semantic)) {
      as.list(attr(processed_semantic, "utf8_sanitized_columns"))
    } else {
      list()
    }
  ),
  microdatasus_processor = switch(
    system_id,
    "SIM-DO" = "process_sim",
    "SINASC" = "process_sinasc",
    "SIH-RD" = "process_sih",
    "CNES-ST" = "process_cnes",
    NA
  ),
  microdatasus_processing_status = microdatasus_processing_status,
  microdatasus_processing_error = microdatasus_processing_error
)

jsonlite::write_json(
  manifest,
  file.path(out_dir, "manifest.json"),
  auto_unbox = TRUE,
  pretty = TRUE,
  null = "null"
)

write_heartbeat("done", rows, "materialization complete")
