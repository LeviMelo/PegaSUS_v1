#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = TRUE)

get_arg <- function(name, default = NA_character_) {
  key <- paste0("--", name)
  idx <- which(args == key)
  if (length(idx) == 0) return(default)
  if (idx[length(idx)] >= length(args)) return(default)
  args[[idx[length(idx)] + 1]]
}

null_if_na <- function(x) {
  if (length(x) == 0) return(NULL)
  if (is.null(x)) return(NULL)
  if (length(x) == 1 && is.na(x)) return(NULL)
  x
}

write_jsonlite <- function(obj, path) {
  if (!requireNamespace("jsonlite", quietly = TRUE)) {
    stop("R package 'jsonlite' is required")
  }
  jsonlite::write_json(obj, path, auto_unbox = TRUE, pretty = TRUE, null = "null")
}

write_heartbeat <- function(out_dir, stage, message = "", rows_so_far = NULL) {
  hb <- list(
    stage = stage,
    timestamp = format(Sys.time(), "%Y-%m-%dT%H:%M:%OS6%z"),
    rows_so_far = null_if_na(rows_so_far),
    message = message
  )
  path <- file.path(out_dir, "heartbeat.json")
  try(write_jsonlite(hb, path), silent = TRUE)
}

safe_package_version <- function(pkg) {
  tryCatch(as.character(utils::packageVersion(pkg)), error = function(e) NA_character_)
}

normalize_system <- function(system) {
  system <- toupper(trimws(system))
  aliases <- list(
    "SIM" = "SIM-DO",
    "SIM-DO" = "SIM-DO",
    "DO" = "SIM-DO",
    "SINASC" = "SINASC",
    "SIH" = "SIH-RD",
    "SIH-RD" = "SIH-RD",
    "CNES" = "CNES-ST",
    "CNES-ST" = "CNES-ST"
  )
  if (!system %in% names(aliases)) {
    stop(paste0("Unsupported DATASUS system: ", system))
  }
  aliases[[system]]
}

process_function_name <- function(system) {
  switch(
    system,
    "SIM-DO" = "process_sim",
    "SINASC" = "process_sinasc",
    "SIH-RD" = "process_sih",
    "CNES-ST" = "process_cnes",
    stop(paste0("Unsupported process function for system: ", system))
  )
}

process_microdatasus <- function(raw_df, system) {
  if (system == "SIM-DO") {
    return(process_sim(raw_df))
  }

  if (system == "SINASC") {
    return(process_sinasc(raw_df))
  }

  if (system == "SIH-RD") {
    return(process_sih(raw_df))
  }

  if (system == "CNES-ST") {
    return(process_cnes(raw_df))
  }

  stop(paste0("Unsupported process function for system: ", system))
}

fetch_microdatasus <- function(system, uf, year_start, month_start, year_end, month_end) {
  fetch_args <- list(
    year_start = year_start,
    year_end = year_end,
    uf = uf,
    information_system = system
  )

  if (!is.na(month_start) && !is.na(month_end)) {
    fetch_args$month_start <- month_start
    fetch_args$month_end <- month_end
  }

  do.call(fetch_datasus, fetch_args)
}

reinforce_processed_columns <- function(raw_df, processed_df) {
  raw_names <- names(raw_df)
  processed_names <- names(processed_df)
  missing_names <- setdiff(raw_names, processed_names)

  if (length(missing_names) == 0) {
    return(list(data = processed_df, reinjected = character(0), skipped = character(0)))
  }

  if (nrow(raw_df) != nrow(processed_df)) {
    return(list(data = processed_df, reinjected = character(0), skipped = missing_names))
  }

  for (name in missing_names) {
    processed_df[[name]] <- raw_df[[name]]
  }

  return(list(data = processed_df, reinjected = missing_names, skipped = character(0)))
}

write_manifest <- function(
  out_dir,
  status,
  error_message = NULL,
  system,
  uf,
  year_start,
  month_start,
  year_end,
  month_end,
  raw_path = NULL,
  processed_path = NULL,
  raw_df = NULL,
  processed_df = NULL,
  fetch_function = "fetch_datasus",
  process_function = NULL,
  started_at,
  completed_at,
  reinjected_processed_columns = character(0),
  skipped_reinjection_columns = character(0)
) {
  raw_columns <- if (is.null(raw_df)) character(0) else colnames(raw_df)
  processed_columns <- if (is.null(processed_df)) character(0) else colnames(processed_df)

  manifest <- list(
    backend = "microdatasus",
    system = system,
    information_system = system,
    uf = uf,
    year_start = year_start,
    month_start = null_if_na(month_start),
    year_end = year_end,
    month_end = null_if_na(month_end),
    fetch_function = fetch_function,
    process_function = process_function,
    raw_path = null_if_na(raw_path),
    processed_path = null_if_na(processed_path),
    raw_rows = if (is.null(raw_df)) NULL else nrow(raw_df),
    raw_cols = if (is.null(raw_df)) NULL else ncol(raw_df),
    processed_rows = if (is.null(processed_df)) NULL else nrow(processed_df),
    processed_cols = if (is.null(processed_df)) NULL else ncol(processed_df),
    raw_columns = raw_columns,
    processed_columns = processed_columns,
    reinjected_processed_columns = reinjected_processed_columns,
    skipped_reinjection_columns = skipped_reinjection_columns,
    r_version = paste(R.version$major, R.version$minor, sep = "."),
    microdatasus_version = safe_package_version("microdatasus"),
    read_dbc_version = safe_package_version("read.dbc"),
    jsonlite_version = safe_package_version("jsonlite"),
    created_at = format(Sys.time(), "%Y-%m-%dT%H:%M:%OS6%z"),
    started_at = started_at,
    completed_at = completed_at,
    status = status,
    error_message = null_if_na(error_message)
  )

  write_jsonlite(manifest, file.path(out_dir, "manifest.json"))
}

exit_with_manifest <- function(code, out_dir, system, uf, year_start, month_start, year_end, month_end, message, started_at) {
  completed_at <- format(Sys.time(), "%Y-%m-%dT%H:%M:%OS6%z")
  try(
    write_manifest(
      out_dir = out_dir,
      status = "error",
      error_message = message,
      system = system,
      uf = uf,
      year_start = year_start,
      month_start = month_start,
      year_end = year_end,
      month_end = month_end,
      started_at = started_at,
      completed_at = completed_at
    ),
    silent = TRUE
  )
  cat(message, "\n", file = stderr())
  quit(status = code, save = "no")
}

system_arg <- get_arg("system")
uf <- toupper(get_arg("uf"))
year_start <- suppressWarnings(as.integer(get_arg("year-start")))
year_end <- suppressWarnings(as.integer(get_arg("year-end")))
month_start <- suppressWarnings(as.integer(get_arg("month-start", NA_character_)))
month_end <- suppressWarnings(as.integer(get_arg("month-end", NA_character_)))
out_dir <- get_arg("out-dir")
started_at <- format(Sys.time(), "%Y-%m-%dT%H:%M:%OS6%z")

if (is.na(system_arg) || is.na(uf) || is.na(year_start) || is.na(year_end) || is.na(out_dir)) {
  cat("Invalid input arguments.\n", file = stderr())
  quit(status = 10, save = "no")
}

dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

system <- tryCatch(
  normalize_system(system_arg),
  error = function(e) {
    write_heartbeat(out_dir, "failed", conditionMessage(e))
    cat(conditionMessage(e), "\n", file = stderr())
    quit(status = 11, save = "no")
  }
)

if (year_end < year_start) {
  exit_with_manifest(
    12, out_dir, system, uf, year_start, month_start, year_end, month_end,
    "Invalid year range.",
    started_at
  )
}

if ((!is.na(month_start) && (month_start < 1 || month_start > 12)) ||
    (!is.na(month_end) && (month_end < 1 || month_end > 12))) {
  exit_with_manifest(
    12, out_dir, system, uf, year_start, month_start, year_end, month_end,
    "Invalid month range.",
    started_at
  )
}

if (!requireNamespace("jsonlite", quietly = TRUE)) {
  cat("R package 'jsonlite' is not installed.\n", file = stderr())
  quit(status = 30, save = "no")
}

if (!requireNamespace("microdatasus", quietly = TRUE)) {
  exit_with_manifest(
    30, out_dir, system, uf, year_start, month_start, year_end, month_end,
    "R package 'microdatasus' is not installed.",
    started_at
  )
}

suppressPackageStartupMessages(library(microdatasus))

raw_path <- file.path(out_dir, "raw.csv")
processed_path <- file.path(out_dir, "processed.csv")
process_function <- process_function_name(system)

write_heartbeat(out_dir, "fetching", "Calling fetch_datasus")

raw_df <- tryCatch(
  fetch_microdatasus(system, uf, year_start, month_start, year_end, month_end),
  error = function(e) {
    exit_with_manifest(
      30, out_dir, system, uf, year_start, month_start, year_end, month_end,
      paste0("microdatasus fetch failure: ", conditionMessage(e)),
      started_at
    )
  }
)

write_heartbeat(out_dir, "writing_raw", "Writing raw.csv", nrow(raw_df))

tryCatch(
  utils::write.csv(raw_df, raw_path, row.names = FALSE, fileEncoding = "UTF-8", na = ""),
  error = function(e) {
    exit_with_manifest(
      32, out_dir, system, uf, year_start, month_start, year_end, month_end,
      paste0("raw output write failure: ", conditionMessage(e)),
      started_at
    )
  }
)

write_heartbeat(out_dir, "processing", paste0("Calling ", process_function), nrow(raw_df))

processed_df <- tryCatch(
  process_microdatasus(raw_df, system),
  error = function(e) {
    completed_at <- format(Sys.time(), "%Y-%m-%dT%H:%M:%OS6%z")

    write_manifest(
      out_dir = out_dir,
      status = "process_error",
      error_message = paste0("microdatasus process failure: ", conditionMessage(e)),
      system = system,
      uf = uf,
      year_start = year_start,
      month_start = month_start,
      year_end = year_end,
      month_end = month_end,
      raw_path = raw_path,
      processed_path = NULL,
      raw_df = raw_df,
      processed_df = NULL,
      process_function = process_function,
      started_at = started_at,
      completed_at = completed_at
    )

    write_heartbeat(
      out_dir,
      "process_error_raw_preserved",
      paste0("Raw data written; process step failed: ", conditionMessage(e)),
      nrow(raw_df)
    )

    cat(paste0("microdatasus process failure: ", conditionMessage(e), "\n"), file = stderr())
    quit(status = 31, save = "no")
  }
)

reinforcement <- reinforce_processed_columns(raw_df, processed_df)
processed_df <- reinforcement$data

write_heartbeat(out_dir, "writing_processed", "Writing processed.csv", nrow(processed_df))

tryCatch(
  utils::write.csv(processed_df, processed_path, row.names = FALSE, fileEncoding = "UTF-8", na = ""),
  error = function(e) {
    exit_with_manifest(
      33, out_dir, system, uf, year_start, month_start, year_end, month_end,
      paste0("processed output write failure: ", conditionMessage(e)),
      started_at
    )
  }
)

completed_at <- format(Sys.time(), "%Y-%m-%dT%H:%M:%OS6%z")

write_manifest(
  out_dir = out_dir,
  status = "ok",
  error_message = NULL,
  system = system,
  uf = uf,
  year_start = year_start,
  month_start = month_start,
  year_end = year_end,
  month_end = month_end,
  raw_path = raw_path,
  processed_path = processed_path,
  raw_df = raw_df,
  processed_df = processed_df,
  process_function = process_function,
  started_at = started_at,
  completed_at = completed_at,
  reinjected_processed_columns = reinforcement$reinjected,
  skipped_reinjection_columns = reinforcement$skipped
)

write_heartbeat(out_dir, "done", "microdatasus fetch/process complete", nrow(processed_df))

cat("raw_rows=", nrow(raw_df), "\n", sep = "")
cat("processed_rows=", nrow(processed_df), "\n", sep = "")
cat("raw_path=", raw_path, "\n", sep = "")
cat("processed_path=", processed_path, "\n", sep = "")
cat("reinjected_processed_columns=", paste(reinforcement$reinjected, collapse = ","), "\n", sep = "")

quit(status = 0, save = "no")