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
raw_path <- get_arg("--raw-path")
processed_path <- get_arg("--processed-path")
out_dir <- get_arg("--out-dir", ".")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(dirname(raw_path), recursive = TRUE, showWarnings = FALSE)
dir.create(dirname(processed_path), recursive = TRUE, showWarnings = FALSE)

heartbeat_path <- file.path(out_dir, "heartbeat.json")
write_heartbeat <- function(stage, rows, message) {
  jsonlite::write_json(
    list(stage = stage, timestamp = format(Sys.time(), "%Y-%m-%dT%H:%M:%OS%z"), rows_so_far = rows, message = message),
    heartbeat_path, auto_unbox = TRUE, pretty = TRUE
  )
}

required <- c("microdatasus", "read.dbc", "arrow", "jsonlite")
missing <- required[!vapply(required, requireNamespace, logical(1), quietly = TRUE)]
if (length(missing) > 0) {
  cat(paste("Missing R packages:", paste(missing, collapse = ", ")), file = stderr())
  quit(status = 30)
}

write_heartbeat("fetching", 0, "microdatasus acquisition started")
information_system <- switch(system_id, "SIM-DO" = "SIM-DO", "SINASC" = "SINASC", "SIH-RD" = "SIH-RD", "CNES-ST" = "CNES-ST", NA)
if (is.na(information_system)) quit(status = 20)

fetch_args <- list(
  year_start = year_start, year_end = year_end, uf = uf,
  information_system = information_system
)
if (!is.null(month_start)) fetch_args$month_start <- as.integer(month_start)
if (!is.null(month_end)) fetch_args$month_end <- as.integer(month_end)

raw <- tryCatch(do.call(microdatasus::fetch_datasus, fetch_args), error = function(e) {
  cat(conditionMessage(e), file = stderr())
  quit(status = 40)
})
saveRDS(raw, raw_path)
write_heartbeat("processing", if (is.data.frame(raw)) nrow(raw) else 0, "raw acquisition complete")

processed <- tryCatch(
  microdatasus::process_datasus(raw, information_system = information_system),
  error = function(e) {
    cat(conditionMessage(e), file = stderr())
    quit(status = 50)
  }
)
arrow::write_parquet(processed, processed_path)
rows <- if (is.data.frame(processed)) nrow(processed) else 0
columns <- if (is.data.frame(processed)) names(processed) else character()

manifest <- list(
  status = "success", system = system_id, uf = uf,
  raw_path = normalizePath(raw_path, winslash = "/", mustWork = TRUE),
  processed_path = normalizePath(processed_path, winslash = "/", mustWork = TRUE),
  row_counts = list(raw = if (is.data.frame(raw)) nrow(raw) else 0, processed = rows),
  column_lists = list(raw = if (is.data.frame(raw)) names(raw) else character(), processed = columns),
  r_version = R.version.string,
  microdatasus_version = as.character(utils::packageVersion("microdatasus")),
  read_dbc_version = as.character(utils::packageVersion("read.dbc"))
)
jsonlite::write_json(manifest, file.path(out_dir, "manifest.json"), auto_unbox = TRUE, pretty = TRUE)
write_heartbeat("done", rows, "materialization complete")
