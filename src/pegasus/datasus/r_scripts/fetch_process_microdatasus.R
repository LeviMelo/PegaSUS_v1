# Slice 1A R boundary skeleton.
# This script is intentionally conservative: it defines the subprocess contract,
# heartbeat behavior, and package gate. Full source-specific fetching is implemented
# after Rscript/microdatasus/read.dbc availability is confirmed.

args <- commandArgs(trailingOnly = TRUE)

get_arg <- function(flag, default = NULL) {
  idx <- match(flag, args)
  if (is.na(idx) || idx == length(args)) {
    return(default)
  }
  args[[idx + 1]]
}

out_dir <- get_arg("--out-dir", ".")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

heartbeat_path <- file.path(out_dir, "heartbeat.json")
write_heartbeat <- function(stage, message) {
  payload <- paste0(
    '{"stage":"', stage,
    '","timestamp":"', format(Sys.time(), "%Y-%m-%dT%H:%M:%OS%z"),
    '","rows_so_far":null,"message":"', message, '"}'
  )
  writeLines(payload, heartbeat_path, useBytes = TRUE)
}

write_heartbeat("fetching", "R boundary started")

if (!requireNamespace("microdatasus", quietly = TRUE)) {
  write_heartbeat("done", "microdatasus missing")
  quit(status = 30)
}

if (!requireNamespace("read.dbc", quietly = TRUE)) {
  write_heartbeat("done", "read.dbc missing")
  quit(status = 31)
}

write_heartbeat("done", "Slice 1A boundary reached; full fetch not enabled")
quit(status = 41)
