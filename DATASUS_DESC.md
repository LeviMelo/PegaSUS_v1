1. DATASUS acquisition is microdatasus-first, not FTP-first
The settled production decision was: for the epidemiological core, use microdatasus as the primary DATASUS retrieval and preprocessing backend. microdatasus exists precisely to download and preprocess DATASUS microdata files in DBC format; its documentation describes it as an R package for downloading and preprocessing DATASUS microdata, including assigning/treating categorical variable labels during preprocessing. (GitHub)
The package exposes the basic split we need:
fetch_datasus(...)
  → download/read raw DATASUS microdata

process_sim(...)
process_sih(...)
process_sinasc(...)
process_cnes(...)
process_sia(...)
process_sinan_dengue(...)
...
  → process/label system-specific returned data

The reference index lists processing functions for SIM, SINASC, SIH, SIA, CNES, and several SINAN disease modules. (Raphael Saldanha) For example, process_sim(data, municipality_data = TRUE) processes SIM variables retrieved by fetch_datasus() and adds labels for categorical variables, including NA values; the municipality_data argument can create additional municipality-residence detail variables. (Raphael Saldanha)
So the clean strategy is not to reimplement DATASUS file retrieval for the core systems unless strictly necessary. Use microdatasus for acquisition and initial processing, then build PegaSUS-specific cache, profiling, normalization, provenance, and legality logic around it.
2. The core DATASUS systems
The core systems we fixed for the first production substrate were:
SIM-DO   → mortality / death certificates
SIH-RD   → hospitalizations / AIH reduced records
SINASC   → live births
CNES-ST  → health establishments / service supply / facility registry

Later SINAN disease modules can be supported, but the generic core should not start by trying to absorb every SINAN disease or every DATASUS FTP family. The narrower core is enough to support mortality, hospitalization, birth, and facility/capacity layers.
The broader project description repeatedly positioned microdatasus and PySUS as access/preprocessing tools, while PegaSUS occupies the next semantic layer: typing epidemiological objects, enforcing numerator/denominator compatibility, generating indicators, and creating reusable analytical substrates.
3. The two DATASUS strategies we considered
There were two distinct strategies in the project history.
The first was a broad, generic DATASUS FTP discovery apparatus:
FTP scan
→ inventory
→ dataset-family inference
→ family registry with member_files
→ candidate-file selection
→ selective download
→ decode/profile
→ variable catalog
→ similarity reports
→ translation bundles/registry

This strategy was coherent for surveying the entire DATASUS universe. The older project note describes the DATASUS subsystem as organized into FTP scanning, inventory/family inference, selective fetch, decode, profile, translate, and discovery utilities, rather than as a parser per known dataset. It also records the corrected operational bridge: family registry → candidate selection → selective download → profiling, with member_files preserved as the actual FTP file paths.
The second, later fixed strategy was simpler and more production-oriented:
microdatasus fetch/process for SIM-DO, SIH-RD, SINASC, CNES-ST
→ cache raw and processed outputs separately
→ profile actual returned columns
→ compare raw vs processed
→ normalize into event/facility substrates
→ expose explicit join affordances

For the rewrite, the second strategy should be the default. The FTP-family discovery apparatus should be optional, not the first architecture. It is useful if the project later needs arbitrary DATASUS families beyond the core, but it should not distract the first production ingestion layer.
4. What microdatasus gives us — and what it does not give us
microdatasus gives us:
direct access to DATASUS microdata
DBC reading/conversion hidden behind fetch_datasus()
system-specific process_* functions
categorical labeling/relabeling
some municipality helper fields in processing functions
a stable R interface around common DATASUS systems

It does not give us the full PegaSUS substrate. We still need:
project-managed caching
raw/processed separation
request manifests
schema profiling
raw vs processed comparison
join-key discovery
municipality-code harmonization
facility-identifier handling
missingness preservation
measurement-process metadata
event/facility normalization
lineage/provenance

In other words, microdatasus is the acquisition and first-pass preprocessing backend, not the epidemiological compiler.
5. Python ↔ R orchestration lesson
Since microdatasus is an R package and the main PegaSUS rewrite may be Python-centered, the clean orchestration is:
Python service
→ writes/executes Rscript call
→ R loads microdatasus
→ R calls fetch_datasus()
→ R saves raw data.frame
→ R calls process_*()
→ R saves processed data.frame
→ Python profiles, compares, normalizes, and stores outputs

The Python layer should not try to mimic microdatasus internals. It should call it through a controlled boundary. The bridge should be a proper module, not a throwaway script.
Suggested module:
src/minipegasus/datasus/client_microdatasus.py

Responsibilities:
build R command
choose microdatasus information_system
pass year/month/UF parameters
run Rscript
collect raw/processed file paths
capture stdout/stderr
write manifest
report failure cleanly

This also isolates the R dependency. If Rscript or microdatasus is missing, the error belongs in this client layer, not throughout the pipeline.
6. Installation/environment lessons
We saw that installing microdatasus on Windows may compile/install read.dbc, and logs showed read.dbc and microdatasus_2.5.0 being built/installed locally. The install was possible, but the dependency chain is nontrivial on Windows because it involves R, Rtools/build tooling, compiled code, and package libraries.
Therefore, the ingestion layer should record environment metadata when possible:
R version
microdatasus version
read.dbc version, if available
OS
timestamp
command arguments
fetch parameters

A future run must be reproducible enough to know which package version produced a processed file.
7. Core microdatasus request contract
For each request, preserve the full acquisition tuple:
system
UF
year_start
month_start
year_end
month_end
information_system
microdatasus function used
raw output path
processed output path
row counts
column lists
timestamp
R/microdatasus version
stdout/stderr log path
status
error message if any

For example:
system=SIM-DO
uf=AL
period=2022-01..2022-12
information_system=SIM-DO
fetch_function=fetch_datasus
process_function=process_sim
raw_path=data/raw/datasus/SIM-DO/uf=AL/2022_01__2022_12/<hash>/raw.parquet
processed_path=data/raw/datasus/SIM-DO/uf=AL/2022_01__2022_12/<hash>/processed.parquet

The hash should be based on the full request tuple, not just system/year.
8. Mandatory dual-layer cache: raw and processed
This was one of the major lessons.
The local data lake must preserve both:
raw cache
processed cache

Do not overwrite raw with processed.
Reason: process_*() functions can recode, label, convert blanks to NA, create derived columns, attach municipality information, and sometimes produce fields that are not simple one-to-one decodes of raw fields. The project memo states this directly: wrappers like microdatasus may silently recode, drop rows, relabel nulls, or merge categorical semantics, so a measurement-aware compiler must preserve raw missingness and formatting artifacts.
Correct chain:
fetch_datasus()
→ raw dataframe
→ save raw immutable output

process_*()
→ processed dataframe
→ save processed immutable output

profile(raw)
profile(processed)
compare(raw, processed)
normalize downstream with explicit rules

Never use only processed outputs as the scientific source of truth.
9. Processed ≠ decoded raw
This became a central warning.
We initially treated processed data as if it were just raw data with labels. That is unsafe. The later compendium audits showed cases where processed variables could contain mixed code/name values, null conversions, extra derived geography columns, altered category distributions, or non-identical value domains. The design notes explicitly preserve the rule: do not assume processed/raw schemas are trivially equivalent.
Therefore, the schema comparison must explicitly report:
raw-only columns
processed-only columns
common columns with changed missingness
common columns with changed cardinality
common columns with changed observed kind
common columns with changed top-value distribution
columns with fewer nonblank values but more unique processed values
binary-like columns with empirical outliers

But these are empirical warnings, not automatic proof of corruption. Processing may intentionally add columns or convert filler codes to missing.
10. Empirical profiling is mandatory
For every raw and processed file, profile every column.
Minimum profile:
column name
dtype observed
n_rows
n_blank_or_na
missing_rate
n_nonblank
n_unique
unique_rate
top values
full category list if low-cardinality
numeric parse rate
numeric min/max if appropriate
date parse rate
date min/max if appropriate
value length distribution
digit length distribution
shape flags
name-based role
observed kind

This distinction became important:
name-based role
≠
shape evidence

A column can be seven digits and still not be a municipality code. Conversely, a municipality field should be recognized by its column name and system context, not by digit length alone.
11. Do not classify codes by digit length alone
This was one of the clearest mistakes.
A 7-digit value can be:
CNES facility ID
municipality code
some other administrative code
numeric-looking string

A 14-digit value can be:
CNPJ
zero filler
other long identifier

A 9- or 10-digit value can look like a procedure code but might be another administrative identifier.
So the profiler should report shape flags like:
digit_code_like
digit_lengths=7
ICD10_shape_like
contains_prefixed_asterisk_values

but it should not turn those into final semantic types unless the variable name and system context support it.
12. Name-based roles we learned to distinguish
Important namespace distinctions:
CNES
CODESTAB
CO_CNES
CNES_* 
  → facility / establishment identifier

CODMUNRES
CODMUNOCOR
CODMUNNASC
CODUFMUN
MUNIC_RES
MUNIC_MOV
  → municipality fields

CGC_HOSP
CNPJ_MANT
CPF-like fields
  → corporate/person identifier fields, not geography

CAUSABAS
CAUSABAS_O
LINHAA/LINHAB/LINHAC/LINHAD/LINHAII
DIAG_PRINC
DIAG_SECUN
CID_*
  → diagnosis/cause code fields

PROC_*
PROC_REA
PROC_SOLIC
  → procedure code fields

VAL_*
  → monetary/numeric measure fields

The “facility join” issue was especially important. The notes warn that the DATASUS/CNES join model is not a simple relational schema. SIM-DO CODESTAB can have substantial missingness; SIH-RD may expose CGC_HOSP/CNPJ-like hospital identifiers rather than a direct CNES field in the profiled schema; and zero-filled CNPJ values such as 00000000000000 can create catastrophic false matches if used in joins.
13. Facility joins are empirical affordances, not guaranteed foreign keys
Do not implement:
all health events INNER JOIN CNES ON event.CNES = cnes.CNES

That is wrong.
Facility linkage should be dynamic and guarded:
Path A — direct facility ID
  if CODESTAB/CNES-like field exists and is valid:
      join to CNES-ST on facility identifier

Path B — corporate identifier bridge
  if only CGC_HOSP/CNPJ-like fields exist:
      use filtered CNPJ ↔ CNES crosswalk
      remove zero fillers and invalid identifiers

Path C — spatial fallback
  if facility ID is missing/invalid:
      keep event
      attach occurrence or movement municipality
      mark reduced spatial precision

Records must not be dropped just because facility ID is missing. Preserve the event at municipality level and mark precision loss.
This was later summarized as “treat facility joins as empirical affordances, not guaranteed foreign keys.”
14. Municipality-code harmonization
DATASUS often uses 6-digit municipality codes, while IBGE/SIDRA and other substrates may use 7-digit municipality codes. The design notes recommend precomputing a crosswalk once, ideally from geobr or another validated source, and storing it in DuckDB/Arrow/Parquet rather than recalculating verifier digits row-by-row during ingestion.
Minimum crosswalk:
cod6
cod7
name
uf
valid_from?
valid_to?
source

Use vectorized joins, not Python loops inside ingestion.
15. Date parsing lessons
We hit a concrete bug: applying pd.to_datetime(..., dayfirst=True) generically to every column produced warning floods and even crashed on impossible/out-of-range timestamps such as year 0.
Correct rule:
Never run generic dateutil parsing across all DATASUS columns.

Date parsing should only run when:
column name suggests date/period semantics
or observed value shape strongly suggests date

Supported formats:
YYYY-MM-DD
YYYY-MM-DD HH:MM:SS
DD/MM/YYYY
DD-MM-YYYY
YYYY/MM/DD
YYYYMMDD
DDMMYYYY

Invalid placeholders must be coerced:
00000000
0000-00-00
00/00/0000
99999999
9999-99-99
0
blank

Numeric identifiers must not be parsed as dates merely because they have 8 digits.
16. Missingness must be preserved
Missingness is not just nuisance. It is a measurement-process signal.
For DATASUS, missingness can reflect:
data-entry workflow
facility administrative quality
system migration
coding practices
race/color observation bias
field not applicable
structural absence
filler code converted to NA

The project’s later context-transfer memo states this in stronger terms: missing race data becomes an observer-process variable and can be scientifically meaningful, not merely something to impute away.
For ingestion, this means:
preserve raw missing/filler values
record processed missingness
record missingness changes raw→processed
expose missingness as variables/quality flags where relevant

Do not erase missingness by default.
17. Race/color fields are measurement-process fields
DATASUS race/color fields are not interchangeable with IBGE self-declared race/color denominators.
Relevant source processes:
SIM-DO   → administrative death declaration / physician/coroner-mediated
SIH-RD   → hospital billing/administrative record
SINASC   → mixed administrative/maternal/hospital documentation
IBGE     → self-declared census denominator

The project memorandum states that operations naively dividing DATASUS administrative race numerators by IBGE self-declared denominators must be blocked unless routed through a race-axis bridge.
For the DATASUS fetch layer, the concrete implication is simpler:
tag race/color fields by source system and measurement process
preserve missingness
do not relabel them as “true race”
do not silently combine with IBGE denominator categories

18. ICD-10 field lesson
We learned not to generalize ICD quirks.
For SIM-DO, prefixed asterisks may appear in certificate-line fields:
LINHAA
LINHAB
LINHAC
LINHAD
LINHAII

Do not claim that CAUSABAS or CAUSABAS_O contain literal asterisks unless the specific profile shows it. The profiler should report asterisk occurrence per field.
So:
field-level observation, not system-level blanket claim

19. Binary flags with outliers
Some fields look like binary flags but empirically contain values outside {0,1} or {Sim,Não}.
Correct language:
nominally binary
binary-like observed
nominally binary with empirical outliers

Incorrect language:
strict 0/1 variable
boolean variable

unless observed domain is truly strict.
This mattered especially in CNES-like fields, where raw values may include unexpected positive integers beyond 1.
20. Column-shift / processing anomaly checks
We also learned that the profiler should support cross-variable and raw/processed sanity checks, but these should be framed as “inspect” flags, not automatic conclusions.
Examples:
GESTRISCO positive-like count > female-like count in SEXO
  → if interpreted as pregnancy-positive flag, verify coding/processing

KOTELCHUCK vs CONSULTAS distribution mismatch
  → treat as observed/processed field; do not assume derivation correctness

processed has fewer nonblank values but more unique values than raw
  → processed is not a simple decode; inspect

raw category N converted to missing in processed but missingness not changed
  → check whether category omitted from top-value display or recoding logic

The lesson is not “these fields are corrupt.” The lesson is that a compiler must not silently accept processed outputs as semantically clean.
21. Value status / quality state for DATASUS
DATASUS ingestion should eventually produce, for every variable or derived field, a quality state. The later project context uses Q_tensor as the central quality/safety state and says it should not be decorative metadata.
For the fetch/profile layer, quality inputs include:
raw_missing_rate
processed_missing_rate
raw_processed_missing_delta
low sample size
unexpected category domain
binary outliers
invalid dates
identifier filler share
race missingness
join precision loss
facility linkage availability
municipality crosswalk failure

These should feed later statuses such as:
verified
fragile
quarantined_descriptive

The fetch layer does not decide epidemiological truth; it provides the evidence.
22. Storage format and local backend
The preferred backend direction that emerged was:
Parquet / Arrow
DuckDB
Polars
GeoParquet where geometry is needed

The notes explicitly recommend DuckDB/Polars/Arrow as the main backend path in the rewrite.
For DATASUS:
raw and processed microdatasus outputs → Parquet
profiles/comparisons → Parquet/JSON
normalized event/facility tables → Parquet partitioned by system/uf/year/month
metadata manifests → JSON

CSV is acceptable for inspection, but not as the main production substrate.
23. Suggested folder layout
A clean microdatasus cache:
data/
  raw/
    datasus/
      SIM-DO/
        uf=AL/
          2022_01__2022_12/
            <request_hash>/
              raw.parquet
              processed.parquet
              manifest.json
              r_stdout.log
              r_stderr.log

      SIH-RD/
      SINASC/
      CNES-ST/

  metadata/
    datasus/
      schema_profiles/
      raw_processed_comparisons/
      variable_catalog/
      request_index.parquet

  processed/
    datasus/
      normalized_events/
      normalized_facilities/

Do not mix temporary probe files with production cache.
24. Request manifest fields
Each request manifest should include:
{
  "backend": "microdatasus",
  "information_system": "SIM-DO",
  "uf": "AL",
  "year_start": 2022,
  "month_start": 1,
  "year_end": 2022,
  "month_end": 12,
  "fetch_function": "fetch_datasus",
  "process_function": "process_sim",
  "raw_path": "...",
  "processed_path": "...",
  "raw_rows": 0,
  "raw_cols": 0,
  "processed_rows": 0,
  "processed_cols": 0,
  "r_version": "...",
  "microdatasus_version": "...",
  "created_at": "...",
  "status": "ok|error",
  "error": ""
}

This is the minimal provenance contract.
25. Dataframe contract from microdatasus orchestration
The client should return an object like:
MicrodatasusResult(
    system="SIM-DO",
    request_hash="...",
    raw_path=Path(...),
    processed_path=Path(...),
    manifest_path=Path(...),
    raw_rows=...,
    processed_rows=...,
    raw_columns=[...],
    processed_columns=[...],
    status="ok",
)

It should not return a naked pandas dataframe as the main product. The product is the cached artifact + manifest.
26. Which processing function to call
Mapping:
SIM-DO   → process_sim
SIH-RD   → process_sih
SINASC   → process_sinasc
CNES-ST  → process_cnes
SIA      → process_sia
SINAN dengue/chikungunya/zika/malaria/chagas/etc.
         → disease-specific process_sinan_* functions

This should be centralized in client_microdatasus.py.
27. Raw/processed comparison output
For each system/request:
raw columns
processed columns
common columns
raw-only columns
processed-only columns
row count raw vs processed
column count raw vs processed
per-column missingness delta
per-column n_unique delta
per-column observed-kind delta
top-value comparison for common low-cardinality fields
identifier validity metrics
date validity metrics

This is not just documentation; it is a safety check.
28. Normalized DATASUS event/facility outputs
After fetch/profile/compare, normalize into domain tables.
SIM-DO normalized event table should eventually include:
event_id / source row id
year/month/date of death
municipality_residence
municipality_occurrence
age / age group
sex_recorded
race_color_recorded
underlying_cause_icd10
cause_line_fields
place/circumstance fields
facility identifier if available
raw/provenance reference
quality flags

SIH-RD:
admission/AIH source row id
competence year/month
admission/discharge dates if available
municipality_residence
municipality_movement/hospital
age
sex_recorded
race_color_recorded
diagnosis fields
procedure fields
death flag
length of stay
cost fields
hospital/facility identifiers if available
raw/provenance reference
quality flags

SINASC:
birth source row id
birth date/year/month
maternal residence municipality
birth municipality
maternal age
newborn sex
race/color field where available
gestational age
birthweight
prenatal visits
delivery type
Apgar
anomaly fields
facility identifier if available
raw/provenance reference
quality flags

CNES-ST:
facility identifier
competence/month
municipality of facility
facility type
legal/administrative attributes
SUS linkage
capacity/service indicators where present
raw/provenance reference
quality flags

These are not hardcoded parsers. They are target normalized forms that should be populated only by verified field mappings.
29. Optional generic DATASUS FTP discovery — what to keep
If the rewrite later includes generic FTP discovery, keep the good parts:
The prior apparatus was described as:
scan FTP
build inventory
infer families
store member_files
pick small candidate set
download candidates/docs
profile
build variable catalog
build translation bundles/registry

The project note says the family registry stores family_id, series_prefix, partition_type, date_format, time_range, member_files, source paths, geographic coverage, path semantics, associated docs, and schema signatures.
Candidate selection should be family-specific, based on actual member_files, with recency, path semantics, geographic coverage diversity, and time coverage diversity as ranking signals.
The valuable principle:
scan everything, download almost nothing

