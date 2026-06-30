1. What SIDRA is, in API terms
SIDRA is exposed programmatically through the IBGE Aggregate Data API v3. The official documentation describes it as the API that feeds SIDRA, “Sistema IBGE de Recuperação Automática,” and explicitly says that each SIDRA table corresponds to an API agregado. In OLAP terms, the official documentation maps variáveis to measures, classificações to dimensions, and categorias to dimension members. (IBGE)
The correct abstract data model is:
observation =
(table/agregado, variable, period, locality, classification-category tuple)
→ value

For miniPegaSUS/PegaSUS, that means SIDRA should be stored as a normalized aggregate fact system, not as a collection of wide CSV-like tables.
2. Core API base and endpoints
The base endpoint is:
https://servicodados.ibge.gov.br/api/v3/agregados

The official docs list the main routes:
GET /agregados
GET /agregados/{agregado}/localidades/{nivel}
GET /agregados/{agregado}/metadados
GET /agregados/{agregado}/periodos
GET /agregados/{agregado}/periodos/{periodos}/variaveis/{variavel}
GET /agregados/{agregado}/variaveis/{variavel}

The docs state that /agregados returns aggregates grouped by research/pesquisa and accepts query parameters such as periodo, assunto, classificacao, periodicidade, and nivel. They also list /localidades/{nivel}, /metadados, /periodos, and the values endpoint with required query parameter localidades plus optional classificacao and view. (IBGE)
Practical meaning:
/agregados
  catalog/search surface

/agregados/{id}/metadados
  full table metadata: variables, classifications, categories, territorial levels, etc.

/agregados/{id}/periodos
  available periods for a table

/agregados/{id}/localidades/{nivel}
  valid localities for a table at a geographic level

/agregados/{id}/periodos/{p}/variaveis/{v}
  actual data values

The shorter endpoint:
/agregados/{agregado}/variaveis/{variavel}

is officially described as functionally equivalent to requesting the last six researched periods via /periodos/-6/variaveis/{variavel}. (IBGE)
3. Fundamental SIDRA entities
3.1 Agregado / table
An agregado is a SIDRA table. It has a numeric ID, such as:
5938
9606
6579
6805

The official API documentation explicitly says that each SIDRA table corresponds to one aggregate. (IBGE)
Engineering consequence: our handpicked file, sidra_selected_cube_specs_inspection.jsonl, should be treated mainly as a selected aggregate/table seed, not as permanent truth about current metadata. The new system should read its table IDs, then rebuild official metadata from the API.
3.2 Variable / measure
A variável is a reported measure inside a table. Examples from our project include:
93      População residente
9324    População residente estimada
37      Produto Interno Bruto a preços correntes
381     Domicílios particulares permanentes ocupados

The API route places variable IDs in the path:
/agregados/{agregado}/periodos/{periodos}/variaveis/{variavel}

Variable expressions can include multiple variables separated with |; wrapper documentation also describes all/allxp style selections, but in our production code it is safer to request explicit variable IDs validated against metadata. (Monitoramento SEPE)
Engineering rule: never assume all variables in a table are analytically valid. Some are raw counts, some are percentages, some are means/medians, some are monetary values, some are derived shares. The registry must explicitly decide which variables are used.
3.3 Classification / dimension
A classificação is a categorical breakdown. Examples:
2       Sexo
86      Cor ou raça
287     Idade
1568    Nível de instrução
11558   Tipo de esgotamento sanitário

The official docs say the classificacao query parameter is available on the values endpoint. (IBGE)
The practical syntax is:
classificacao=86[2776,2777]|2[4,5]

meaning:
race/color categories 2776,2777
AND sex categories 4,5

3.4 Category / dimension member
A categoria is a member of a classification. For example, within Sexo:
4 = Homens
5 = Mulheres

Within Cor ou raça:
2776 = Branca
2777 = Preta
2778 = Amarela
2779 = Parda
2780 = Indígena

Important: category 0 or a label like Total is often an aggregate category. It is not a normal member. The API wrapper documentation explicitly notes that when classification is not specified, the API returns the Total category, and describes Total as a special aggregate across all categories. (Monitoramento SEPE)
Engineering consequence: every classification needs a category policy:
total_only
non_total
explicit_categories
all_categories_with_total_allowed

Do not blindly fetch all.
3.5 Period
Periods are table-specific and depend on periodicity. Annual tables use years like 2010, 2022; monthly tables may use 202001; quarterly/semiannual tables encode period differently. Wrapper documentation explains that period codes encode both date and periodicity and that the same code shape can mean different things depending on the aggregate’s periodicity. (Monitoramento SEPE)
Period expressions can use forms like:
2010
2010|2022
-6
201701-201706

For our use case, explicit period lists are safest.
3.6 Locality and geographic level
The values endpoint requires query parameter:
localidades=...

Common locality expressions:
BR
N1
N2
N3
N6
N6[2700300]
N6[2700300,2704302]
N6[N3[27]]

The wrapper documentation summarizes common levels as N1 Brazil, N2 major region, N3 state, N6 municipality, and warns that different geographic levels have different code systems, e.g. N6 municipality IDs differ from N7 metropolitan area IDs. (Monitoramento SEPE)
Engineering consequence: never treat geography as “just a code.” A locality ID must be stored with its level:
level = N6
locality_id = 2700300

not merely:
locality_id = 2700300

For miniPegaSUS, the main level is N6 municipality.
4. Metadata endpoints and their role
The correct metadata pipeline is:
selected table IDs
→ GET /agregados/{id}/metadados
→ GET /agregados/{id}/periodos
→ GET /agregados/{id}/localidades/N6
→ normalize metadata
→ validate extraction plan

The official docs define /metadados as returning metadata associated with an aggregate, /periodos as returning associated periods, and /localidades/{nivel} as returning the localities associated with the aggregate at a given geographic level. (IBGE)
The metadata layer must produce normalized records:
sidra_tables
sidra_variables
sidra_classifications / axes
sidra_categories
sidra_periods
sidra_localities

The old project plan already identified useful SIDRA logic as local metadata warehousing, metadata ingestion for tables/variables/classifications/categories/periods/localities, territorial coverage probing, search keys, and search/index support.
5. Value endpoint anatomy
The canonical values request is:
GET /agregados/{table}/periodos/{periods}/variaveis/{variables}
    ?localidades={localities}
    &classificacao={classification_expr}
    &view={view}

Example:
https://servicodados.ibge.gov.br/api/v3/agregados/9606/periodos/2010|2022/variaveis/93
  ?localidades=N6[2700300]
  &classificacao=86[2776,2777,2778,2779,2780]|2[4,5]|287[all]
  &view=flat

Conceptual output:
for each selected locality
for each selected period
for each selected variable
for each selected category combination
return value

6. View modes
The official documentation says API v3 permits three visualization/view modes for variables and points to the view parameter. (IBGE)
Practical modes:
default JSON
OLAP
flat

The ibger documentation states that view="flat" returns a flat response where the first element is metadata and data begins at the second element. (Monitoramento SEPE)
Our experience confirmed this: view=flat often produces a header/descriptor row that looks like data if not filtered. The normalizer must detect and drop it.
Production preference for miniPegaSUS:
Use view=flat for extraction,
then normalize into long facts.

Do not preserve flat output as final storage.
7. Request-size limits and chunking
The current external wrapper documentation reports a request limit of at most 100,000 values per request, with formula:
categories × periods × localities ≤ 100,000

and says excessive requests can return HTTP 500. (Monitoramento SEPE)
Our project convention was more conservative:
SIDRA_MAX_CELLS_PER_REQUEST = 49,900

or an operational target around:
45,000–49,000 cells

That lower ceiling is not “official truth”; it is a safety margin. Keep it unless live extraction tests show that raising it is reliable.
For our multidimensional requests, the safer full formula is:
estimated_cells =
n_localities
× n_periods
× n_variables
× product(n_selected_categories_per_classification)

The ibger formula omits variables in its displayed example because it discusses one variable, but our planner must include variables if multiple are requested.
Chunking rule:
if estimated_cells > ceiling:
    split deterministically

Split priority:
1. localities
2. highest-cardinality classification
3. periods
4. variables

This is especially important for N6 municipality extraction. A classification with 100+ age categories × 5,570 municipalities can explode immediately.
8. Value-status semantics
SIDRA values are not just floats. The extraction layer must preserve:
value_raw
value_numeric
value_status

Typical statuses to handle:
numeric
blank
dash_zero_or_nil
not_available
suppressed_or_unidentified
non_numeric_symbol
header_row

Do not coerce everything to numeric. Do not convert all dashes to zero globally. For count variables, - may be operationally interpretable as zero/nil, but for rates, means, medians, or protected/suppressed cells the semantics differ.
Safe schema:
table_code
variable_id
period
locality_level
locality_id
classification_tuple
category_tuple
value_raw
value_numeric
value_status
request_url
fetched_at
metadata_version/hash

For count-like variables, optionally add:
value_count_numeric

where dash_zero_or_nil can become 0 while raw status remains preserved.
9. Total categories and marginal logic
SIDRA often includes total categories inside classifications. This creates two risks.
First, if you include Total plus non-total members, you can double-count if you later aggregate naively.
Second, Total is often needed for validation, not modeling. Example:
Σ race-specific cells ≈ Total race cell
Σ sex-specific cells ≈ Total sex cell

Therefore, the registry should distinguish:
extraction_for_modeling
extraction_for_validation

For modeling:
use non-total categories unless the measure is inherently total-only

For validation:
fetch total categories and compare margins

10. Locality coverage is table-specific
Do not assume every table supports every locality level or every N6 municipality. The API exposes localities per aggregate through:
/agregados/{agregado}/localidades/{nivel}

This must be checked before extraction. The official endpoint exists precisely to ask which localities are associated with an aggregate at a given geographic level. (IBGE)
Known practical issues:
Some older census tables have fewer N6 municipalities.
Municipality lattices differ across years.
Some tables support only broader levels.
Some tables support N6 but not all modern municipalities.

So store:
source_table
source_period
source_locality_level
source_locality_id

Then harmonize geography downstream.
11. Metadata is not optional
For a serious pipeline, every value request must be validated against metadata before execution:
table exists
periods requested are valid
variables requested are valid
classification IDs exist
category IDs exist for those classifications
locality level is supported
localities exist for that table/level
estimated cell count is under ceiling

Do not interpret empty or symbolic output before validating the request. A missing or symbolic value can be caused by a bad variable-period combination, an unsupported category, an invalid locality expression, or a true missing/suppressed value.
12. Parallelism
SIDRA calls should be parallelized, but in a bounded way.
Metadata phase:
parallel over table IDs
fetch metadata + periods + localities per table

Value phase:
build chunk plan
parallel over chunks
bounded concurrency
retry transient failures
cache every response

Do not put API logic inside scripts. The project notes identified the need for a SIDRA subsystem with metadata ingest, search, coverage, and value normalization; the rewrite should make that a centralized API/service layer, not repeated ad hoc code.
A correct layering is:
scripts
  → workflows/services
    → sidra.api
      → HTTP/cache/retry/concurrency

13. Caching and provenance
Cache both metadata and values.
Metadata cache:
data/cache/sidra/http/{hash}.json
data/metadata/sidra/raw/{table}.metadata.json
data/metadata/sidra/raw/{table}.periods.json
data/metadata/sidra/raw/{table}.localities.N6.json

Value cache:
data/cache/sidra/values/{request_hash}.json

Each cached response should have sidecar metadata:
{
  "url": "...",
  "params": {...},
  "status_code": 200,
  "fetched_at": "...",
  "attempt": 1,
  "seconds": 0.123,
  "bytes": 12345,
  "sha256": "..."
}

Never let downstream data depend on ephemeral probe folders. Probe outputs can be archived as evidence, but production should rebuild metadata from official endpoints.
14. Recommended internal SIDRA architecture
A clean SIDRA module should have:
sidra/
  api.py
  cache.py
  metadata.py
  registry.py
  plan.py
  extract.py
  normalize.py
  category_maps.py
  quality.py

Responsibilities:
api.py
  endpoint construction, request execution, retry, bounded concurrency

cache.py
  persistent HTTP/JSON cache, request hashing, sidecar metadata

metadata.py
  fetch/rebuild raw metadata, normalize table/variable/axis/category/locality metadata

registry.py
  model-specific extraction definitions; category policies; variable policies

plan.py
  validate registry against metadata; estimate cells; split requests

extract.py
  execute chunk plans; save raw and normalized facts

normalize.py
  flatten API response; drop header rows; parse values; attach category/locality metadata

category_maps.py
  source category → canonical indicator mappings

quality.py
  value statuses, missingness, suppressed values, diagnostics

The old plan identified SIDRA’s missing piece as value retrieval and normalization: previous work was stronger at table discovery, metadata cataloging, and indexing than at producing PegaSUS-ready numerical observables.
15. SIDRA as normalized fact store, not wide tables
Do not fetch a SIDRA table and immediately pivot it into:
municipality-year rows × many columns

That is tempting but wrong as a base representation.
Reasons:
classification categories shift across years/tables
some categories are totals
some variables are percentages, some counts, some means/medians
age/race/education/income bins need harmonization
high-cardinality axes explode column counts
different source tables have different support lattices

Correct base representation:
sidra_fact:
  table_code
  variable_id
  period
  locality_level
  locality_id
  classification_id_1
  category_id_1
  classification_id_2
  category_id_2
  ...
  value_raw
  value_numeric
  value_status

Wide panels are compiled later from validated, canonical indicators.
16. Canonical category mapping
SIDRA category labels are source categories, not automatically canonical epidemiological variables.
Examples:
education categories differ across tables/years
income brackets differ across tables/years
sanitation categories differ between 2010 and 2022
age axes can be single-year, grouped, infant-month, 5-year, or mixed
race/color axes may vary by total/sem declaração/self-considered/territory definitions

Therefore the extraction registry should not just say “fetch table.” It should say:
fetch table T
use variable V
use categories C
map them to canonical indicator K
with aggregation/projection rule R

Examples:
water source categories → safe/unsafe/network/piped indicators
education categories → low_education / secondary_or_more / higher
income brackets → no_income / <=1/4 SM / <=1/2 SM / <=1 SM / >2 SM ...
age categories → canonical age bins via aggregation/inversion rules

17. SIDRA and the demographic tensor
SIDRA’s most important role in our project is the denominator/context substrate.
Key population tables from our selected set:
9606  age × sex × race population, 2010 and 2022
6579  annual estimated total population, partial series
9514  age × sex marginal check, 2022
9605  race/color marginal check, 2010/2022
4714  population/area/density context

The intended denominator tensor is:
N[g, t, age, sex, race]

where:
g = municipality
t = year

Core reconstruction logic:
N[g, 2010, a, s, r] = observed from 9606
N[g, 2022, a, s, r] = observed from 9606
Σ_a,s,r N[g,t,a,s,r] = total_population[g,t]

Total population spine:
2010 = census total
2011–2021 = 6579 where available
2022 = census total
2023 = gap unless another source is added
2024–2025 = 6579 where available, if extending forward

The SIDRA API provides the observed anchors; the modeling layer reconstructs annual joint distributions.
18. Major selected SIDRA table families
The selected JSONL/inspection work should be preserved as the table-selection seed. Conceptually, our selected tables covered:
Population:
  9606, 6579, 9514, 9605, 4714

Economy:
  5938, 21, 1685, 9509, 6450, 9528

Sanitation/housing:
  2065, 3154, 3218, 6803, 6804, 6805, 6806, 6892, 9860

Education:
  1554, 3547, 10141, 10142, 9543, 10062

Income:
  3563, 3578, 3571, 10295, 10296

Labor:
  616, 3572, 3579, 9517, 10280, 10289, 10292, 10300, 10381

Fertility history:
  10075, 10076, 10077, 2573

Indigenous/Quilombola:
  9764, 9718, 9578, 8176, 10089, 9723, 9765

Disability:
  1495, 3425, 3426

This list must be verified from the JSONL in the new code, not manually retyped as truth.
19. API expression conventions to support
Your SIDRA client should support these expression types.
Periods:
2010
2010|2022
-1
-6
201701-201706

Variables:
93
93|1000093
all

Production preference:
explicit variable IDs

Localities:
BR
N1
N2
N3
N6
N6[2700300]
N6[2700300,2704302]
N6[N3[27]]

Classifications:
86[2776,2777,2778,2779,2780]
2[4,5]
86[... ]|2[... ]|287[...]

Views:
flat
OLAP
default JSON

Production preference:
flat → normalized facts

20. Error and retry behavior
Expected failure modes:
HTTP 500 for oversized requests
HTTP 429 or transient server throttling
connection timeouts
empty or symbolic responses from invalid combinations
malformed assumptions about periods/categories
header rows in flat output
tables with no N6 support
high-cardinality category explosions

Retry strategy:
retry 429, 500, 502, 503, 504
exponential backoff
bounded concurrency
do not retry deterministic validation failures

Deterministic validation failures include:
unknown table
unknown variable
unsupported period
unsupported classification
unsupported category
unsupported locality level
estimated cell count over ceiling

21. Monetary variables and units
SIDRA includes many units:
pessoas
domicílios
%
R$
mil R$
salários mínimos
índices
área
densidade

Do not mix monetary variables without unit harmonization. PIB/current-price variables require deflation/normalization before entering time-series comparisons. SIDRA metadata units must be preserved from the variable metadata.
22. Race/color axis caveat
SIDRA census population denominators usually use IBGE self-declaration. DATASUS numerator race/color fields are administratively recorded or mediated by health records. So a race-specific rate like:
SIM deaths recorded as race=r / SIDRA population self-declared as race=r

is not measurement-process clean.
The SIDRA API itself does not solve this; the registry/model layer must tag axis provenance:
IBGE population denominator race = self_declared
SIM/SIH/SINASC race = administrative/health-recorded

This is a modeling/epistemology issue, not an API issue.
23. Practical production workflow
A robust SIDRA ingestion workflow should be:
1. Load selected table seed
   input: sidra_selected_cube_specs_inspection.jsonl
   output: selected table IDs

2. Fetch official metadata
   /metadados
   /periodos
   /localidades/N6

3. Normalize metadata
   tables
   variables
   classifications
   categories
   periods
   localities

4. Build/validate registry
   decide variables
   decide period policies
   decide category policies
   decide locality level
   decide modeling role

5. Plan requests
   estimate cells
   split chunks
   assign request IDs

6. Extract values
   bounded parallel calls
   persistent cache
   retry transient failures

7. Normalize values
   drop flat header rows
   parse value status
   store long facts

8. Run diagnostics
   row count equals planned cell count
   margin checks for Total categories
   missing/suppression status report
   table/variable/period coverage report

9. Build canonical indicators
   category maps
   age maps
   income thresholds
   sanitation indicators
   education indicators

24. Non-negotiable implementation rules
The rewritten code should obey these rules:
1. Scripts do not construct SIDRA URLs directly.
2. API endpoint logic lives in sidra.api only.
3. Metadata is rebuilt from official API endpoints.
4. Manual JSONL is table-selection seed, not permanent metadata truth.
5. Every response is cached with provenance.
6. Every value request is planned before execution.
7. Every request plan is checked against a conservative cell ceiling.
8. Every extracted value preserves raw value and value status.
9. Every source category remains source-coded until canonical mapping.
10. Wide panels are compiled downstream, never used as base storage.
11. Locality level is always stored with locality code.
12. Total categories are handled explicitly, not accidentally.
13. Monetary units are preserved and harmonized later.
14. Race/color measurement provenance is tagged outside the API layer.
15. Parallelism is bounded and centralized.

25. Minimal SIDRA client contract
The client should expose roughly:
class SidraClient:
    def catalog(self, *, periodo=None, assunto=None, classificacao=None,
                periodicidade=None, nivel=None) -> JSON: ...

    def metadata(self, table_code: str) -> JSON: ...

    def periods(self, table_code: str) -> JSON: ...

    def localities(self, table_code: str, level: str) -> JSON: ...

    def values(
        self,
        table_code: str,
        periods: list[str],
        variables: list[str],
        localities: str,
        classifications: dict[str, list[str]] | None,
        view: str = "flat",
    ) -> JSON: ...

But the rest of the code should not call these methods directly except through service/workflow objects.
26. Minimal normalized output contract
Metadata tables:
sidra_table
  table_code
  official_name
  research
  subject
  periodicity
  first_period
  last_period

sidra_variable
  table_code
  variable_id
  variable_name
  unit

sidra_classification
  table_code
  classification_id
  classification_name

sidra_category
  table_code
  classification_id
  category_id
  category_name
  parent_category_id?
  level?

sidra_period
  table_code
  period_id
  period_label?
  modified_at?

sidra_locality
  table_code
  level
  locality_id
  locality_name

Fact table:
sidra_fact
  table_code
  variable_id
  period
  locality_level
  locality_id
  classification_tuple_json
  category_tuple_json
  value_raw
  value_numeric
  value_status
  request_hash
  fetched_at

Diagnostics:
sidra_request_log
sidra_chunk_plan
sidra_value_status_summary
sidra_margin_check
sidra_coverage_check

27. Final synthesis
SIDRA’s API is structurally clean but operationally dangerous if treated casually. It gives explicit metadata and multidimensional values, but table-specific dimensions, total categories, changing category systems, request-size ceilings, locality coverage, and value-status symbols require disciplined engineering.
The correct mental model is:
SIDRA API = official aggregate-cube source
PegaSUS SIDRA layer = metadata-normalized, cached, chunked, long-fact extraction system
Modeling layer = canonical indicator compiler over those facts

The API layer should not know epidemiology. It should know only how to retrieve, validate, chunk, cache, and normalize SIDRA aggregate facts. The epidemiological meaning comes later through registries, category maps, denominator reconstruction, and measurement-provenance rules.

