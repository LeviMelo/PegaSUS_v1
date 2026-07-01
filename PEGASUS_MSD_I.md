# PegaSUS — Master System Document

Core Architecture, Formal Operational Specification, and Mathematical Verification Contract

## Status of This Document

This Master System Document defines the formal architecture of PegaSUS: what the system is, which mathematical objects it manipulates, which source realities it must preserve, which transformations it admits, which transformations it rejects, and which artifacts every run emits.

This document is the authoritative system-level specification. It does not define package layout, function signatures, implementation slices, Python/R subprocess protocols, or file trees. Those belong to the Technical Design Document.

The project name is **PegaSUS**. The historical name `miniPegaSUS` is retired. Legacy “problem” terminology is not used in the operative architecture. The production architecture is expressed through three formal components:

- **Substrate Harmonization Engine (SHE):** deterministic substrate compiler for ingestion, source decoding, source normalization, geospatial harmonization, denominator reconstruction, SIDRA contextual routing, longitudinal stitching, classification projection, ST-DFM interpolation, and admissible substrate-field generation.
- **Epidemiological Field Graph (EFG):** autonomous field compiler that constructs typed, provenance-aware epidemiological variables as a directed acyclic graph.
- **Parametric Inference & Residual Scanner (PIRS):** statistical and non-linear residual-inference layer that fits parametric models, extracts residual fields, and performs HSIC-based association scanning.

## Implementation Status Semantics

Every system component carries an implementation status.

| Status | Meaning |
|---|---|
| **Stable** | Interface, invariants, admissible inputs/outputs, metadata, state semantics, and failure modes are locked. |
| **Calibrated** | Mathematical structure is stable, but thresholds, priors, and hyperparameters remain tunable against validation data. |
| **Mutable Solver** | Inputs, outputs, state semantics, warnings, and invariants are locked; numerical backend or optimizer may change. |
| **Experimental** | Architecturally valid but not dashboard-safe by default. |
| **Deferred** | Architecturally recognized but outside the first production-safe execution scope. |

The production rule is:

$$
\boxed{
\text{Interfaces are stable; numerical solvers are replaceable only if they preserve inputs, outputs, states, warnings, and provenance.}
}
$$

# 1. System Ontology and Global Pipeline

## 1.1 System Definition

PegaSUS is an intent-conditioned, autonomous, measurement-process-aware epidemiological field compiler.

It maps Brazilian public-health event streams, health-service registries, demographic denominators, and socioeconomic/contextual cubes into a typed field space. Each generated field carries support, axes, carrier, unit, aggregation law, provenance, state tensor diagnostics, warnings, and full lineage.

PegaSUS is not a dashboard. PegaSUS is not an ETL script. PegaSUS is not a causal discovery oracle. It is a compiler that constructs legally admissible epidemiological variables and then generates statistically adjusted, residual-based, and non-linear causal-hypothesis candidates.

The system produces causal hypotheses. It does not certify causal identification without an additional identification layer.

## 1.2 Global Execution Chain

The global pipeline is:

$$
\mathcal{D}
\longrightarrow
SHE
\longrightarrow
EFG
\longrightarrow
PIRS
\longrightarrow
\mathcal{O}_{run}
$$

where:

$$
\mathcal{D}
=
\left\{
\mathcal{E}_{SIM},
\mathcal{E}_{SIH},
\mathcal{E}_{SINASC},
\mathcal{R}_{CNES},
\mathcal{C}_{SIDRA}
\right\}
$$

and:

- $\mathcal{E}_{SIM}$ is the SIM-DO death-event stream.
- $\mathcal{E}_{SIH}$ is the SIH-RD hospitalization-event stream.
- $\mathcal{E}_{SINASC}$ is the SINASC live-birth-event stream.
- $\mathcal{R}_{CNES}$ is the CNES facility-stock registry.
- $\mathcal{C}_{SIDRA}$ is the SIDRA/IBGE aggregate contextual cube substrate.

The execution chain is formally:

$$
\mathcal{D}
\xrightarrow{SHE}
\mathcal{B}
\xrightarrow{EFG}
\mathcal{G}_{EFG}
\xrightarrow{PIRS}
\mathcal{M}_{stat}
\times
HSIC_{residual}
\longrightarrow
\mathcal{O}_{run}
$$

where:

- $\mathcal{B}$ is the admissible substrate-field space emitted by SHE.
- $\mathcal{G}_{EFG}$ is the typed epidemiological field graph.
- $\mathcal{M}_{stat}$ is the family of fitted parametric or semiparametric models.
- $HSIC_{residual}$ is the residual non-linear dependence scanner.
- $\mathcal{O}_{run}$ is the immutable run bundle.

## 1.3 User Intent Boundary

Users control scope, budget, and admissible domain, not arbitrary formulas.

The user intent object is:

$$
\mathcal{I}
=
\left(
G,
T,
\mathcal{H}_0,
\mathcal{V}_0,
\mathbf{w}_{systems},
\mathcal{C}_{policy},
B,
\mathcal{M}_{geo},
\mathcal{R}_{force},
\mathcal{D}_{exclude}
\right)
$$

where:

- $G \subseteq S_t$ is the geographic domain.
- $T=[t_{start},t_{end}]\subset\mathbb{Y}$ is the temporal support.
- $\mathcal{H}_0\subseteq\mathfrak{I}_{ICD}$ is the initial health-event seed set.
- $\mathcal{V}_0$ is the set of mandatory anchor variables or field families.
- $\mathbf{w}_{systems}\in[0,1]^5$ weights source priority across SIM, SIH, SINASC, CNES, and SIDRA.
- $\mathcal{C}_{policy}$ selects allowed contextual families.
- $B\in\{fast,standard,deep\}$ is the computational budget token.
- $\mathcal{M}_{geo}\in\{native,AMC,geneallocated,hybrid\}$ selects spatial support behavior.
- $\mathcal{R}_{force}$ preserves sparse or unstable strata for forced reporting.
- $\mathcal{D}_{exclude}\subseteq\{SIM,SIH,SINASC,CNES,SIDRA\}$ hard-excludes source systems.

The system accepts bounded intent. It rejects free-form user-defined epidemiological formulas unless they are introduced through developer-level typed registry extensions.

## 1.4 Geospatial Support Function

Let $S_t$ be the dynamic Brazilian municipality lattice at year $t$. PegaSUS defines a mode-dependent support:

$$
S^*(\mathcal{M}_{geo})
=
\begin{cases}
S_t & \mathcal{M}_{geo}=native \\
\bar S & \mathcal{M}_{geo}=AMC \\
S_{t_{current}} & \mathcal{M}_{geo}=geneallocated \\
(\bar S,S_{t_{current}}) & \mathcal{M}_{geo}=hybrid
\end{cases}
$$

The modes are:

- **native:** preserves yearly municipalities and blocks lagged operations across structural boundary breaks.
- **AMC:** contracts all years into stable Minimum Comparable Areas.
- **geneallocated:** redistributes historical mass to current municipalities and is inferentially fragile.
- **hybrid:** uses AMC for modeling and scanning, and geneallocated support only for presentation.

AMC contraction is:

$$
\nu_{\bar{s},t}
=
(\pi_*\nu_t)(\bar{s})
=
\sum_{j\in W_t^{-1}(\bar{s})}\nu_t(j)
$$

Geneallocation is:

$$
(\mathcal{A}_{t*}\nu_{\bar{s}})(s)
=
\nu_{\bar{s},t}
\frac{\xi_t(s)}
{\sum_{j\in W_t^{-1}(\bar{s})}\xi_t(j)}
$$

Direct allocation of rates is forbidden:

$$
\mathcal{A}_t\left(\frac{\nu}{\mu}\right)
\quad
\text{is illegal}
$$

The only admissible intensive-field reconstruction is:

$$
\rho_s
=
\frac{\mathcal{A}_t(\nu_{\bar{s}})}
{\mathcal{A}_t(\mu_{\bar{s}})}
$$

Thus:

$$
\mathcal{A}_t\left(\frac{\nu}{\mu}\right)
\neq
\frac{\mathcal{A}_t(\nu)}
{\mathcal{A}_t(\mu)}
$$

and only the right-hand construction is admissible.

## 1.5 Run Profiles

Every run declares:

$$
RunProfile\in\{core\_vital,contextual,full\}
$$

- **core_vital:** SIM, SIH, SINASC, CNES, and the independent population denominator are in scope. `B_reconstructed`, `B_latent`, and `B_cross-sectional` may be empty only when the output bundle records an explicit `empty_by_profile` warning for each empty first-class key.
- **contextual:** `core_vital` plus the curated SIDRA compendium routed through regime classification, longitudinal stitching, classification projection, bounded high-dimensional pushforward, and gated ST-DFM. If these paths cannot run, compilation aborts instead of emitting silently empty context artifacts.
- **full:** `contextual` plus the full demographic tensor over `(s,t,a,x,r)` and race-bridge execution over real census strata.

The immutable 17-key output contract is unchanged: every key must exist. The run profile governs which keys must be non-empty and which empty keys are valid only with an explicit `empty_by_profile` warning row.

# 2. Substrate Harmonization Engine

## 2.1 SHE Definition

The Substrate Harmonization Engine is the deterministic substrate compiler. It ingests source systems, maps raw fields into canonical typed fields, decodes composite administrative structures, preserves missingness, enforces source-specific measurement semantics, reconstructs or harmonizes denominators when allowed, routes SIDRA contextual variables, stitches longitudinal concepts, projects SIDRA classifications to canonical axes, and emits admissible substrate fields.

SHE output is:

$$
\mathcal{B}
=
\mathcal{B}_{official}
\cup
\mathcal{B}_{harmonized}
\cup
\mathcal{B}_{deflated}
\cup
\mathcal{B}_{reconstructed}
\cup
\mathcal{B}_{latent}
\cup
\mathcal{B}_{cross-sectional}
\cup
\mathcal{B}_{excluded}
$$

Every emitted substrate field carries:

$$
(source,\ support,\ axes,\ carrier,\ unit,\ aggregation,\ provenance,\ state,\ warnings)
$$

No downstream module may treat official, harmonized, deflated, reconstructed, latent, cross-sectional, and excluded fields as equivalent.

## 2.2 Canonical Event Substrate Rule

The canonical health-event substrates are:

$$
\mathcal{E}_{SIM},
\quad
\mathcal{E}_{SIH},
\quad
\mathcal{E}_{SINASC}
$$

SIM-DO, SIH-RD, and SINASC remain canonical event streams because they carry marked event-level information required for:

$$
\sigma_C\nu,
\quad
\Psi_\varphi(\nu,m),
\quad
ICD\ traversal,
\quad
observer\ diagnostics,
\quad
record\text{-}level\ date\ logic
$$

SIDRA Civil Registry birth/death aggregates cannot replace SIM-DO or SINASC as primary vital-flow streams.

The compiler must reject:

$$
D^{SIM}\leftarrow D^{SIDRA}
$$

and:

$$
B^{SINASC}\leftarrow B^{SIDRA}
$$

as primary replacement modes.

SIDRA Civil Registry aggregates may be used only as validation, context, and observer-process comparators.

## 2.3 Safe Missingness Rule

The default missingness rule is:

$$
\boxed{
\text{Missing is never silently negative unless the official coding manual defines blank/null as negative.}
}
$$

Otherwise, missing or invalid source content is mapped to explicit states:

$$
Unknown,\ Missing,\ Invalid,\ Unparseable,\ NotRecorded,\ invalid\_mark\_state
$$

Missingness is an observer-process signal. It is never erased by default.

## 2.4 Source Field Registry

The Source Field Registry maps raw administrative fields into canonical typed fields before those fields enter $\mathcal{B}$. No raw source column may enter the SHE output as an untyped scalar.

Every raw column must pass through one of four registry routes:

$$
Route(x)
\in
\{
Decode,\ Parse,\ PreserveMark,\ Exclude
\}
$$

where:

- $Decode$ transforms structural numeric/string codes into canonical absolute units or canonical categories.
- $Parse$ transforms diagnostic, date, time, municipality, occupation, procedure, and identifier strings into typed semantic objects.
- $PreserveMark$ retains source marks that are not legal axes but are valid observer, context, socioeconomic, facility, maternal, or quality marks.
- $Exclude$ removes unmapped or illegal raw fields from $\mathcal{B}$ while preserving an audit record.

The registry-level invariant is:

$$
x_{raw}\notin\mathcal{B}
\quad
\text{unless}
\quad
x_{raw}\xrightarrow{\mathcal{R}_{source}}x_{canonical}
$$

Every emitted canonical field carries:

$$
(
source,
raw\_field,
canonical\_field,
decoder,
parser,
support,
axes,
carrier,
unit,
aggregation,
declaration\_process,
missingness\_state,
provenance,
warnings
)
$$

Missingness remains explicit:

$$
Unknown\neq Missing\neq Invalid\neq Unparseable\neq NotRecorded
$$

No blank, null, zero, sentinel, or administrative code is interpreted as negative unless the source registry explicitly declares that interpretation under official coding rules.

### 2.4.0 Composite Decoding Contract

Administrative numeric-looking fields are not scalar quantities by default. A field may be a structural code whose digits jointly encode unit, magnitude, category, sentinel status, or measurement process. The SHE must decode these fields before they enter $\mathcal{B}$.

For every raw field $x$ with decoder $d$, define:

$$
Decode_d:
\mathcal{X}_{raw}\times\mathcal{M}_{field}
\to
\mathbb{R}
\cup
\mathcal{C}_{canon}
\cup
Interval(\mathbb{R})
\cup
State_{invalid}
$$

where $\mathcal{M}_{field}$ is the field-specific metadata record.

The admissibility rule is:

$$
x_{raw}\xrightarrow{Decode_d}x_{canonical}
\Rightarrow
x_{canonical}\in\mathcal{B}
$$

and:

$$
x_{raw}\not\xrightarrow{Decode_d}x_{canonical}
\Rightarrow
x_{raw}\in\mathcal{B}_{excluded}
\quad
\text{with warning}
$$

#### 2.4.0.1 DATASUS Structural Age Decoder

SIM-DO structural age fields such as `IDADE` use a composite code. Let:

$$
z\in\{000,\ldots,599\}
$$

be the raw three-character age string after left-padding. Define:

$$
u(z)=\lfloor z/100\rfloor,
\quad
m(z)=z-100u(z)
$$

where $u(z)$ is the unit code and $m(z)$ is the magnitude code.

The canonical decoder is registry-driven:

$$
Decode_{SIM\_IDADE}(z)
=
\begin{cases}
(m,\ hours) & u=1 \\
(m,\ days) & u=2 \\
(m,\ months) & u=3 \\
(m,\ years) & u=4 \\
(100+m,\ years) & u=5 \\
UnknownAge & u\notin\{1,2,3,4,5\}
\end{cases}
$$

The canonical absolute age outputs are:

$$
age\_years
=
\begin{cases}
m/(24\cdot365.25) & u=1 \\
m/365.25 & u=2 \\
m/12 & u=3 \\
m & u=4 \\
100+m & u=5
\end{cases}
$$

and:

$$
age\_days
=
\begin{cases}
m/24 & u=1 \\
m & u=2 \\
30.4375m & u=3 \\
365.25m & u=4 \\
365.25(100+m) & u=5
\end{cases}
$$

Example:

$$
Decode_{SIM\_IDADE}(474)=(74,\ years)
$$

so:

$$
age\_years=74
$$

The raw structural code remains available only as a provenance mark:

$$
raw\_age\_code=474
$$

It is not the epidemiological age value.

SIH-RD age decoding uses the pair:

$$
(COD\_IDADE,\ IDADE)
$$

and must not infer the unit from `IDADE` alone. Its decoder is:

$$
Decode_{SIH\_AGE}(c,m)
=
UnitMap_{SIH}(c)\circ m
$$

where $UnitMap_{SIH}$ is the registry table for days, months, years, and special age-unit states. If $COD\_IDADE$ is missing or invalid:

$$
State=UnknownAgeUnit,
\quad
warning=age\_unit\_missing
$$

#### 2.4.0.2 Physical Scalar Decoder

Fields such as `PESO` are physical measurements, not automatically trusted scalars. They must pass through a physical scalar decoder:

$$
Decode_{physical}(x;unit,bounds,sentinels)
\to
(value,unit,state)
$$

For birth weight:

$$
Decode_{PESO}(x)
=
\begin{cases}
(x,\ grams,\ valid) & x\in[L_{peso},U_{peso}] \\
InvalidWeight & x\in Sentinels_{peso} \\
OutOfRangeWeight & x<L_{peso}\lor x>U_{peso} \\
MissingWeight & x=null
\end{cases}
$$

The canonical unit is:

$$
unit(PESO)=grams
$$

The raw numeric string is preserved as a source mark but cannot be used as a scalar field until decoded.

#### 2.4.0.3 Count-with-Leading-Zero Decoder

Fields such as `QTDFILVIVO`, `QTDFILMORT`, prior pregnancies, prior vaginal deliveries, and prior cesareans use string-like count encodings. Define:

$$
Decode_{count2}(z)
=
\begin{cases}
int(z) & z\in\{00,\ldots,99\}\land z\notin Sentinels \\
UnknownCount & z\in Sentinels \\
InvalidCount & otherwise
\end{cases}
$$

The canonical unit is:

$$
unit=count
$$

Leading zeros are representation, not magnitude modifiers.

#### 2.4.0.4 Boolean Sentinel Decoder

Administrative flag fields are not quantitative measures. The SHE must decode every boolean-like source flag through a sentinel-safe boolean decoder before the field enters $\mathcal{B}$.

Let $x$ be a raw administrative flag value. Define:

$$
Clamp_{bool}(x)
\to
(value,\ state,\ raw\_value,\ warning)
$$

where:

$$
value\in\{0,1,null\}
$$

and:

$$
state
\in
\{
ValidFalse,
ValidTrue,
MissingFlag,
InvalidFlagState,
UnparseableFlag
\}
$$

The decoder is:

$$
Clamp_{bool}(x)
=
\begin{cases}
(0,ValidFalse,x,\varnothing) & norm(x)\in\{0,\text{"0"},\text{"Não"},\text{"Nao"},\text{"No"},false\} \\
(1,ValidTrue,x,\varnothing) & norm(x)\in\{1,\text{"1"},\text{"Sim"},\text{"Yes"},true\} \\
(null,MissingFlag,x,missing\_flag) & x\in\{null,\text{""},NA\} \\
(null,InvalidFlagState,x,boolean\_flag\_outlier) & x\in\mathbb{Z}\land x\notin\{0,1\} \\
(null,InvalidFlagState,x,boolean\_flag\_outlier) & ParseInt(x)>1 \\
(null,InvalidFlagState,x,boolean\_flag\_outlier) & ParseInt(x)<0 \\
(null,UnparseableFlag,x,unparseable\_flag) & otherwise
\end{cases}
$$

The invariant is:

$$
x>1
\not\Rightarrow
Clamp_{bool}(x)=1
$$

and:

$$
x>1
\Rightarrow
state=InvalidFlagState
$$

Therefore raw administrative outliers such as:

$$
GESPRG6E=131,
\quad
GESPRG6M=97,
\quad
GESPRG5M=32,
\quad
NIVATE\_H=63,
\quad
URGEMERG=5
$$

must not be cast to `true`, must not enter quantitative sums as magnitudes, and must not be treated as valid binary states.

The affected CNES-ST boolean families include, at minimum:

$$
GESPRG*,
\quad
NIVATE\_*,
\quad
ATENDAMB,
\quad
ATENDHOS,
\quad
URGEMERG,
\quad
CENTRCIR,
\quad
CENTROBS,
\quad
SERAP*,
\quad
LEITHOSP
$$

A decoded boolean flag may generate a facility share:

$$
Share_{flag}(s,t)
=
\frac{
\sum_{f\in\mathcal{F}_{s,t}}\omega_f\mathbf{1}\{Clamp_{bool}(x_f)=1\}
}{
\sum_{f\in\mathcal{F}_{s,t}}\omega_f\mathbf{1}\{state_f\in\{ValidFalse,ValidTrue\}\}
}
$$

but invalid flag states must generate a parallel observer-process field:

$$
InvalidFlagShare_{flag}(s,t)
=
\frac{
\sum_{f\in\mathcal{F}_{s,t}}\mathbf{1}\{state_f=InvalidFlagState\}
}{
|\mathcal{F}_{s,t}|
}
$$

Any field derived from a boolean family with nonzero invalid-flag share inherits:

$$
warning=boolean\_flag\_outlier
$$

and receives a $Q(v)$ penalty through:

$$
missingness(v)
\leftarrow
missingness(v)+InvalidFlagShare_{flag}
$$

#### 2.4.0.5 Corporate Linkage Gate

Corporate identifiers and facility linkage identifiers are not epidemiological measures. They are fragile linkage keys. The SHE must sanitize CNES and SIH corporate identifiers before any facility-flow, facility-stock, or cross-system bridge operation.

Define:

$$
Filter_{CNPJ}(x)
\to
(cnpj^\*,state,warning)
$$

with:

$$
state
\in
\{
ValidCNPJ,
NullifiedZeroCNPJ,
InvalidCNPJLength,
InvalidCNPJDigits,
MissingCNPJ,
UnparseableCNPJ
\}
$$

Let:

$$
digits(x)=\text{only decimal digits of }x
$$

left-preserving leading zeros as string content. Then:

$$
Filter_{CNPJ}(x)
=
\begin{cases}
(null,MissingCNPJ,missing\_cnpj) & x\in\{null,\text{""},NA\} \\
(null,NullifiedZeroCNPJ,zero\_cnpj\_nullified) & digits(x)\in\{\text{"0"},\text{"00"},\ldots,\text{"00000000000000"}\} \\
(digits(x),ValidCNPJ,\varnothing) & |digits(x)|=14\land digits(x)\neq\text{"00000000000000"} \\
(null,InvalidCNPJLength,invalid\_cnpj\_length) & |digits(x)|\neq14 \\
(null,InvalidCNPJDigits,invalid\_cnpj\_digits) & |digits(x)|=14\land digits(x)\notin\mathbb{D}^{14} \\
(null,UnparseableCNPJ,unparseable\_cnpj) & otherwise
\end{cases}
$$

The mandatory nullification rule is:

$$
x\in\{0,\text{"0"},\text{"00000000000000"}\}
\Rightarrow
Filter_{CNPJ}(x)=null
$$

The affected fields include, at minimum:

$$
CNES.CPF\_CNPJ,
\quad
CNES.CNPJ\_MAN,
\quad
SIH.CGC\_HOSP,
\quad
SIH.CNPJ\_MANT
$$

The linkage relation is admissible only when both sides pass the gate:

$$
Link_{facility}(e,f)=1
\iff
Filter_{CNPJ}(id_e).state=ValidCNPJ
\land
Filter_{CNPJ}(id_f).state=ValidCNPJ
\land
Filter_{CNPJ}(id_e).cnpj^\*
=
Filter_{CNPJ}(id_f).cnpj^\*
$$

Zero or all-zero identifiers are non-links:

$$
Filter_{CNPJ}(id).state=NullifiedZeroCNPJ
\Rightarrow
Link_{facility}=0
$$

The EFG must reject any $Bridge_{FacilityFlow}$ operation that uses unsanitized corporate identifiers:

$$
Bridge_{FacilityFlow}(SIH,CNES)
\land
\neg SanitizedCNPJ
\Rightarrow
\Delta_{carrier}=0
$$

The purpose of this gate is to prevent catastrophic many-to-many expansion:

$$
\{00000000000000\}_{SIH}
\times
\{00000000000000\}_{CNES}
$$

which is a false Cartesian linkage, not a facility-flow signal.

If a valid CNPJ linkage is unavailable, the admissible fallback support is:

$$
(s_{movement},t)
\quad
\text{or}
\quad
(s_{occurrence},t)
$$

with:

$$
warning=facility\_linkage\_fragile
$$

#### 2.4.0.6 Zero-Variance Drop Invariant

The SHE must remove source columns that are empirically 100% missing or 100% constant during profiling before those columns can enter $\mathcal{B}$, $V_{fields}$, PIRS design matrices, or HSIC candidate sets.

For a source column $X_j$ over records $i=1,\ldots,n$, define:

$$
Miss(X_j)
=
\frac{1}{n}
\sum_{i=1}^{n}
\mathbf{1}\{X_{ij}\in MissingSet_j\}
$$

and:

$$
Card^+(X_j)
=
\left|
\{X_{ij}:X_{ij}\notin MissingSet_j\}
\right|
$$

The zero-variance exclusion predicate is:

$$
Drop_{ZV}(X_j)=1
$$

iff:

$$
Miss(X_j)=1
$$

or:

$$
Miss(X_j)<1
\land
Card^+(X_j)=1
$$

The SHE invariant is:

$$
Drop_{ZV}(X_j)=1
\Rightarrow
X_j\notin\mathcal{B}_{official}
\cup
\mathcal{B}_{harmonized}
\cup
\mathcal{B}_{reconstructed}
\cup
V_{fields}
$$

Dropped fields are emitted only into audit metadata:

$$
X_j\in\mathcal{B}_{excluded}
$$

with:

$$
reason
\in
\{
zero\_variance,
all\_missing,
constant\_sentinel
\}
$$

Examples include:

$$
DIAGSEC4,\ldots,DIAGSEC9
$$

when 100% missing, and fields such as:

$$
VAL\_SADT=0000,
\quad
DIAG\_SECUN=0000,
\quad
NATUREZA=00,
\quad
TPDISEC4,\ldots,TPDISEC9=0
$$

when 100% constant.

The PIRS design matrix admissibility condition is:

$$
X_j\in DesignMatrix
\Rightarrow
Drop_{ZV}(X_j)=0
$$

This prevents singular matrix inversion, rank-deficient Fisher information, degenerate QR decompositions, and null HSIC kernels:

$$
Var(X_j)=0
\Rightarrow
X_j\notin PIRS
$$

If a user forces a zero-variance field through $\mathcal{R}_{force}$, the field remains:

$$
State=quarantined\_descriptive
$$

and:

$$
ModelCovariate=0,
\quad
ModelOutcome=0,
\quad
HSICCandidate=0
$$

#### 2.4.0.7 Socioeconomic Categorical Decoder

Socioeconomic administrative fields are decoded into canonical marks, not silently treated as ordinal numbers.

For a raw categorical socioeconomic field $x$:

$$
Decode_{cat}(x)
=
(category,\ category\_system,\ category\_version,\ state)
$$

The output may be used as:

$$
role(v)\subseteq\{covariate,observer,context,demographic,model\_only\}
$$

but it cannot be treated as a continuous scalar unless a registered ordinal score or contrast operator is declared.

### 2.4.1 SIM-DO Core Fields

SIM-DO provides death-event records. Its canonical event carrier is:

$$
Carrier(SIM\text{-}DO)=Deaths
$$

SIM-DO fields are divided into event-time marks, demographic axes, socioeconomic marks, maternal/perinatal marks, diagnostic topology fields, facility marks, and observer-process marks.

| Raw field family | Canonical field | Carrier | Axis/mark | Decoder/parser | Rule |
|---|---|---|---|---|---|
| `DTOBITO` | `death_date` | Deaths | $T$ | date parser | Required for event time; null aborts record. |
| `HORAOBITO` | `death_hour` | Deaths | time mark | hour parser | Invalid hour becomes InvalidTimeMark. |
| `DTNASC` | `birth_date` | Deaths | birth mark | date parser | If null, age may be decoded from `IDADE` with warning. |
| `IDADE` | `age_years`, `age_days`, `age_unit`, `raw_age_code` | Deaths | $\mathcal{A}$ | `Decode_SIM_IDADE` | Structural composite code; raw value is not a scalar age. |
| `SEXO` | `sex` | Deaths | $\mathcal{S}$ | categorical decoder | Unknown if null or invalid. |
| `RACACOR` | `race_color_admin` | Deaths | $\mathcal{R}^{SIM}_{admin}$ | race-axis decoder | Administrative death-declaration race axis; not self-declared IBGE race. |
| `ESTCIV` | `marital_status` | Deaths | socioeconomic mark | categorical decoder | Preserved as individual-level socioeconomic/context mark. |
| `ESC` | `education_legacy` | Deaths | socioeconomic mark | categorical decoder | Legacy education scale; not automatically ordinal. |
| `ESC2010` | `education_2010` | Deaths | socioeconomic mark | categorical decoder | 2010 education scale; preferred education mark when valid. |
| `SERIESCFAL` | `school_grade_completed` | Deaths | socioeconomic mark | categorical/count decoder | Preserved with high-missingness state. |
| `OCUP` | `occupation_cbo` | Deaths | socioeconomic mark | CBO parser | CBO occupation code; invalid/sentinel codes preserved as occupation state. |
| `CODMUNRES` | `mun_residence` | Deaths | geography | municipality parser | Invalid code excluded from resident-risk fields. |
| `LOCOCOR` | `place_of_death` | Deaths | event-setting mark | categorical decoder | Preserved as observer/context mark. |
| `CODESTAB` | `facility_code` | Deaths | facility mark | CNES-code parser | Null preserved; no forced linkage. |
| `CODMUNOCOR` | `mun_occurrence` | Deaths | geography | municipality parser | Invalid code excluded from occurrence-burden fields. |
| `IDADEMAE` | `maternal_age_years` | Deaths | maternal mark | physical/count decoder | Maternal age in infant/perinatal death contexts; sparse by design. |
| `ESCMAE` | `maternal_education_legacy` | Deaths | maternal socioeconomic mark | categorical decoder | Preserved as maternal context mark. |
| `ESCMAE2010` | `maternal_education_2010` | Deaths | maternal socioeconomic mark | categorical decoder | Preferred maternal education mark when valid. |
| `SERIESCMAE` | `maternal_school_grade` | Deaths | maternal socioeconomic mark | categorical/count decoder | Preserved with missingness state. |
| `OCUPMAE` | `maternal_occupation_cbo` | Deaths | maternal socioeconomic mark | CBO parser | Preserved as maternal occupation context. |
| `QTDFILVIVO` | `maternal_living_children_count` | Deaths | maternal parity mark | `Decode_count2` | Count of living children; raw leading-zero string is not numeric magnitude. |
| `QTDFILMORT` | `maternal_deceased_children_count` | Deaths | maternal parity mark | `Decode_count2` | Count of deceased children; preserved even when sparse. |
| `GRAVIDEZ` | `pregnancy_type` | Deaths | maternal mark | categorical decoder | Unknown if null. |
| `SEMAGESTAC` | `gestational_weeks_death` | Deaths | perinatal mark | numeric decoder | Unknown if null/out of range. |
| `GESTACAO` | `gestational_age_group_death` | Deaths | perinatal mark | categorical decoder | Bracketed gestational age; not identical to weeks. |
| `PARTO` | `delivery_type_death_context` | Deaths | maternal mark | categorical decoder | Applies only to maternal/perinatal contexts. |
| `OBITOPARTO` | `death_timing_relative_to_delivery` | Deaths | maternal mark | categorical decoder | Unknown if null. |
| `PESO` | `birth_weight_death_context_grams` | Deaths | perinatal mark | `Decode_PESO` | Physical scalar decoder; invalid/sentinel values not treated as grams. |
| `OBITOGRAV` | `death_during_pregnancy` | Deaths | maternal mark | categorical decoder | Unknown if null or sentinel. |
| `OBITOPUERP` | `death_during_puerperium` | Deaths | maternal mark | categorical decoder | Unknown if null or sentinel. |
| `ASSISTMED` | `medical_assistance` | Deaths | observer mark | categorical decoder | Unknown if null or sentinel. |
| `EXAME` | `exam_performed` | Deaths | observer mark | categorical decoder | Preserved as observer mark. |
| `CIRURGIA` | `surgery_performed` | Deaths | observer mark | categorical decoder | Preserved as observer mark. |
| `NECROPSIA` | `autopsy_performed` | Deaths | observer mark | categorical decoder | Preserved as observer mark. |
| `CAUSABAS` | `underlying_icd` | Deaths | $\mathcal{H}_{underlying}$ | topological ICD parser | Underlying cause; not equivalent to chain mentions. |
| `LINHAA`–`LINHAD` | `terminal_causal_chain_icd` | Deaths | $\mathcal{H}_{terminal\_chain}$ | ordered topological ICD parser | Ordered Part I causal chain; order must be preserved. |
| `LINHAII` | `associated_conditions_icd` | Deaths | $\mathcal{H}_{associated}$ | unordered topological ICD parser | Other significant conditions; not a terminal sequence. |
| `COMUNSVOIM` | `svo_iml_municipality` | Deaths | observer/facility mark | municipality parser | Preserved for observer-process analysis. |
| `DTATESTADO` | `certificate_date` | Deaths | observer time mark | date parser | Used in reporting/certification delay fields. |
| `TPPOS` | `investigation_status` | Deaths | observer mark | categorical decoder | Unknown if null. |
| `DTINVESTIG` | `investigation_date` | Deaths | observer time mark | date parser | Preserved with invalid-date state. |

SIM-DO socioeconomic marks may generate individual-level context fields and interaction terms against SIDRA contextual fields, but only through registered operators. They do not become population denominators, and they do not override SIDRA/IBGE aggregate context.

### 2.4.2 SIH-RD Core Fields

SIH-RD provides hospitalization-event records. Its canonical event carrier is:

$$
Carrier(SIH\text{-}RD)=HospitalAdmissions
$$

SIH-RD fields are divided into event-time marks, residence/service geography, patient axes, diagnostic topology fields, hospital-outcome marks, ICU-use marks, economic burden components, facility-linkage marks, and observer-process marks.

| Raw field family | Canonical field | Carrier | Axis/mark | Decoder/parser | Rule |
|---|---|---|---|---|---|
| `DT_INTER` | `admit_date` | HospitalAdmissions | $T$ | date parser | Required for hospitalization event time. |
| `DT_SAIDA` | `discharge_date` | HospitalAdmissions | discharge mark | date parser | Null preserved; stay derived only when valid. |
| `MUNIC_RES` | `mun_residence` | HospitalAdmissions | geography | municipality parser | Invalid code excluded from resident-risk fields. |
| `MUNIC_MOV` | `mun_movement` | HospitalAdmissions | service geography | municipality parser | Invalid code excluded from service-flow fields. |
| `IDADE`, `COD_IDADE` | `age_years`, `age_days`, `age_unit` | HospitalAdmissions | $\mathcal{A}$ | SIH age decoder | Unit must be decoded from `COD_IDADE`; `IDADE` alone is not sufficient. |
| `SEXO` | `sex` | HospitalAdmissions | $\mathcal{S}$ | categorical decoder | Unknown if null or invalid. |
| `RACA_COR` | `race_color_billing` | HospitalAdmissions | $\mathcal{R}^{SIH}_{billing}$ | race-axis decoder | Billing-record race axis; not IBGE self-declared race. |
| `DIAG_PRINC` | `principal_icd` | HospitalAdmissions | $\mathcal{H}_{principal}$ | topological ICD parser | Principal diagnosis; not equivalent to SIM underlying cause. |
| `DIAGSEC1`–`DIAGSEC3` | `secondary_icd_set` | HospitalAdmissions | $\mathcal{H}_{secondary}$ | unordered ICD parser | Unordered comorbidity/secondary-diagnosis set. |
| `DIAGSEC4`–`DIAGSEC9` | excluded if all missing | HospitalAdmissions | excluded | zero-variance drop | Excluded when 100% missing. |
| `TPDISEC1`–`TPDISEC3` | `secondary_diagnosis_type` | HospitalAdmissions | diagnostic mark | categorical decoder | Acquired/pre-existing state; zero-variance fields excluded. |
| `PROC_SOLIC` | `procedure_requested` | HospitalAdmissions | procedure mark | procedure parser | Procedure code; not ICD. |
| `PROC_REA` | `procedure_performed` | HospitalAdmissions | procedure mark | procedure parser | Procedure code; not ICD. |
| `DIAS_PERM` / `QT_DIARIAS` | `stay_length_days` | HospitalAdmissions | duration mark | numeric decoder | Null/invalid becomes invalid_mark_state; do not clamp to zero. |
| `MARCA_UTI` | `icu_type_mark` | HospitalAdmissions | ICU mark | categorical decoder | `"00"` means no ICU only if registry declares it. |
| `UTI_MES_TO` | `icu_days_month_total` | HospitalAdmissions | ICU utilization mark | numeric decoder | Total ICU days in month; not a boolean. |
| `UTI_INT_TO` | `icu_days_hospitalization_total` | HospitalAdmissions | ICU utilization mark | numeric decoder | Total ICU days in hospitalization. |
| `VAL_SH` | `hospital_service_cost_real` | HospitalAdmissions | economic mark | monetary decoder | Hospital-services share; distinct economic burden component. |
| `VAL_SP` | `professional_service_cost_real` | HospitalAdmissions | economic mark | monetary decoder | Professional-services share; distinct economic burden component. |
| `VAL_UTI` | `icu_cost_real` | HospitalAdmissions | economic mark | monetary decoder | ICU-specific cost; not interchangeable with `VAL_SH` or `VAL_SP`. |
| `VAL_TOT` | `total_admission_cost_real` | HospitalAdmissions | economic mark | monetary decoder | Composite total; usable only as total billing burden, not as a substitute for components. |
| `VAL_SH_FED` | `federal_hospital_service_cost_real` | HospitalAdmissions | economic mark | monetary decoder | Federal hospital-share component; sparse/zero-heavy. |
| `VAL_SP_FED` | `federal_professional_service_cost_real` | HospitalAdmissions | economic mark | monetary decoder | Federal professional-share component; sparse/zero-heavy. |
| `VAL_UCI` | `intermediate_care_cost_real` | HospitalAdmissions | economic mark | monetary decoder | Intermediate-care cost component. |
| `VAL_SADT`, `VAL_RN`, `VAL_ACOMP`, `VAL_ORTP`, `VAL_SANGUE`, `VAL_SADTSR`, `VAL_TRANSP`, `VAL_OBSANG`, `VAL_PED1AC` | excluded if constant | HospitalAdmissions | excluded | zero-variance drop | Excluded when 100% constant. |
| `MORTE` | `hospital_death` | HospitalAdmissions | outcome mark | boolean/categorical decoder | Unknown if null; do not assume survived. |
| `GESTRISCO` | `high_risk_pregnancy_mark` | HospitalAdmissions | maternal mark | boolean sentinel decoder | Raw outliers invalid; not quantitative. |
| `CGC_HOSP` | `hospital_cnpj` | HospitalAdmissions | facility-link mark | `Filter_CNPJ` | Zero/all-zero identifiers nullified before linkage. |
| `CNPJ_MANT` | `maintainer_cnpj` | HospitalAdmissions | facility-link mark | `Filter_CNPJ` | Fragile corporate linkage; not facility identity. |
| `GESTOR_CPF` | `manager_identifier` | HospitalAdmissions | administrative mark | identifier parser | Not a facility key; never used for facility joins. |
| `AUD_JUST`, `SIS_JUST` | `audit_system_justification` | HospitalAdmissions | observer mark | sparse text parser | High-sparsity observer mark; not modeled by default. |

Economic decomposition is mandatory. The cost vector for admission $e$ is:

$$
\mathbf{c}^{SIH}(e)
=
\left(
c_{SH}(e),
c_{SP}(e),
c_{UTI}(e),
c_{TOT}(e)
\right)
$$

where:

$$
c_{SH}=VAL\_SH,
\quad
c_{SP}=VAL\_SP,
\quad
c_{UTI}=VAL\_UTI,
\quad
c_{TOT}=VAL\_TOT
$$

The components are not mutually substitutable:

$$
c_{SH}
\not\equiv
c_{SP}
\not\equiv
c_{UTI}
\not\equiv
c_{TOT}
$$

The component-wise marked functionals are:

$$
MeanCost_{SH,G}
=
\Psi_{mean}(c_{SH}\mid H^{SIH}_{principal}\in G)
$$

$$
MeanCost_{SP,G}
=
\Psi_{mean}(c_{SP}\mid H^{SIH}_{principal}\in G)
$$

$$
MeanCost_{UTI,G}
=
\Psi_{mean}(c_{UTI}\mid H^{SIH}_{principal}\in G)
$$

Total admission cost may be used only under:

$$
Role=total\_billing\_burden
$$

and must not replace component-specific economic fields when the estimand is hospital-service, professional-service, or ICU-specific burden.

### 2.4.3 SINASC Core Fields

SINASC provides live-birth-event records. Its canonical event carrier is:

$$
Carrier(SINASC)=LiveBirths
$$

SINASC fields are divided into newborn axes, maternal socioeconomic marks, reproductive-history marks, birth-outcome marks, diagnostic anomaly marks, facility marks, and observer-process marks.

| Raw field family | Canonical field | Carrier | Axis/mark | Decoder/parser | Rule |
|---|---|---|---|---|---|
| `DTNASC` | `birth_date` | LiveBirths | $T$ | date parser | Required for live-birth event time. |
| `HORANASC` | `birth_hour` | LiveBirths | time mark | hour parser | Invalid hour becomes InvalidTimeMark. |
| `CODESTAB` | `facility_code` | LiveBirths | facility mark | CNES-code parser | Null preserved; no forced linkage. |
| `CODMUNNASC` | `mun_birth_occurrence` | LiveBirths | geography | municipality parser | Occurrence support; distinct from residence. |
| `CODMUNRES` | `mun_residence` | LiveBirths | geography | municipality parser | Invalid code excluded from resident fields. |
| `LOCNASC` | `birth_location` | LiveBirths | event-setting mark | categorical decoder | Preserved as facility/context mark. |
| `IDADEMAE` | `maternal_age_years` | LiveBirths | maternal age mark | numeric decoder | Maternal age in years; invalid values become UnknownMaternalAge. |
| `DTNASCMAE` | `maternal_birth_date` | LiveBirths | maternal mark | date parser | Used only when valid; does not replace `IDADEMAE` without rule. |
| `ESTCIVMAE` | `maternal_marital_status` | LiveBirths | maternal socioeconomic mark | categorical decoder | Preserved as maternal socioeconomic context. |
| `ESCMAE` | `maternal_education_legacy` | LiveBirths | maternal socioeconomic mark | categorical decoder | Legacy maternal education scale. |
| `ESCMAE2010` | `maternal_education_2010` | LiveBirths | maternal socioeconomic mark | categorical decoder | Preferred maternal education scale when valid. |
| `SERIESCMAE` | `maternal_school_grade` | LiveBirths | maternal socioeconomic mark | categorical/count decoder | Preserved with missingness state. |
| `CODOCUPMAE` | `maternal_occupation_cbo` | LiveBirths | maternal socioeconomic mark | CBO parser | CBO occupation code; not a scalar. |
| `QTDFILVIVO` | `maternal_living_children_count` | LiveBirths | reproductive-history mark | `Decode_count2` | Count of living children. |
| `QTDFILMORT` | `maternal_deceased_children_count` | LiveBirths | reproductive-history mark | `Decode_count2` | Count of deceased children. |
| `QTDGESTANT` | `prior_pregnancy_count` | LiveBirths | reproductive-history mark | `Decode_count2` | Count of prior pregnancies. |
| `QTDPARTNOR` | `prior_vaginal_delivery_count` | LiveBirths | reproductive-history mark | `Decode_count2` | Count of prior normal deliveries. |
| `QTDPARTCES` | `prior_cesarean_delivery_count` | LiveBirths | reproductive-history mark | `Decode_count2` | Count of prior cesareans. |
| `NATURALMAE` | `maternal_state_of_birth` | LiveBirths | maternal origin mark | categorical/geographic parser | Preserved as maternal migration/origin context. |
| `CODMUNNATU` | `maternal_municipality_of_birth` | LiveBirths | maternal origin geography | municipality parser | Invalid codes preserved as InvalidOriginMunicipality. |
| `CODUFNATU` | `maternal_birth_state_code` | LiveBirths | maternal origin mark | UF parser | Preserved as maternal origin context. |
| `SEXO` | `newborn_sex` | LiveBirths | $\mathcal{S}$ | categorical decoder | Unknown if null/invalid. |
| `RACACOR` | `newborn_race_admin` | LiveBirths | $\mathcal{R}^{SINASC}_{mixed}$ | race-axis decoder | Administrative mixed race axis. |
| `RACACORMAE` | `maternal_race_admin` | LiveBirths | maternal race mark | race-axis decoder | Distinct from newborn race and IBGE self-declared denominator race. |
| `GESTACAO` | `gestational_age_group` | LiveBirths | birth-outcome mark | categorical decoder | Bracketed gestational age; not continuous weeks. |
| `SEMAGESTAC` | `gestational_weeks` | LiveBirths | birth-outcome mark | numeric decoder | Used when available; invalid becomes UnknownGestationalWeeks. |
| `GRAVIDEZ` | `pregnancy_type` | LiveBirths | birth-outcome mark | categorical decoder | Singleton/twin/etc.; unknown if null. |
| `PARTO` | `delivery_type` | LiveBirths | birth-outcome mark | categorical decoder | Used for cesarean/vaginal fields. |
| `CONSULTAS` | `prenatal_visit_group` | LiveBirths | care-process mark | categorical decoder | Ordinal use requires registered scoring. |
| `APGAR1` | `apgar_1min` | LiveBirths | birth-outcome mark | numeric/sentinel decoder | 99 and invalid values become UnknownApgar. |
| `APGAR5` | `apgar_5min` | LiveBirths | birth-outcome mark | numeric/sentinel decoder | 99 and invalid values become UnknownApgar. |
| `PESO` | `birth_weight_grams` | LiveBirths | birth-outcome mark | `Decode_PESO` | Physical scalar decoder; canonical unit is grams. |
| `IDANOMAL` | `anomaly_flag` | LiveBirths | anomaly mark | categorical decoder | Unknown if null/sentinel; no forced no-anomaly. |
| `CODANOMAL` | `anomaly_icd` | LiveBirths | $\mathcal{H}_{anomaly}$ | topological ICD parser | Congenital anomaly code; parsed separately from mortality/hospital diagnosis. |
| `DTCADASTRO` | `birth_record_entry_date` | LiveBirths | observer time mark | date parser | Preserved for reporting-delay analysis. |
| `DTRECEBIM` | `birth_record_receipt_date` | LiveBirths | observer time mark | date parser | Preserved for reporting-delay analysis. |
| `DIFDATA` | `birth_reporting_delay` | LiveBirths | observer mark | numeric decoder | Null/invalid becomes invalid_mark_state. |

SINASC maternal socioeconomic marks may generate maternal-context covariates and interactions with SIDRA education, income, labor, sanitation, and demographic fields. These marks remain individual-event attributes and do not replace aggregate contextual denominators.

### 2.4.4 CNES-ST Core Fields

CNES-ST provides facility-stock states. Its canonical carrier is:

$$
Carrier(CNES\text{-}ST)=Facilities
$$

CNES-ST does not expose a single scalar “bed count” or “room count.” It exposes a vector of facility-capacity components indexed by source-specific structural codes.

For facility $f$ at support $(s,t)$, define the capacity object:

$$
\mathbf{C}^{CNES}_{f,s,t}
=
\left(
\mathbf{q}^{inst}_{f,s,t},
\mathbf{q}^{leit}_{f,s,t},
\mathbf{b}_{flag,f,s,t},
\mathbf{z}_{link,f,s,t}
\right)
$$

where:

$$
\mathbf{q}^{inst}_{f,s,t}
=
(QTINST01,\ldots,QTINSTK)
$$

is the physical-room/infrastructure vector,

$$
\mathbf{q}^{leit}_{f,s,t}
=
(QTLEIT05,\ldots,QTLEITK,QTLEITP1,QTLEITP2,QTLEITP3,\ldots)
$$

is the specialized-bed vector,

$$
\mathbf{b}_{flag,f,s,t}
$$

is the sentinel-decoded boolean-service vector, and:

$$
\mathbf{z}_{link,f,s,t}
$$

is the sanitized facility/corporate linkage vector.

| Raw field family | Canonical field | Carrier | Axis/mark | Decoder/parser | Rule |
|---|---|---|---|---|---|
| `CNES` | `facility_id` | Facilities | $\mathcal{F}$ | facility-id parser | Required for facility stock. |
| municipality code | `facility_municipality` | Facilities | geography | municipality parser | Invalid code excluded from municipal capacity fields. |
| `CPF_CNPJ` | `facility_cnpj` | Facilities | facility-link mark | `Filter_CNPJ` | `0` and `00000000000000` nullified. |
| `CNPJ_MAN` | `maintainer_cnpj` | Facilities | corporate-link mark | `Filter_CNPJ` | All-zero identifiers nullified; not a facility key. |
| `QTINST*` | `facility_room_capacity_vector[k]` | FacilityCapacityVector | capacity vector index | nonnegative integer decoder | Each `QTINST` field is a distinct room/infrastructure component. |
| `QTLEIT*` | `facility_bed_capacity_vector[k]` | FacilityCapacityVector | capacity vector index | nonnegative integer decoder | Each `QTLEIT` field is a distinct bed component. |
| `QTLEITP1` | `clinical_bed_capacity` | FacilityCapacityVector | bed vector index | nonnegative integer decoder | Clinical beds; not generic beds. |
| `QTLEITP2` | `surgical_bed_capacity` | FacilityCapacityVector | bed vector index | nonnegative integer decoder | Surgical beds; not generic beds. |
| `QTLEITP3` | `obstetric_bed_capacity` | FacilityCapacityVector | bed vector index | nonnegative integer decoder | Obstetric beds; not generic beds. |
| `GESPRG*` | `programmatic_flag_vector[k]` | Facilities | boolean service mark | `Clamp_bool` | Values greater than 1 become InvalidFlagState. |
| `NIVATE_A`, `NIVATE_H` | `care_complexity_flag[k]` | Facilities | boolean service mark | `Clamp_bool` | Raw outliers invalid; not quantitative levels. |
| `ATENDAMB`, `ATENDHOS` | `care_modality_flag[k]` | Facilities | boolean service mark | `Clamp_bool` | Boolean support flags. |
| `URGEMERG` | `urgency_emergency_flag` | Facilities | boolean service mark | `Clamp_bool` | Outliers greater than 1 invalid. |
| `CENTRCIR`, `CENTROBS` | `surgical_obstetric_center_flag[k]` | Facilities | boolean service mark | `Clamp_bool` | Outliers invalid; not room counts. |
| `SERAP*` | `support_service_flag_vector[k]` | Facilities | boolean service mark | `Clamp_bool` | Own/third-party support-service flags. |
| `LEITHOSP` | `hospital_bed_presence_flag` | Facilities | boolean service mark | `Clamp_bool` | Presence flag; not a bed count. |
| `VINC_SUS` | `sus_linkage` | Facilities | service mark | categorical/boolean decoder | Unknown if null; do not force non-SUS. |
| professional-count fields | `workforce_capacity_vector[k]` | FacilityCapacityVector | workforce vector index | nonnegative integer decoder | Generated only when the professional module is available. |

The vector index set is:

$$
\mathcal{K}_{capacity}
=
\mathcal{K}_{inst}
\cup
\mathcal{K}_{leit}
\cup
\mathcal{K}_{workforce}
$$

A facility-capacity scalar is legal only after selecting a vector component:

$$
C_{s,t,k}
=
\sum_{f\in\mathcal{F}_{s,t}}q_{f,s,t,k}
\quad
k\in\mathcal{K}_{capacity}
$$

Generic capacity requests are illegal:

$$
Beds
\quad
\text{without}
\quad
k\in\mathcal{K}_{leit}
\Rightarrow
\Delta_{carrier}=0
$$

Examples of legal bed-capacity fields are:

$$
ClinicalBeds_{s,t}
=
\sum_{f\in\mathcal{F}_{s,t}}QTLEITP1_{f,s,t}
$$

$$
SurgicalBeds_{s,t}
=
\sum_{f\in\mathcal{F}_{s,t}}QTLEITP2_{f,s,t}
$$

$$
ObstetricBeds_{s,t}
=
\sum_{f\in\mathcal{F}_{s,t}}QTLEITP3_{f,s,t}
$$

The capacity-density operator is:

$$
CapacityDensity_{k}(s,t)
=
\frac{
C_{s,t,k}
}{
P_{s,t}
}
$$

and is legal only when:

$$
k\in\mathcal{K}_{capacity}
$$

Every capacity field carries:

$$
CapacityVectorMetadata
=
(
raw\_field,
capacity\_family,
capacity\_index,
source\_definition,
unit,
decoder,
invalid\_state\_share,
warnings
)
$$

### 2.4.5 SIDRA Core Fields

SIDRA provides aggregate contextual facts.

| Raw field | Canonical object | Carrier | Rule |
|---|---|---|---|
| table id | `sidra_table` | Context | Required. |
| variable id | `sidra_variable` | Context | Required. |
| period | `period` | Context | Required. |
| locality | `locality` | Context | Must map to $S^*$. |
| classifications | `classification_tuple` | Context | Full tuple preserved. |
| value | `value` | Context | Parsed by unit/type registry. |

SIDRA is a normalized OLAP fact system. It is not a wide-table substrate.

A SIDRA fact has the logical form:

$$
(table,\ variable,\ period,\ locality,\ classification\text{-}category\ tuple)
\longrightarrow
value
$$

## 2.5 ICD Parsing, Diagnostic Topology, and Health-Event Role Contract

Diagnostic strings are not homogeneous code mentions. The SHE must preserve diagnostic topology before emitting any health-event field into $\mathcal{B}$.

For every diagnostic source field $h$, the parser emits:

$$
ParseICD(h)
=
(code,\ parse\_state,\ topology\_role,\ position,\ source\_field,\ warnings)
$$

where:

$$
parse\_state
\in
\{
valid,
ill\text{-}defined,
blank,
invalid,
unparseable,
missing
\}
$$

and:

$$
topology\_role
\in
\{
underlying\_cause,
principal\_diagnosis,
terminal\_chain,
associated\_condition,
secondary\_comorbidity,
congenital\_anomaly,
quality\_code,
unknown\_diagnostic\_role
\}
$$

The parser must strip source-specific line markers such as leading asterisks without erasing the original raw string:

$$
raw(h)=*A419,
\quad
normalized(h)=A419,
\quad
raw\_marker=*
$$

The normalized code is parsed for ICD validity. The raw string remains in lineage.

### 2.5.1 Six-State ICD Parse Automaton

$$
ParseState(h)
\in
\{
valid,
ill\text{-}defined,
blank,
invalid,
unparseable,
missing
\}
$$

| State | Condition | Route |
|---|---|---|
| `valid` | Matches ICD format and exists in ICD catalog | Eligible for topology-specific health-event use. |
| `ill-defined` | Valid code inside curated ill-defined/garbage/low-specificity set | Observer-process route unless explicitly allowed. |
| `blank` | Empty string | Data-quality buffer; no biological role. |
| `invalid` | Formatted but not valid ICD catalog code | Data-quality buffer. |
| `unparseable` | Fails ICD pattern after permitted source-marker stripping | Data-quality buffer. |
| `missing` | Field structurally absent | Source warning; no biological role. |

Blank, invalid, unparseable, and missing codes must not be coerced into R99.

### 2.5.2 SIM-DO Diagnostic Topology

SIM-DO contains three distinct diagnostic objects.

#### 2.5.2.1 Underlying Cause

The underlying cause field is:

$$
H^{SIM}_{underlying}(e)=CAUSABAS(e)
$$

It defines the canonical mortality-cause numerator when parse state permits:

$$
\nu^{SIM}_{G,underlying}
=
\sum_e
\mathbf{1}
\{
H^{SIM}_{underlying}(e)\in G
\}
$$

This object is not equivalent to a line-chain mention.

#### 2.5.2.2 Ordered Terminal Causal Chain

The terminal causal chain is the ordered Part I sequence:

$$
H^{SIM}_{chain}(e)
=
(h_A,h_B,h_C,h_D)
$$

where:

$$
h_A=LINHAA(e),
\quad
h_B=LINHAB(e),
\quad
h_C=LINHAC(e),
\quad
h_D=LINHAD(e)
$$

The order relation is:

$$
LINHAA\prec LINHAB\prec LINHAC\prec LINHAD
$$

The chain support is:

$$
\mathcal{H}_{terminal\_chain}
=
\mathcal{C}_{ICD}
\times
\{A,B,C,D\}
$$

A chain-derived field must preserve position unless an explicit topology-erasing projection is declared:

$$
\pi^{chain\to mention}_*
:
\mathcal{C}_{ICD}\times\{A,B,C,D\}
\to
\mathcal{C}_{ICD}
$$

This projection emits:

$$
Role=observer\ \text{or}\ exploratory
$$

unless a registry entry promotes a specific chain-use case.

#### 2.5.2.3 Associated Conditions

The Part II field is:

$$
H^{SIM}_{associated}(e)=LINHAII(e)
$$

It represents other significant conditions contributing to death. It is not ordered with the Part I terminal sequence and is not equivalent to the underlying cause.

Associated-condition fields may generate comorbidity, multimorbidity, observer, or adjustment covariates. They do not generate cause-specific mortality numerators unless an explicit bridge grammar or topology-specific operator declares that estimand.

### 2.5.3 SIH-RD Diagnostic Topology

SIH-RD contains a principal diagnosis and unordered secondary diagnoses.

The principal hospitalization diagnosis is:

$$
H^{SIH}_{principal}(e)=DIAG\_PRINC(e)
$$

Secondary diagnoses are:

$$
H^{SIH}_{secondary}(e)
=
\{DIAGSEC_1(e),\ldots,DIAGSEC_K(e)\}
$$

with unordered-set semantics:

$$
DIAGSEC_i\not\prec DIAGSEC_j
$$

unless a source-specific registry explicitly supplies temporal or causal order.

Principal-diagnosis hospitalization numerators are:

$$
\nu^{SIH}_{G,principal}
=
\sum_e
\mathbf{1}
\{
H^{SIH}_{principal}(e)\in G
\}
$$

Secondary diagnoses define comorbidity or adjustment fields:

$$
\nu^{SIH}_{G,secondary\_mention}
=
\sum_e
\mathbf{1}
\{
\exists k:\ DIAGSEC_k(e)\in G
\}
$$

These are not hospitalization incidence numerators unless explicitly declared.

### 2.5.4 SINASC Diagnostic Topology

SINASC anomaly codes are:

$$
H^{SINASC}_{anomaly}(e)=CODANOMAL(e)
$$

They belong to:

$$
\mathcal{H}_{anomaly}
$$

and are not directly comparable to SIM underlying-cause codes or SIH principal diagnoses without a registered bridge.

### 2.5.5 Diagnostic Object Tuple

Every health diagnostic field emitted by SHE must carry:

$$
\mathcal{D}_H(v)
=
(
code\_set,
source,
diagnostic\_role,
topology,
position,
parse\_state,
quality\_class,
lineage
)
$$

where:

$$
topology
\in
\{
single\_underlying,
single\_principal,
ordered\_chain,
unordered\_set,
single\_anomaly
\}
$$

The EFG must use $\mathcal{D}_H(v)$ when evaluating health-axis legality.

## 2.6 Clinical Event-Definition Registry

SHE defines event carriers through explicit clinical definitions.

### 2.6.1 Age in Days

$$
AgeDays(e)
=
death\_date(e)-birth\_date(e)
$$

when both dates are valid. If not, decoded raw age may be used with warning.

### 2.6.2 Infant Death

$$
InfantDeath(e)=1
\iff
AgeDays(e)<365.25
\land
death\_type(e)=liveborn\ death
$$

### 2.6.3 Neonatal Death

$$
NeonatalDeath(e)=1
\iff
AgeDays(e)<28
\land
death\_type(e)=liveborn\ death
$$

### 2.6.4 Postneonatal Death

$$
PostNeonatalDeath(e)=1
\iff
28\le AgeDays(e)<365.25
$$

### 2.6.5 Perinatal Death

Allowed definition variants are:

$$
PerinatalDefinition
\in
\{WHO\_standard,\ Brazil\_official,\ custom\}
$$

Default perinatal death includes fetal deaths with gestational age at least 22 weeks and early neonatal deaths under 7 days, subject to data availability.

The denominator is:

$$
LiveBirths+EligibleFetalDeaths
$$

If fetal records are unavailable:

$$
State=fragile,
\quad
warning=perinatal\_denominator\_incomplete
$$

### 2.6.6 Maternal Death

Allowed maternal-death variants are:

$$
MaternalDeathDefinition
\in
\{strict\_MMR,\ pregnancy\_related,\ late\_maternal,\ ICD\_obstetric,\ custom\}
$$

The default strict MMR definition uses official maternal ICD sets and pregnancy/puerperium timing, excluding accidental/incidental causes when identifiable.

Pregnancy-related variants are labeled separately and do not collapse into strict maternal mortality.

## 2.7 CNES Facility Stock Transformation

CNES records are facility-stock states:

$$
\mathcal{R}_{CNES}
=
\{(f,s,t,\mathbf{m}_f)\}
$$

where $\mathbf{m}_f$ includes decoded capacity-vector components, sanitized corporate linkage marks, and boolean-service states.

For vector-indexed additive stock metrics:

$$
C_{s,t,k}
=
\sum_{f\in\mathcal{F}_{s,t}}q_{f,s,t,k}
\quad
k\in\mathcal{K}_{capacity}
$$

For categorical operational capacities:

$$
C^{(\omega)}_{s,t,c}
=
\frac{
\sum_{f\in\mathcal{F}_{s,t}}\omega_f\mathbf{1}\{m_{f,c}=true\}
}{
\sum_{f\in\mathcal{F}_{s,t}}\omega_f
}
$$

Allowed weights are:

$$
\omega_f
\in
\{1,\ FacilityCapacityVector_{k},\ Admissions_f,\ SUS_f\cdot FacilityCapacityVector_{k},\ Workforce_{f,k}\}
$$

Allowed CNES views are:

- facility_share
- vector_capacity_weighted
- volume_weighted
- SUS_restricted
- population_density
- invalid_flag_share
- linkage_quality_share

Facility linkage is fragile unless a validated crosswalk is supplied:

$$
SIH.CGC\_HOSP
\not\equiv
CNES.CNES
$$

and:

$$
SIH.CGC\_HOSP
\not\equiv
CNES.CPF\_CNPJ
$$

unless $Filter_{CNPJ}$ has validated both sides and a linkage contract is declared.

## 2.8 Population Denominator Tensor

### 2.8.1 Tensor Definition

The canonical demographic denominator tensor is:

$$
P_{s,t,a,x,r}
$$

over:

$$
\Omega_{epi}(\mathcal{M}_{geo})
=
S^*(\mathcal{M}_{geo})
\times T
\times \mathcal{A}
\times \mathcal{S}
\times \mathcal{R}
$$

The optimizer estimates:

$$
\widehat{\mathbf{P}},\widehat{\boldsymbol{\eta}}
=
\arg\min_{\mathbf{P},\boldsymbol{\eta}}
\mathcal{L}(\mathbf{P},\boldsymbol{\eta})
$$

subject to:

$$
P_{s,t,a,x,r}\ge0
$$

and:

$$
\eta_{s,t,a,x,r}\in[-E_{s,t},E_{s,t}]
$$

**Axis identity with §2.12.2/§3.7.4 (architectural note).** $(a,x,r)$ here are not a
tensor-local convenience — they *are* $\mathcal{A},\mathcal{S},\mathcal{R}$, the same
canonical demographic axes §2.12.2's Classification Projection Matrix $\Pi_{Clsf\to
Axis}$ projects SIDRA classification tuples onto, and the same axes §3.7.4 requires a
DATASUS health-event count to be aligned to before it may be divided by a population
denominator. This identity is what lets a stratified DATASUS-origin numerator (e.g.
SIM deaths in one age/sex/race cell) be divided by this tensor's matching cell to
produce an age/sex/race-specific rate — the numerator and denominator are legal to
combine only because they share one axis registry, not two independently-labelled
ones. Implementation: `pegasus.registries.demographic_axis` is the single source of
truth for the $(a,x,r)$ category labels on both sides (SIDRA-origin denominator
strata and DATASUS-origin numerator events); `config/registries/demographic_axis_maps.yaml`
is the registry-driven crosswalk (§3.7.4: "Silent redistribution is forbidden unless a
declared allocation policy is used" — sex crosswalks the raw DATASUS code directly
because administrative and declared sex are the same declaration-process object;
race does not, for the reason given next).

Race is the one axis on this list where the crosswalk is not a simple code map. DATASUS
administrative race/color (SIM `race_color_admin`, SINASC `newborn_race_admin`) is a
declaration-process object distinct from IBGE self-declared census race ($\mathcal{R}$
itself, per §2.12.2's "Race/color projections from SIDRA census tables map to
$\mathcal{R}^{IBGE}_{self}$, not to administrative health-system race axes"). A
DATASUS-origin race-stratified numerator may only enter this tensor's $r$ axis through
`pegasus.efg.race_bridge`'s Bridge_R posterior (§3.7.4's crosswalk-only-when-compatible
rule) — never a direct category map. §2.8.5 and §2.8.6 below route SINASC/SIM race
through Bridge_R for exactly this reason; §2.8.8's race composition loss does not, because
it never touches DATASUS-origin race at all (both sides of that term are SIDRA
self-declared).

### 2.8.2 Denominator Independence Modes

PegaSUS distinguishes independent denominators from death-assisted denominators:

$$
B^{POP}
\in
\left\{
B^{POP}_{independent},
B^{POP}_{SIM\text{-}informed}
\right\}
$$

The canonical default is:

$$
B^{POP}_{independent}
$$

In independent mode:

$$
\lambda_D=0
$$

so SIM deaths do not influence the denominator used to calculate SIM mortality rates.

SIM mortality fields must use:

$$
B^{POP}_{independent}
$$

unless death-assisted reconstruction is explicitly enabled.

In SIM-informed mode:

$$
\lambda_D>0
$$

and the denominator receives provenance:

$$
Prov(B^{POP})=SIM\_informed\_denominator\_prior
$$

Any SIM mortality field:

$$
\rho_G^{SIM}
=
\frac{\nu_G^{SIM}}
{B^{POP}_{SIM\text{-}informed}}
$$

inherits:

$$
warning=SIM\_denominator\_feedback\_risk
$$

By default:

$$
State(\rho_G^{SIM})\le fragile,
\quad
DashboardSafe=0
$$

unless explicitly promoted by validation.

### 2.8.3 Global Population Loss

The population loss is:

$$
\mathcal{L}(\mathbf{P},\boldsymbol{\eta})
=
\sum_{t\in T}
\left[
\lambda_A\mathcal{L}_{aging}(t)
+
\lambda_B\mathcal{L}_{birth}(t)
+
\lambda_D\mathcal{L}_{death}(t)
+
\lambda_M\mathcal{L}_{migration}(t)
+
\lambda_R\mathcal{L}_{race}(t)
+
\lambda_S\mathcal{L}_{smooth}(t)
\right]
$$

### 2.8.4 Aging Loss

For interior ages:

$$
\mathcal{L}^{int}_{aging}(t)
=
\sum_{s,a=1}^{98,x,r}
\left[
P_{s,t,a+1,x,r}
-
\left(
P_{s,t-1,a,x,r}(1-\delta_{s,t-1,a,x,r})
+
\eta_{s,t-1,a+1,x,r}
\right)
\right]^2
$$

Age-1 boundary:

$$
\mathcal{L}^{a=1}_{aging}(t)
=
\sum_{s,x,r}
\left[
P_{s,t,1,x,r}
-
\left(
P_{s,t-1,0,x,r}(1-\delta_{s,t-1,0,x,r})
+
\eta_{s,t-1,1,x,r}
\right)
\right]^2
$$

Terminal age pool:

$$
\mathcal{L}^{100+}_{aging}(t)
=
\sum_{s,x,r}
\left[
P_{s,t,100+,x,r}
-
\left(
P_{s,t-1,99,x,r}(1-\delta_{s,t-1,99,x,r})
+
P_{s,t-1,100+,x,r}(1-\delta_{s,t-1,100+,x,r})
+
\eta_{s,t-1,100+,x,r}
\right)
\right]^2
$$

### 2.8.5 Birth Loss

$$
\mathcal{L}_{birth}(t)
=
\sum_{s,x,r}
\left[
P_{s,t,0,x,r}
-
\left(
B^{newborn}_{s,t-1,x,r}
+
\eta_{s,t-1,0,x,r}
\right)
\right]^2
$$

Births are separated into maternal and newborn axes:

$$
B^{maternal}_{s,t,a_m,r_m}
$$

for fertility analysis, and:

$$
B^{newborn}_{s,t,x_n,r_n}
$$

for population entry at age zero.

If newborn race/color is unavailable, fallback is ordered:

$$
P(r_n\mid r_m,s)
\to
P(r_n\mid r_m,UF)
\to
P(r_n\mid r_m,Region)
\to
P(r_n\mid r_m,Brazil)
$$

Fallback triggers only when local support is insufficient:

$$
n<30
$$

All imputed newborn-race entries carry:

$$
Prov=birth\_race\_conditional\_allocation
$$

**Implementation.** $B^{newborn}_{s,t,x,r}$ is built from SINASC (`newborn_sex`,
`newborn_race_admin`, `mun_residence_cod6`, `birth_year`), grouped by (municipality,
year, sex) and, since newborn race is administrative (§2.8.1's axis-identity note),
bridged through `pegasus.efg.race_bridge` to $\mathcal{R}$ before landing in a
race-stratified cell. Without a configured Bridge_R prior (`race_tensor_mode=decoupled`
or `downstream_bridge`), births cannot be honestly placed on a real (non-degenerate)
race axis and are left out of $\mathcal{L}_{birth}$ entirely rather than collapsed onto
an unmodeled total — $\lambda_B=0$ for that run.

The $P(r_n\mid r_m,s)$ head of the race fallback cascade **is** used: SINASC exposes
both the newborn's own administrative race (`newborn_race_admin`) and the mother's
(`maternal_race_admin`), and when the newborn's is missing/invalid the mother's declared
race stands in as a first-order estimate of $r_n$ (i.e. $r_n:=r_m$, the leading term of
the cascade), both routed through the same Bridge_R. The deeper $UF\to Region\to Brazil$
conditional-allocation tail (triggered at $n<30$) is not yet a distinct estimator;
Bridge_R's own local-$\pi$ posterior (with bootstrap uncertainty, MSD-II's Bridge_R spec)
is used uniformly regardless of local support size. Implementation:
`pegasus.sidra.population_cube.build._sinasc_birth_priors` (`_coalesce_race_columns` for
the newborn→maternal fallback).

### 2.8.6 Death Prior Loss

$$
\mathcal{L}_{death}(t)
=
\sum_{s,a,x,r}
\left[
\delta_{s,t,a,x,r}P_{s,t,a,x,r}
-
D^{SIM}_{s,t,a,x,r}
\right]^2
$$

This term is a soft prior, never an official identity.

In independent denominator mode, $\lambda_D=0$. In SIM-informed mode, $\lambda_D>0$ and downstream SIM mortality receives feedback-risk warnings.

**Implementation.** $D^{SIM}_{s,t,a,x,r}$ is built from SIM-DO events grouped by
(municipality, year, sex, single-year age bucket) and, like births, bridged from
administrative race to $\mathcal{R}$ through `pegasus.efg.race_bridge` — the same
"no bridge ⇒ left unmodeled on that axis" rule as §2.8.5 applies. $\delta_{s,t,a,x,r}$
itself is currently estimated as the empirical ratio $D^{SIM}_{s,t,a,x,r}/C_{s,t,a,x,r}$
against this tensor's own census strata anchor $C$ — a known limitation, not a design
choice: $C$ is only observed at census years, so this term only activates for deaths
falling in a year with a coincident census anchor, which is the minority case once
Tab 6579 intercensal stitching (§2.8.10) is in play. A rate estimator that does not
require anchor-year coincidence (e.g. a UF/Region/Brazil reference rate, or an
iterative rate implied by the solver's own current population iterate) is future
work. Implementation: `pegasus.sidra.population_cube.build._sim_death_priors`.

### 2.8.7 Migration Smoothness Loss

$$
\mathcal{L}_{migration}(t)
=
\sum_{s,a,x,r}
\left[
\eta_{s,t,a,x,r}
-
2\eta_{s,t-1,a,x,r}
+
\eta_{s,t-2,a,x,r}
\right]^2
$$

Migration enclosure is:

$$
\sum_{s\in S^*}\eta_{s,t,a,x,r}
=
M^{national}_{t,a,x,r}
$$

where:

$$
M^{national}_{t,a,x,r}
=
\begin{cases}
0 & closed\ national \\
\Delta^{residual}_{t,a,x,r} & open\ national\ residual \\
\Psi^{prior}_{t,a,x,r} & external\ migration\ prior
\end{cases}
$$

**Implementation — net-migration residual (the "open national residual" case
$\Delta^{residual}$).** No annual, *(age,sex,race)-stratified* internal-migration flow
source exists (SIDRA's migration tables are census-decennial origin-destination
counts). But the aggregate net migration into a municipality-year is recoverable *by
exclusion* from the demographic balancing equation, which is exactly the residual case
above. From $P_{s,t}=P_{s,t-1}+B_{s,t}-D_{s,t}+\text{NetMig}_{s,t}$:

$$
\widehat{\text{NetMig}}_{s,t}
=
E_{s,t}-E_{s,t-1}-B_{s,t}+D_{s,t}
$$

where $E$ is the §2.8.10 closure total and $B,D$ are annual municipal births/deaths.
This enters the objective as a soft per-locality-year anchor on the total migration
flow, $\lambda_{M_{tot}}\sum_{s,t}\big(\sum_{a,x,r}\eta_{s,t,a,x,r}-\widehat{\text{NetMig}}_{s,t}\big)^2$
(a new term alongside $\mathcal{L}_{migration}$'s smoothness; the two coexist as
§2.8.7 already anticipates). It constrains only the *total* flow — the demographic
composition of migration remains reconstructed, since no source stratifies it.

Vital totals $B,D$ come from the **SIDRA civil-registry** tables (2609 births / 2683
deaths) by preference: they share IBGE's statistical universe with the population
estimates $E$, so the residual isolates migration rather than cross-system coverage
divergence. DATASUS SIM/SINASC counts are the fallback when civil-registry facts were
not acquired. The residual is formed only for **consecutive** calendar years with both
$E_{s,t}$ and $E_{s,t-1}$ present (over a multi-year gap it would be a cumulative, not
annual, flow, and is left unobserved). $\eta$'s box bound is taken from the closure
total $E_{s,t}$ (present every year), not the per-cell census anchor (present only at
census years), so intercensal years — where the residual matters most — are not
starved of migration headroom. Implementation:
`pegasus.sidra.population_cube.build._migration_residual_totals` +
`she.reconstruction.loss` (`migration_total` term). The stratified external prior
$\Psi^{prior}$ remains unwired (no stratified source); the closed-national case
$M^{national}=0$ is still available when no residual is observed.

### 2.8.8 Race/Color Composition Loss

Let:

$$
p_{s,t,a,x,\cdot}\in\Delta^{|\mathcal{R}|-1}
$$

Then:

$$
\mathcal{L}_{race}(t)
=
\sum_{s,a,x}
\left\|
ilr(p_{s,t,a,x,\cdot})
-
z^{bridge}_{s,t,a,x}
\right\|_2^2
$$

Between census anchors $t_0$ and $t_1$:

$$
z^{bridge}_{s,t,a,x}
=
\left(1-\frac{t-t_0}{t_1-t_0}\right)
ilr(C_{s,t_0,a,x,\cdot})
+
\left(\frac{t-t_0}{t_1-t_0}\right)
ilr(C_{s,t_1,a,x,\cdot})
$$

This is a demographic composition path, not a claim that health administrative race and IBGE self-declared race are identical.

**Implementation.** $z^{bridge}_{s,t,a,x}$ is built entirely from this tensor's own
SIDRA 9606 self-declared race strata $C$ at whichever census years are present in the
run — it needs no `pegasus.efg.race_bridge` involvement at all (contrast §2.8.5/§2.8.6):
both ends of the interpolation are already on the $\mathcal{R}$ axis. A run outside
$[t_0,t_1]$ (before the first or after the last observed census year) is clamped to the
nearest census composition rather than left undefined — a conservative extension beyond
this section's literal two-anchor definition, since $\mathcal{L}_{race}$ is a soft prior
and a flat carry-forward is preferable to no prior at all for those years. With fewer
than two census-year race compositions in the run's window (or a degenerate
total-only race axis), $\lambda_R=0$. Implementation:
`pegasus.sidra.population_cube.build._census_race_composition_prior`.

### 2.8.9 Age Smoothness Loss

Let:

$$
\mathcal{A}_{interior}=\{2,\ldots,99\}
$$

Then:

$$
\mathcal{L}_{smooth}(t)
=
\sum_{s,a\in\mathcal{A}_{interior},x,r}
\left[
P_{s,t,a,x,r}
-
2P_{s,t,a-1,x,r}
+
P_{s,t,a-2,x,r}
\right]^2
$$

**Implementation.** $a-1$ and $a-2$ presuppose $\mathcal{A}$ is ordered chronologically
(each index one calendar year older than the last) — the same requirement §2.8.4's
Aging Loss and §2.8.5's Birth Loss (age index 0 = newborn entry) depend on. The
canonical age labels (`age_0`..`age_99`, `age_100_plus`) do not sort into that order
under a plain lexical/alphabetical sort (`"age_10" < "age_2"`); the age axis must be
built with `pegasus.registries.demographic_axis.age_group_sort_key`, not
`sorted()` on the bare strings.

### 2.8.10 Closure Constraints

Annual municipal closure:

$$
\sum_{a,x,r}P_{s,t,a,x,r}=E_{s,t}
$$

Census training anchors:

$$
P_{s,t_c,a,x,r}
=
C^{train}_{s,t_c,a,x,r}
\quad
s\in S_{train}
$$

Validation cells:

$$
C^{val}_{s,t_c,a,x,r}
\quad
s\in S_{val}
$$

are excluded from hard constraints and used only for tuning.

Empty census cell rule:

$$
C_{s,t_c,a,x,r}=0
$$

locks the cell to zero only when no vital-flow entry permits a nonzero trajectory. Otherwise the cell may reopen with:

$$
warning=zero\_anchor\_reopened\_by\_vital\_flow
$$

**Annual closure source (multi-table SIDRA stitching).** The annual total $E_{s,t}$ that the closure constraint anchors to is not observed by a single SIDRA table for every $t$. The canonical municipal-population source, SIDRA Tab 9606 (the full Sex × Race × Age matrix, §2.8.1's $Race\times Sex\times Age$ support), exists only for census years — verified live against IBGE SIDRA, currently $\{2010, 2022\}$. For every other year in a run's window, $E_{s,t}$ is instead taken from SIDRA Tab 6579 ("Estimativas de População", post-censal series): an annual, total-only resident-population estimate with no sex/race/age disaggregation.

$$
E_{s,t}
=
\begin{cases}
E^{9606}_{s,t} & t\ \text{is a census year covered by Tab 9606} \\
E^{6579}_{s,t} & \text{otherwise, when Tab 6579 covers}\ t \\
\text{unanchored} & \text{neither table covers}\ t
\end{cases}
$$

Tab 9606 takes priority whenever both tables cover the same year (a safety net: Tab 6579's own official periods already exclude census years). A year neither table covers — e.g. the 2007 IBGE estimation gap, or the processing lag immediately after a census (currently 2023) — is simply left with no closure anchor for that year, exactly as any other missing closure cell in this section: the reconstruction (aging/birth/death/migration/smoothness losses, §2.8.4–§2.8.9) interpolates it rather than requiring every year anchored. Neither table's exact gap years are hardcoded anywhere in the implementation — the acquisition layer trusts each table's own live SIDRA period metadata, so a resumed Tab 6579 estimate or a new census year requires no spec or code change. Implementation: `pegasus.sidra.population_cube` (`anchor.py` loads each table; `build.py` stitches them into one closure panel); the stitching policy is recorded in `config/registries/sidra/sidra_stitching.yaml`.

Only Tab 9606 supplies the $(a,x,r)$ disaggregation itself (§2.8.1's full support); intercensal years contribute a closure total only, with the age/sex/race breakdown for those years being a *reconstruction*, not an observation — consistent with this tensor's purpose.

### 2.8.11 Hyperparameter Tuning

$$
\Lambda_{pop}
=
(\lambda_A,\lambda_B,\lambda_D,\lambda_M,\lambda_R,\lambda_S)
$$

The tuned hyperparameter vector is:

$$
\widehat{\Lambda}_{pop}
=
\arg\min_{\Lambda}
\sum_{s\in S_{val},t_c,a,x,r}
\left[
\widehat{P}_{s,t_c,a,x,r}(\Lambda)
-
C^{val}_{s,t_c,a,x,r}
\right]^2
+
\kappa\mathcal{L}_{roughness}
$$

### 2.8.12 Population Solver Backend Registry

Let:

$$
|\Omega_{epi}|
=
|S^*|\cdot |T|\cdot |\mathcal{A}|\cdot |\mathcal{S}|\cdot |\mathcal{R}|
$$

The population solver backend is:

$$
Solver_{POP}
\in
\{
projected\_gradient\_small,
regional\_block\_coordinate,
sparse\_ADMM,
state\text{-}space\_smoother
\}
$$

Selection rule:

$$
Solver_{POP}
=
\begin{cases}
projected\_gradient\_small & |\Omega_{epi}|\le 10^7 \\
regional\_block\_coordinate & 10^7<|\Omega_{epi}|\le 5\times10^7 \\
sparse\_ADMM & |\Omega_{epi}|>5\times10^7\land RAM\ constraint\ active \\
state\text{-}space\_smoother & reduced\ UF/region\ path
\end{cases}
$$

Dense finite-difference gradients are forbidden. Gradients, constraint Jacobians, and Hessian approximations are sparse or block-sparse objects.

Any registry entry whose backend is scaffold-only is non-executable. If a scale band or explicit request resolves only to such an entry, the compiler aborts with `population_solver_unavailable_for_scale`; it must not silently downgrade to another backend.

For Brazil-scale execution, the tensor decomposes:

$$
S^*
=
\bigcup_{b=1}^{B_S}S_b
$$

with blocks defined by UFs, health macroregions, AMC clusters, or graph partitions. Boundary consistency is restored through ADMM consensus variables or equivalent coupling.

ADMM form:

$$
\min_{\{P_b,\eta_b\},Z}
\sum_b
\mathcal{L}_b(P_b,\eta_b)
+
\mathcal{R}(Z)
$$

subject to:

$$
A_bP_b=Z_b
$$

Block update:

$$
(P_b^{k+1},\eta_b^{k+1})
=
\arg\min_{P_b,\eta_b}
\mathcal{L}_b(P_b,\eta_b)
+
\frac{\rho}{2}
\left\|
A_bP_b-Z_b^k+u_b^k
\right\|^2
$$

Consensus update:

$$
Z^{k+1}
=
Consensus(\{A_bP_b^{k+1}+u_b^k\})
$$

Dual update:

$$
u_b^{k+1}
=
u_b^k+A_bP_b^{k+1}-Z_b^{k+1}
$$

Termination requires:

$$
\|r^k\|_2\le\epsilon_{pri},
\quad
\|s^k\|_2\le\epsilon_{dual}
$$

The output contract is unchanged:

$$
\widehat{P},
\quad
\widehat{\eta},
\quad
Q(B^{POP}),
\quad
Prov(B^{POP})
$$

## 2.9 SIDRA Contextual Regime Classifier

Each SIDRA field $q$ receives a regime:

$$
R(q)
\in
\{
direct,
harmonize,
deflate,
bounded\_interpolate,
cross\_sectional,
do\_not\_reconstruct
\}
$$

The classifier is:

$$
R(q)
=
\begin{cases}
direct & Missing_t=0\land SchemaStable=1 \\
harmonize & SchemaMismatch=1\land Projectable=1 \\
deflate & Unit(q)\in\{R\$,milR\$,SM\} \\
bounded\_interpolate & IsGated(ST\text{-}DFM)=1 \\
cross\_sectional & IsGated(ST\text{-}DFM)=0\land TemporalPoints\ge1 \\
do\_not\_reconstruct & otherwise
\end{cases}
$$

The ST-DFM gate is:

$$
IsGated(ST\text{-}DFM)
=
\mathbf{1}
\{
IsBounded(anchors)
\land
CompatConcept(q_t,q_{t'})=1
\land
TemporalPoints\ge3
\land
Dynamics(q)\in\{continuous,semi\text{-}continuous,proportion,positive\}
\}
$$

Concept compatibility requires stable universe, denominator, classification version, category boundaries, unit, and locality support.

## 2.10 ST-DFM Contract

For fields with:

$$
R(q)=bounded\_interpolate
$$

SHE defines:

$$
g_q(Y_{s,t,q})
=
\lambda_q^\top F_{s,t}
+
\epsilon_{s,t,q}
$$

Latent transition:

$$
F_{s,t}
=
AF_{s,t-1}
+
BZ_{s,t}
+
\eta_{s,t}
$$

Noise:

$$
\epsilon_{s,t,q}\sim N(0,\sigma_q^2),
\quad
\eta_{s,t}\sim N(0,\Sigma_\eta)
$$

Spatial regularization:

$$
\mathcal{L}_{spatial}
=
\sum_t
tr(F_t^\top L_WF_t)
$$

Estimator:

$$
\widehat{F},\widehat{\Theta}
=
\arg\min_{F,\Theta}
\left[
\sum_{M=1}
\frac{
(g_q(Y_{s,t,q})-\lambda_q^\top F_{s,t})^2
}{\sigma_q^2}
+
\gamma_T\sum_{s,t}\|\Delta_t^2F_{s,t}\|_2^2
+
\gamma_S\mathcal{L}_{spatial}
\right]
$$

### 2.10.1 Link Functions

$$
g_q(Y)
=
\begin{cases}
\log(Y+\epsilon_q) & positive/monetary/countlike \\
\log\frac{Y^*}{1-Y^*} & proportion \\
clr(Y+\epsilon) & simplex \\
identity & approximately\ Gaussian \\
forbidden & median/quantile/noncommensurable
\end{cases}
$$

For proportions:

$$
Y^*
=
\frac{Y(n-1)+0.5}{n}
$$

where $n$ is the official denominator for the SIDRA percentage. If no denominator exists:

$$
Y^*=\frac{Y+\epsilon}{1+2\epsilon},
\quad
warning=proportion\_denominator\_unknown
$$

This fallback cannot produce verified state.

### 2.10.2 Identifiability

The loading matrix is lower triangular with positive diagonal under fixed registry order:

$$
\lambda_{q,k}=0
\quad
k>q
$$

$$
\lambda_{q,q}>0
\quad
q\le K
$$

Factor covariance is fixed:

$$
E[F_{s,t}F_{s,t}^\top]=I_K
$$

Column order is registry-anchored and signs are locked by positive anchor loadings.

### 2.10.3 Multi-Start Stability

For random starts:

$$
\widehat{F}^{(1)},\ldots,\widehat{F}^{(R)}
$$

align factors by Procrustes rotation:

$$
\widetilde{F}^{(r)}
=
Procrustes(\widehat{F}^{(r)},\widehat{F}^{(1)})
$$

Define:

$$
Stability(F)
=
median_{r<r'}
corr(\widetilde{F}^{(r)},\widetilde{F}^{(r')})
$$

Certification requires:

$$
Stability(F)\ge\theta_F
$$

with default:

$$
\theta_F=0.85
$$

### 2.10.4 Certification

Candidate reconstructed field:

$$
\widetilde{X}_{s,t,q}
=
g_q^{-1}(\widehat{\lambda}_q^\top\widehat{F}_{s,t})
$$

Certification checks include:

$$
MAPE_{holdout},
\quad
RMSE_{spatial},
\quad
\sigma_q^2/Var(Y_q),
\quad
boundary\_violations,
\quad
Stability(F)
$$

State assignment:

$$
State(\widetilde X_q)
=
\begin{cases}
verified & MAPE\le\theta^{valid}_{MAPE}\land Ratio_\sigma\le\theta^{valid}_{\sigma}\land Stability(F)\ge\theta_F \\
fragile & MAPE\le\theta^{fragile}_{MAPE}\land Ratio_\sigma\le\theta^{fragile}_{\sigma} \\
illegal\_excluded & otherwise
\end{cases}
$$

Default thresholds:

$$
\theta^{valid}_{MAPE}=0.15,
\quad
\theta^{fragile}_{MAPE}=0.35
$$

$$
\theta^{valid}_{\sigma}=0.25,
\quad
\theta^{fragile}_{\sigma}=0.40
$$

Latent uncertainty propagates into $Q(v)$:

$$
Var_{latent}(v)
\leftarrow
Var(\widehat{F}_{s,t})+\sigma_q^2
$$

## 2.11 Civil–Health Substrate Divergence

SIDRA Civil Registry tables are admissible as independent measurement-system comparators, not as replacements.

Death-system divergence:

$$
v^{death}_{civil\_health\_divergence}(s,t)
=
\log
\left(
\frac{
D^{SIM}_{s,t,all}+\epsilon
}{
D^{SIDRA\_death}_{s,t,all}+\epsilon
}
\right)
$$

Birth-system divergence:

$$
v^{birth}_{civil\_health\_divergence}(s,t)
=
\log
\left(
\frac{
B^{SINASC}_{s,t,all}+\epsilon
}{
B^{SIDRA\_birth}_{s,t,all}+\epsilon
}
\right)
$$

These nodes are observer-process fields:

$$
Role(v)=observer\_proxy
$$

$$
Prov(v)=civil\_health\_measurement\_comparison
$$

Before divergence is computed, alignment evaluates:

$$
Align_{civil\_health}
$$

over:

- year basis
- geographic basis
- residence, occurrence, or registration basis
- event definition
- coverage correction process

If SIDRA is by registration year and DATASUS is by occurrence year:

$$
warning=registration\_occurrence\_temporal\_mismatch
$$

If divergence exceeds threshold:

$$
|v_{civil\_health\_divergence}(s,t)|>\tau_{sys}
$$

the system adds:

$$
system\_measurement\_divergence\_risk
$$

to affected fields. It does not automatically invalidate SIM or SINASC.

## 2.12 Longitudinal Stitching and Classification Projection

SIDRA contextual and demographic concepts are not guaranteed to exist as a single stable table across all years. The SHE must construct concept-stable temporal fields before exposing them to the EFG or to ST-DFM.

A SIDRA raw fact is:

$$
F^{SIDRA}
=
(
tab,
var,
t,
s,
\chi,
y,
meta
)
$$

where $\chi$ is the full classification-category tuple.

A SIDRA concept is:

$$
q
=
(concept,\ measure,\ universe,\ unit,\ denominator,\ classification\_semantics,\ source\_survey)
$$

The SHE must not treat table identity as concept identity:

$$
tab_A=tab_B
\not\Leftrightarrow
q_A=q_B
$$

and:

$$
tab_A\neq tab_B
\not\Rightarrow
q_A\neq q_B
$$

### 2.12.1 Longitudinal Stitching Contract

Let a concept $q$ be observed through table segments:

$$
\mathcal{T}_q
=
\{Tab_1,\ldots,Tab_M\}
$$

with temporal supports:

$$
T_m=[t_m^-,t_m^+]
$$

and segment-specific observations:

$$
Y^{(m)}_{s,t,\chi,q}
\quad
t\in T_m
$$

The SHE may construct a stitched temporal field:

$$
Y^\dagger_{s,t,\chi,q}
$$

only if the concept-compatibility predicate is satisfied:

$$
CompatStitch(Tab_m,Tab_n,q)=1
$$

for every adjacent or overlapping segment required by the stitch.

The compatibility predicate is:

$$
CompatStitch
=
C_{universe}
C_{measure}
C_{unit}
C_{denominator}
C_{classification}
C_{locality}
C_{period}
C_{method}
$$

where every factor must be nonzero or explicitly bridged.

The stitched field is:

$$
Y^\dagger_{s,t,\chi,q}
=
\phi_m
\left(
Y^{(m)}_{\tau_m(s),t,\psi_m(\chi),q}
\right)
\quad
\text{for}
\quad
t\in T_m
$$

where:

- $\tau_m$ maps segment geography into $S^*$.
- $\psi_m$ maps segment classifications into canonical classification tuples.
- $\phi_m$ is a registered comparability transform.

Allowed comparability transforms are:

$$
\phi_m
\in
\{
identity,
deflate,
unit\_scale,
category\_projection,
overlap\_calibrated\_affine,
overlap\_calibrated\_logscale
\}
$$

If adjacent segments overlap over $T_{mn}=T_m\cap T_n$, calibration may estimate:

$$
\widehat{\theta}_{mn}
=
\arg\min_{\theta}
\sum_{s,t\in T_{mn},\chi}
\left[
g(Y^{(n)}_{s,t,\chi,q})
-
g(\phi_{\theta}(Y^{(m)}_{s,t,\chi,q}))
\right]^2
$$

where $g$ is the link function appropriate to unit and aggregation law.

If no overlap exists, stitching is legal only under registry-declared methodological continuity:

$$
NoOverlap(Tab_m,Tab_n)
\land
MethodContinuity(Tab_m,Tab_n,q)=1
\Rightarrow
State(Y^\dagger)\le fragile
$$

If methodological continuity is absent:

$$
CompatStitch=0
\Rightarrow
Y^\dagger\in\mathcal{B}_{excluded}
$$

with:

$$
warning=longitudinal\_concept\_break
$$

Examples of mandatory concept-level stitching include:

$$
GDP:
Tab\ 21 \to Tab\ 5938
$$

and:

$$
Labor/CEMPRE:
Tab\ 1685 \to Tab\ 9509
$$

The stitch must preserve segment provenance:

$$
Prov(Y^\dagger)
=
JoinProv(Prov(Tab_1),\ldots,Prov(Tab_M),LongitudinalStitch)
$$

and must emit:

$$
StitchMetadata(Y^\dagger)
=
(
segments,
periods,
transforms,
overlap\_diagnostics,
methodological\_warnings,
state
)
$$

ST-DFM may consume only $Y^\dagger$ or a declared single-segment field. It must not infer continuity across disjoint SIDRA tables by table-name similarity alone.

### 2.12.2 Classification Projection Matrix

SIDRA classifications are arbitrary source tuples. The EFG operates on canonical axes:

$$
\mathcal{A},
\quad
\mathcal{S},
\quad
\mathcal{R},
\quad
\mathcal{C}
$$

Therefore SHE must project SIDRA classification tuples before EFG materialization.

Let:

$$
\chi
=
((c_1,k_1),\ldots,(c_J,k_J))
$$

be a SIDRA classification-category tuple, where $c_j$ is a classification identifier and $k_j$ is a source category identifier.

Define the projection:

$$
\Pi_{Clsf\to Axis}^{(q)}
:
\mathbb{R}^{\mathcal{K}_{c_1}\times\cdots\times\mathcal{K}_{c_J}}
\to
\mathbb{R}^{\mathcal{A}\times\mathcal{S}\times\mathcal{R}\times\mathcal{C}}
$$

with entries:

$$
\Pi^{(q)}_{\alpha,\chi}
=
P(axis=\alpha\mid classification\ tuple=\chi,q)
$$

where $\alpha$ is a canonical axis cell.

For exact category mappings:

$$
\Pi^{(q)}_{\alpha,\chi}\in\{0,1\}
$$

For interval, overlapping, or partially compatible mappings:

$$
0\le\Pi^{(q)}_{\alpha,\chi}\le1
$$

and the output receives:

$$
warning=classification\_projection\_fractional
$$

For additive measures:

$$
Y^{axis}_{s,t,\alpha,q}
=
\sum_{\chi}
\Pi^{(q)}_{\alpha,\chi}
Y^{raw}_{s,t,\chi,q}
$$

For proportions or rates, projection is legal only through numerator and denominator recovery:

$$
p^{axis}
=
\frac{
\sum_{\chi}\Pi_{\alpha,\chi}N_{\chi}
}{
\sum_{\chi}\Pi_{\alpha,\chi}D_{\chi}
}
$$

Direct projection of proportions is illegal unless the source denominator is explicitly available and the weighted operation is registered.

Total categories are not ordinary categories. If $k_j$ is a total category:

$$
k_j=Total
$$

then it represents marginalization over that classification:

$$
k_j=Total
\Rightarrow
\pi_{*,c_j}
$$

It must not be mixed with non-total categories inside a modeled axis unless the registry declares a total-only view.

Race/color projections from SIDRA census tables map to:

$$
\mathcal{R}^{IBGE}_{self}
$$

not to administrative health-system race axes.

Age projections from SIDRA categories map to intervals:

$$
k_j\mapsto [a^-,a^+]
$$

Single-year age support is exact only when the source category is single-year. Five-year or open-ended cohorts remain intervals unless an allocation kernel is declared.

The projected field carries:

$$
ProjectionMetadata(v)
=
(
source\_classifications,
source\_categories,
target\_axes,
projection\_matrix\_id,
total\_category\_policy,
fractional\_mapping\_warnings
)
$$

### 2.12.3 High-Dimensional SIDRA Exposure Bounding at SHE Boundary

SHE must not expose unbounded high-dimensional SIDRA matrices directly to the EFG.

For a SIDRA fact family $Y_{s,t,\chi,q}$, define raw cell cardinality:

$$
Card(Y)
=
|S^*|
\cdot
|T|
\cdot
\prod_{j=1}^{J}|\mathcal{K}_{c_j}|
\cdot
|Vars(q)|
$$

Define the EFG-demanded axis set:

$$
Axes_{demand}(q,\mathcal{I},Operator)
\subseteq
\{\mathcal{A},\mathcal{S},\mathcal{R},\mathcal{C}\}
$$

The bounded SHE export is:

$$
Export_{SHE\to EFG}(Y)
=
\pi_{*,Axes_{drop}}
\left(
\Pi_{Clsf\to Axis}^{(q)}Y
\right)
$$

where:

$$
Axes_{drop}
=
Axes(Y)\setminus Axes_{demand}(q,\mathcal{I},Operator)
$$

If:

$$
Card(Y)>Card_{max}^{EFG}
$$

or:

$$
\prod_{j=1}^{J}|\mathcal{K}_{c_j}|>H_{max}^{SIDRA}
$$

then this bounded projection and pushforward is mandatory before EFG node creation.

For SIDRA Tab 9606:

$$
Y_{9606}
=
Population(s,t,r,x,a)
$$

with high-dimensional support:

$$
Race\times Sex\times Age
$$

The SHE may preserve the normalized long-form fact store, but the EFG receives only demanded marginals, such as:

$$
Population(s,t)
$$

$$
Population(s,t,x)
$$

$$
Population(s,t,a,x)
$$

$$
Population(s,t,a,x,r)
$$

when explicitly required and legality permits.

If a requested marginal cannot be produced without illegal aggregation:

$$
Export_{SHE\to EFG}(Y)\to FailedBranch
$$

with:

$$
reason=high\_dimensional\_projection\_illegal
$$

# 3. Epidemiological Field Graph

## 3.1 EFG Definition

The Epidemiological Field Graph is the autonomous compiler that expands admissible substrate fields into typed epidemiological variables.

It builds a directed acyclic graph:

$$
\mathcal{G}_{EFG}
=
(V_{fields},E_{DAG})
$$

A node is a field:

$$
v:L_v\to\mathbb{R}
$$

where:

$$
L_v
\subseteq
S^*(\mathcal{M}_{geo})
\times
T
\times
\mathcal{A}
\times
\mathcal{S}
\times
\mathcal{R}
\times
\mathcal{H}
\times
\mathcal{F}
\times
\mathcal{C}
$$

The graph is a field graph, not a per-cell graph. Each node represents a field over a support. Cell arrays live in tabular or tensor artifacts; nodes live in the field registry.

## 3.2 Node Metadata

Each node carries:

$$
\Gamma(v)
=
(
id,
name,
kind,
carrier,
unit,
L_v,
axes,
aggregation,
role,
source,
operator,
provenance,
state,
warnings,
lineage
)
$$

Lineage is:

$$
Lineage(v)
=
(parent\_ids,\ operator\_type,\ operator\_params,\ registry\_versions)
$$

Node identity is content-addressed:

$$
id(v)=SHA256(Lineage(v))
$$

Allowed node kinds are:

$$
kind
\in
\{
extensive\_measure,
intensive\_density,
marked\_functional,
context\_gradient,
bridge\_divergence,
bridge\_module,
observer\_proxy,
latent\_context
\}
$$

Allowed roles are multi-valued:

$$
role(v)
\subseteq
\{
outcome,
exposure\_offset,
covariate,
demographic,
capacity,
observer,
exploratory,
dashboard,
model\_only
\}
$$

## 3.3 Carrier Ontology

The carrier registry defines admissible numerator-denominator semantics. EFG nodes must refer to explicit carriers, not vague administrative aggregates.

The carrier set includes:

$$
\mathcal{K}_{carrier}
=
\{
Deaths,
HospitalAdmissions,
LiveBirths,
BirthsToWomen,
Population,
Women15\_49,
Facilities,
FacilityCapacityVector_k,
HospitalCosts_{SH},
HospitalCosts_{SP},
HospitalCosts_{UTI},
HospitalCosts_{TOT},
HospitalDays,
ICUDays,
Domiciles,
DomicilesWithSanitation,
Physicians_k
\}
$$

where:

$$
k\in\mathcal{K}_{capacity}
$$

is a mandatory facility-capacity vector index.

Core legal relations include:

| Numerator carrier | Denominator carrier | Role |
|---|---|---|
| Deaths | Population | mortality_rate |
| HospitalAdmissions | Population | hospitalization_rate |
| LiveBirths | Population | birth_rate |
| BirthsToWomen | Women15_49 | fertility_rate |
| InfantDeaths | LiveBirths | infant_mortality |
| NeonatalDeaths | LiveBirths | neonatal_mortality |
| PostNeonatalDeaths | LiveBirths | postneonatal_mortality |
| MaternalDeaths | LiveBirths | maternal_mortality |
| LowBirthWeightBirths | LiveBirths | birth_outcome_share |
| PretermBirths | LiveBirths | birth_outcome_share |
| CesareanBirths | LiveBirths | delivery_share |
| CongenitalAnomalies | LiveBirths | anomaly_prevalence |
| HospitalDeaths | HospitalAdmissions | inpatient_mortality |
| ICUAdmissions | HospitalAdmissions | ICU_use_share |
| ICUDays | HospitalAdmissions | mean_icu_days |
| HospitalDays | HospitalAdmissions | mean_LOS |
| HospitalCosts_SH | HospitalAdmissions | mean_cost_facility |
| HospitalCosts_SP | HospitalAdmissions | mean_cost_professional |
| HospitalCosts_UTI | HospitalAdmissions | mean_cost_icu |
| HospitalCosts_TOT | HospitalAdmissions | mean_cost_total_billing |
| FacilityCapacityVector_k | Population | capacity_density_k |
| FacilityCapacityVector_k | Facilities | mean_capacity_per_facility_k |
| Facilities | Population | facility_density |
| Physicians_k | Population | workforce_density_k |
| DomicilesWithSanitation | Domiciles | sanitation_coverage |

The following generic carrier is forbidden:

$$
Beds
$$

unless rewritten as:

$$
FacilityCapacityVector_k
\quad
\text{with}
\quad
k\in\mathcal{K}_{leit}
$$

Therefore:

$$
RN(Beds,Population)
\Rightarrow
\Delta_{carrier}=0
$$

but:

$$
RN(FacilityCapacityVector_{QTLEITP3},Population)
\Rightarrow
\Delta_{carrier}=1
$$

when all other legality terms hold.

Similarly, generic hospital cost is not a primitive carrier. The carrier must specify:

$$
HospitalCosts_{SH},
\quad
HospitalCosts_{SP},
\quad
HospitalCosts_{UTI},
\quad
HospitalCosts_{TOT}
$$

The EFG may derive a total-billing burden from `VAL_TOT`, but it must not collapse component-specific cost fields into a single economic carrier unless the operator explicitly declares a component-summing estimand and preserves component provenance.

## 3.4 Unit Ontology

The unit registry defines admissible unit transformations. Units are attached to decoded canonical fields, not raw administrative strings.

| Numerator unit | Denominator unit | Output unit | Role |
|---|---|---|---|
| counts | person-years | rate | mortality_rate, hospitalization_rate, birth_rate |
| counts | counts | proportion | shares, fatality, ICU use, birth outcomes |
| reais_hospital_services | admissions | mean_cost_facility | SIH `VAL_SH` economic burden |
| reais_professional_services | admissions | mean_cost_professional | SIH `VAL_SP` economic burden |
| reais_icu_services | admissions | mean_cost_icu | SIH `VAL_UTI` economic burden |
| reais_total_billing | admissions | mean_cost_total_billing | SIH `VAL_TOT` total billing burden |
| days | admissions | mean_duration | length of stay |
| icu_days | admissions | mean_icu_days | ICU utilization |
| grams | live births | mean_weight | birth-weight functionals |
| facility_capacity_units_k | person-years | capacity_density_k | CNES vector-indexed capacity density |
| facility_capacity_units_k | facilities | mean_capacity_per_facility_k | CNES vector-indexed facility capacity |
| facilities | person-years | facility_density | facility density |
| physicians_k | person-years | workforce_density_k | workforce density |
| domiciles | domiciles | coverage | sanitation and household coverage |

The monetary-unit registry separates SIH cost components:

$$
unit(VAL\_SH)=reais\_hospital\_services
$$

$$
unit(VAL\_SP)=reais\_professional\_services
$$

$$
unit(VAL\_UTI)=reais\_icu\_services
$$

$$
unit(VAL\_TOT)=reais\_total\_billing
$$

These units are not interchangeable:

$$
reais\_hospital\_services
\not\equiv
reais\_professional\_services
\not\equiv
reais\_icu\_services
\not\equiv
reais\_total\_billing
$$

Therefore:

$$
\Delta_{unit}=0
$$

for any operator that pools `VAL_SH`, `VAL_SP`, and `VAL_UTI` as if they shared the same economic estimand.

A component-summing operator is legal only when declared as:

$$
CostComponentSum:
(reais\_hospital\_services,
reais\_professional\_services,
reais\_icu\_services)
\to
reais\_declared\_composite
$$

and only when it records:

$$
ComponentProvenance
=
(VAL\_SH,VAL\_SP,VAL\_UTI)
$$

CNES capacity units are vector-indexed:

$$
unit(QTLEITP3)=facility\_capacity\_units_{QTLEITP3}
$$

$$
unit(QTINST14)=facility\_capacity\_units_{QTINST14}
$$

Generic `beds` is not an admissible unit unless a registry-defined aggregation explicitly maps a set of bed-vector indices into a declared composite bed family.

Thus:

$$
unit(Beds)=illegal
$$

unless:

$$
Beds
=
\sum_{k\in K_{bed\_composite}}FacilityCapacityVector_k
$$

with:

$$
K_{bed\_composite}
$$

declared in the capacity registry.

## 3.5 Aggregation Laws

Each field carries one aggregation law:

$$
aggregation
\in
\{
additive,
weighted\_mean,
statistical\_functional,
compositional,
non\_aggregable
\}
$$

Counts may be pushed forward by summation. Rates and proportions require numerator/denominator recovery or explicit weighted averaging. Medians and quantiles are non-aggregable unless a distributional model exists.

Vector-indexed facility capacity is additive only within the same capacity index:

$$
FacilityCapacityVector_k+FacilityCapacityVector_k
\quad
\text{is legal}
$$

but:

$$
FacilityCapacityVector_i+FacilityCapacityVector_j
$$

is illegal unless a declared composite capacity family contains both $i$ and $j$.

Cost components are additive only inside declared economic-composite operators. They are not interchangeable units.

## 3.6 Provenance Algebra

Primitive provenance labels are:

$$
Prov_0
=
\{
official,
harmonized,
deflated,
AMC\_contracted,
geneallocated,
reconstructed,
SIM\_informed\_denominator\_prior,
latent,
synthetic,
forced\_fragile,
bridge\_derived,
model\_derived,
observer\_rerouted,
bayesian\_race\_axis\_bridge,
civil\_health\_measurement\_comparison,
classification\_projected,
longitudinally\_stitched,
facility\_linkage\_filtered,
zero\_variance\_excluded
\}
$$

Propagation is:

$$
Prov(child)
=
JoinProv(Prov(parent_1),\ldots,Prov(parent_k),operator)
$$

Rules:

| Parent provenance | Child consequence |
|---|---|
| any synthetic | model_only unless developer override. |
| any latent | latent warning inherited; dashboard unsafe by default. |
| any SIM_informed_denominator_prior with SIM numerator | variance inflation warning. |
| any forced_fragile | descendants inherit forced warning. |
| observer_rerouted | disease outcome role blocked. |
| geneallocated | inferential warning depending on geo mode. |
| bridge_derived | interpretation label required. |
| bayesian_race_axis_bridge | race-axis warning and uncertainty required. |
| classification_projected | projection metadata required. |
| longitudinally_stitched | stitch metadata and segment provenance required. |
| facility_linkage_filtered | linkage-quality warning required when link support is incomplete. |
| zero_variance_excluded | audit-only; cannot enter PIRS. |

## 3.7 Alignment Contract

Alignment decomposes by axis:

$$
Align(\nu,\mu)
=
\prod_{d\in\{\mathcal{A},\mathcal{S},\mathcal{R},\mathcal{H},S^*,T,\mathcal{F},\mathcal{C}\}}
Align_d(\nu,\mu)
$$

### 3.7.1 Spatial Alignment

If spatial supports differ:

- If both map to AMC, contract to $\bar S$.
- If one is native and structural break exists, reject lagged transformation.
- If geneallocated mode is enabled, use admissible $\mathcal{A}_t$ only for additive components.
- Otherwise reject.

### 3.7.2 Temporal Alignment

Counts aggregate from monthly to annual by summation.

Rates and proportions aggregate only by reconstructing numerator and denominator or by legal weighted averaging.

Medians and quantiles are non-aggregable unless a distributional model exists.

### 3.7.3 Age Alignment

Counts aggregate by summation:

$$
m([a,b])=\sum_{j=a}^{b}m(j)
$$

Rates/proportions require numerator and denominator recovery. Non-nested intervals are rejected unless an explicit allocation kernel exists.

### 3.7.4 Sex and Race Alignment

Sex and race use projection/crosswalk matrices only when the declaration-process semantics are compatible.

$$
\Pi_{raw\to canon}
$$

Unknown, Missing, and Other remain distinct:

$$
Unknown\neq Missing\neq Other
$$

Silent redistribution is forbidden unless a declared allocation policy is used.

### 3.7.5 Health-Event Alignment

ICD groups use code-set incidence matrices:

$$
M_{G,c}\in\{0,1\}
$$

Fine groups may be pushed to coarse ancestor groups. Overlapping non-nested groups require overlap contracts or branch splitting.

Underlying-cause mortality, ordered terminal chains, associated conditions, SIH principal diagnosis, SIH secondary comorbidities, and SINASC congenital anomaly diagnoses are distinct health-axis objects.

### 3.7.6 Facility Alignment

Facility-level CNES fields may aggregate to municipality/year support.

SIH facility linkage through corporate identifiers is fragile unless:

$$
Filter_{CNPJ}(SIH.id).state=ValidCNPJ
$$

and:

$$
Filter_{CNPJ}(CNES.id).state=ValidCNPJ
$$

and a linkage contract is declared.

All-zero identifiers are null links, not shared facilities.

### 3.7.7 Context Alignment

SIDRA classification tuples are preserved in the normalized fact store. EFG exposure requires projection through:

$$
\Pi_{Clsf\to Axis}
$$

and, when needed, high-dimensional bounded pushforward.

## 3.8 Seven-Part Structural Legality Predicate with Declaration Gate

The EFG evaluates a seven-part structural legality predicate:

$$
\Delta_{struct}
=
\Delta_{support}
\Delta_{axes}
\Delta_{carrier}
\Delta_{unit}
\Delta_{aggregation}
\Delta_{provenance}
\Delta_{quality}
$$

A transformation is structurally legal only if every term is nonzero.

PegaSUS then applies a mandatory epistemic declaration-process gate:

$$
\Delta
=
\Delta_{struct}
\Delta_{declaration}
$$

Thus the production legality predicate is structurally seven-part and epistemically declaration-aware.

### 3.8.1 Support Term

$$
\Delta_{support}=1
\iff
L_\nu\cap L_\mu\neq\varnothing
$$

after alignment.

### 3.8.2 Axes Term

The axes term evaluates whether parent fields occupy compatible canonical axes after SHE-level decoding, SIDRA classification projection, longitudinal stitching, and required high-dimensional bounding.

$$
\Delta_{axes}=1
$$

iff:

$$
Axes(\nu')=Axes(\mu')
$$

or an operator-specific axis contract permits the mismatch.

Before this comparison, every source-specific axis representation must be normalized:

$$
Axes^{raw}
\xrightarrow{SHE}
Axes^{canonical}
$$

For SIDRA facts:

$$
Axes^{canonical}
=
\Pi_{Clsf\to Axis}^{(q)}(ClsfTuple)
$$

For composite DATASUS fields:

$$
Axes^{canonical}
=
Decode(raw\_field)
$$

For diagnostic fields:

$$
Axes^{canonical}
=
(code\_set,\ diagnostic\_role,\ topology,\ position)
$$

not merely an ICD code set.

The axes term must also enforce an axis-demand bound. Let:

$$
Axes_{demand}(v)
$$

be the axes required by the active intent, operator, model role, and registry entry. Let:

$$
Axes_{raw}(v)
$$

be the axes available after SHE projection.

If:

$$
Axes_{raw}(v)\supset Axes_{demand}(v)
$$

and the extra axes are legally marginalizable, the EFG must apply immediate pushforward:

$$
v'
=
\pi_{*,Axes_{raw}\setminus Axes_{demand}}(v)
$$

before node materialization.

If the field is high-cardinality:

$$
Card(v)>Card_{max}^{EFG}
$$

or:

$$
\prod_{d\in Axes_{raw}}|d|>H_{max}^{EFG}
$$

then pushforward is mandatory:

$$
Card(v)>Card_{max}^{EFG}
\Rightarrow
v_{EFG}
=
\pi_{*,drop}
\left(
\Pi_{Clsf\to Axis}v_{SHE}
\right)
$$

The uncompressed high-dimensional node must not be materialized in $V_{fields}$ unless it is explicitly selected by:

$$
\mathcal{V}_0
\cup
\mathcal{R}_{force}
$$

and passes memory, legality, and state gates.

For additive measures, high-dimensional pushforward is:

$$
(\pi_{*,D}Y)(l_{-D})
=
\sum_{l_D}Y(l_D,l_{-D})
$$

For rates and proportions, pushforward is legal only through numerator and denominator recovery:

$$
\pi_{*,D}\left(\frac{\nu}{\mu}\right)
\quad
\text{is illegal}
$$

unless rewritten as:

$$
\frac{\pi_{*,D}\nu}{\pi_{*,D}\mu}
$$

For non-aggregable statistical functionals:

$$
\Delta_{axes}=0
$$

unless a distributional reconstruction operator is registered.

Therefore:

$$
\Delta_{axes}=0
$$

when a high-dimensional field requires marginalization but its aggregation law forbids the required pushforward.

### 3.8.3 Carrier Term

$$
\Delta_{carrier}=1
\iff
(Carrier_\nu,Carrier_\mu,Role)\in\mathcal{K}_{carrier}
$$

Generic beds, generic costs, and unsanitized corporate linkages fail this term.

### 3.8.4 Unit Term

$$
\Delta_{unit}=1
\iff
(Unit_\nu,Unit_\mu,Role)\in\mathcal{K}_{unit}
$$

Distinct SIH economic components and vector-indexed CNES capacity units must not be collapsed without a registered composite operator.

### 3.8.5 Aggregation Term

$$
\Delta_{aggregation}=1
$$

iff the operator respects the parent fields’ aggregation laws.

### 3.8.6 Provenance Term

$$
\Delta_{provenance}=1
$$

iff:

$$
ProvPolicy(operator,parents)
\in
\{
allowed,
allowed\_with\_warning,
model\_only,
report\_only
\}
$$

Blocked states include:

$$
synthetic+synthetic,
\quad
illegal\_excluded,
\quad
observer\_rerouted\to disease\ outcome
$$

### 3.8.7 Quality Term

$$
\Delta_{quality}=1
\iff
Perm(State(parent),operator)=1
$$

### 3.8.8 Declaration-Process Term

$$
\Delta_{declaration}=1
$$

iff every semantic axis used in numerator and denominator is declaration-process commensurable.

For race/color:

$$
\Delta_{declaration}^{race}=1
$$

iff:

$$
RaceAxis(\nu)=RaceAxis(\mu)
$$

or:

$$
\nu'=Bridge_{\mathcal{R}}(\nu)
\land
RaceAxis(\nu')=self\_declared\_aligned
$$

Otherwise:

$$
\Delta_{declaration}^{race}=0
$$

This prevents valid-looking but epistemologically invalid rates.

## 3.9 Operator Registry

The EFG operator registry is:

$$
\mathcal{O}
=
\{
\sigma_C,
\pi_*,
\pi^*,
\Pi_{Clsf\to Axis},
\pi^{bound}_*,
RN,
\Psi_\varphi,
Std_W,
Comp_\epsilon,
Lag_\tau,
Bridge,
Contrast,
Shrink,
CostComponentSum,
ModelResidual
\}
$$

| Operator | Meaning |
|---|---|
| $\sigma_C$ | Restriction by condition $C$. |
| $\pi_*$ | Pushforward/marginalization over legal axes. |
| $\pi^*$ | Contextual pullback only; never denominator broadcasting. |
| $\Pi_{Clsf\to Axis}$ | Projection of SIDRA classification-category tuples into canonical EFG axes. |
| $\pi^{bound}_*$ | Mandatory early pushforward at the SHE/EFG boundary for high-cardinality fields. |
| $RN(\nu,\mu)$ | Finite-cell Radon–Nikodym density. |
| $\Psi_\varphi$ | Marked functional: mean, median, variance, quantile when legal. |
| $Std_W$ | Direct standardization over basis $W$. |
| $Comp_\epsilon$ | Zero-safe compositional transform. |
| $Lag_\tau$ | Lag operator over valid temporal support. |
| $Bridge$ | Cross-system grammar instantiation. |
| $Contrast$ | Difference, log-ratio, or spatial contrast. |
| $Shrink$ | Empirical/Bayesian shrinkage view. |
| $CostComponentSum$ | Declared summation of compatible SIH economic components into an explicit composite burden. |
| $ModelResidual$ | Residual field extraction from PIRS. |

### 3.9.1 Bounded Pushforward Operator

The bounded pushforward operator is mandatory for high-dimensional facts whose full raw axis product exceeds EFG cardinality limits.

Let:

$$
v:L_v\to\mathbb{R}
$$

and let:

$$
L_v
=
L_{keep}\times L_{drop}
$$

Then:

$$
\pi^{bound}_*(v)(l_{keep})
=
\sum_{l_{drop}\in L_{drop}}v(l_{keep},l_{drop})
$$

for additive measures.

For ratios, rates, and proportions:

$$
\pi^{bound}_*
\left(
\frac{\nu}{\mu}
\right)
$$

is forbidden. The legal operation is:

$$
RN
\left(
\pi^{bound}_*\nu,
\pi^{bound}_*\mu
\right)
=
\frac{
\pi^{bound}_*\nu
}{
\pi^{bound}_*\mu
}
$$

For compositional fields, bounded pushforward must preserve closure:

$$
Comp_\epsilon
\left(
\pi^{bound}_* \mathbf{x}
\right)
$$

only after the composition denominator is recomputed.

For non-aggregable functionals:

$$
\pi^{bound}_*(\Psi_\varphi)
$$

is illegal unless the registry declares a distributional reconstruction law.

### 3.9.2 Classification Projection Operator

The classification projection operator is:

$$
\Pi_{Clsf\to Axis}^{(q)}
$$

It is executed before $\Delta_{axes}$ and before high-cardinality materialization.

For additive fields:

$$
Y^{axis}
=
\Pi_{Clsf\to Axis}^{(q)}Y^{raw}
$$

For non-additive fields, the operator must either recover the source numerator/denominator or fail.

The operator must preserve:

$$
ProjectionMetadata
=
(
source\_classification,
source\_category,
target\_axis,
projection\_matrix,
total\_category\_policy,
warnings
)
$$

### 3.9.3 Operator Ordering at the SHE/EFG Boundary

For SIDRA fields, the mandatory boundary order is:

$$
Y^{raw}
\xrightarrow{LongitudinalStitch}
Y^\dagger
\xrightarrow{\Pi_{Clsf\to Axis}}
Y^{axis}
\xrightarrow{\pi^{bound}_*}
Y^{bounded}
\xrightarrow{\Delta}
V_{fields}
$$

The EFG must not evaluate utility pruning on an unbounded high-dimensional SIDRA field when a legal bounded pushforward exists.

The full normalized SIDRA fact store may remain in SHE storage, but $V_{fields}$ receives only legal bounded fields unless the high-dimensional field is explicitly forced and passes memory and legality gates.

### 3.9.4 Cost Component Sum Operator

The cost component sum operator is legal only when the target estimand is a declared composite economic burden.

$$
CostComponentSum:
\prod_{j\in J}HospitalCosts_j
\to
HospitalCosts_{declared\_composite}
$$

where:

$$
J\subseteq\{SH,SP,UTI,TOT\}
$$

The operator is illegal if it mixes `VAL_TOT` with components that are already part of `VAL_TOT` unless the registry declares that the total field is used for validation rather than summation.

The operator must emit:

$$
ComponentProvenance
=
\{VAL\_SH,VAL\_SP,VAL\_UTI,VAL\_TOT\}\cap J
$$

and:

$$
warning=cost\_component\_composite
$$

## 3.10 Core Seed Registry

The seed set is:

$$
V_{seed}
=
V_{core}
\cup
\mathcal{V}_0
\cup
\mathcal{H}_0
$$

The mandatory core families are:

$$
V_{core}
=
V_M
\cup
V_H
\cup
V_B
\cup
V_C
\cup
V_K
\cup
V_O
\cup
V_X
$$

### 3.10.1 Mortality Fields

| ID | Field | Operator | Numerator | Denominator | Role |
|---|---|---|---|---|---|
| V_M01 | CrudeMortality | RN | all SIM deaths | population | outcome, demographic |
| V_M02 | ChapterMortality | RN | SIM deaths by ICD chapter, underlying cause | population | outcome |
| V_M03 | BlockMortality | RN | SIM deaths by ICD block, underlying cause | population | outcome |
| V_M04 | CuratedCauseMortality | RN | SIM deaths by curated cause group, underlying cause | population | outcome |
| V_M05 | TerminalChainMentionShare | $\pi^{chain\to mention}_*$ / RN | terminal-chain mentions | deaths | observer, exploratory |
| V_M06 | AssociatedConditionMentionShare | RN | associated-condition mentions | deaths | covariate, observer |

### 3.10.2 Hospitalization Fields

| ID | Field | Operator | Numerator | Denominator | Role |
|---|---|---|---|---|---|
| V_H01 | CrudeHospitalization | RN | all SIH admissions | population | outcome |
| V_H02 | DiagnosisHospitalization | RN | SIH admissions by principal diagnosis | population | outcome |
| V_H03 | InpatientFatality | RN | SIH hospital deaths | SIH admissions | outcome, covariate |
| V_H04 | DiagnosisSpecificLOS | $\Psi_{mean}$ | stay length by principal diagnosis | admissions | covariate |
| V_H05A | DiagnosisSpecificHospitalServiceCost | $\Psi_{mean}$ | `VAL_SH` by principal diagnosis | admissions | covariate |
| V_H05B | DiagnosisSpecificProfessionalCost | $\Psi_{mean}$ | `VAL_SP` by principal diagnosis | admissions | covariate |
| V_H05C | DiagnosisSpecificICUCost | $\Psi_{mean}$ | `VAL_UTI` by principal diagnosis | admissions | covariate |
| V_H05D | DiagnosisSpecificTotalBillingCost | $\Psi_{mean}$ | `VAL_TOT` by principal diagnosis | admissions | covariate |
| V_H06 | ICUUseShare | RN | ICU admissions | admissions | covariate |
| V_H07 | SecondaryComorbidityMentionShare | RN | SIH secondary-diagnosis mentions | admissions | covariate, observer |

### 3.10.3 Fertility and Birth Fields

| ID | Field | Operator | Numerator | Denominator | Role |
|---|---|---|---|---|---|
| V_B01 | CrudeBirthRate | RN | SINASC live births | population | demographic, outcome |
| V_B02 | GeneralFertility | RN | births to women 15–49 | women 15–49 | demographic, covariate |
| V_B03 | AgeSpecificFertility | RN | births by maternal age | women by age | demographic |

### 3.10.4 Maternal-Child Fields

| ID | Field | Operator | Numerator | Denominator | Role |
|---|---|---|---|---|---|
| V_C01 | InfantMortality | RN | infant deaths | live births | outcome |
| V_C02 | NeonatalMortality | RN | neonatal deaths | live births | outcome |
| V_C03 | PostNeonatalMortality | RN | postneonatal deaths | live births | outcome |
| V_C04 | PerinatalMortality | RN | perinatal deaths | live births plus eligible fetal deaths | outcome |
| V_C05 | MaternalMortality | RN | strict maternal deaths | live births | outcome |
| V_C06 | LowBirthWeight | RN | weight under 2500 g | live births | outcome |
| V_C07 | VeryLowBirthWeight | RN | weight under 1500 g | live births | outcome |
| V_C08 | PrematurityShare | RN | gestation under 37 weeks | live births | outcome |
| V_C09 | CesareanShare | RN | cesarean births | live births | covariate, outcome |
| V_C10 | LowApgar1 | RN | APGAR1 under 7 | live births | outcome |
| V_C11 | LowApgar5 | RN | APGAR5 under 7 | live births | outcome |
| V_C12 | KotelchuckAdequacy | RN | Kotelchuck group | live births | covariate |
| V_C13 | RobsonGroupShare | RN | Robson group | live births | covariate |
| V_C14 | CongenitalAnomaly | RN | anomaly flag yes | live births | outcome |

### 3.10.5 Capacity and Health-System Fields

| ID | Field | Operator | Numerator | Denominator | Role |
|---|---|---|---|---|---|
| V_K01_k | CapacityDensity_k | RN | FacilityCapacityVector_k | population | capacity, covariate |
| V_K02_k | SUSRestrictedCapacityDensity_k | RN | SUS-restricted FacilityCapacityVector_k | population | capacity, covariate |
| V_K03_k | WorkforceDensity_k | RN | Physicians_k or workforce vector k | population | capacity, covariate |
| V_K04 | FacilityDensity | RN | facilities | population | capacity, covariate |
| V_K05 | MeanLengthOfStay | $\Psi_{mean}$ | stay length | admissions | covariate |
| V_K06 | ICUDaysPerHosp | $\Psi_{mean}$ | ICU days | admissions | covariate |
| V_K07A | MeanHospitalServiceCost | $\Psi_{mean}$ | `VAL_SH` | admissions | covariate |
| V_K07B | MeanProfessionalCost | $\Psi_{mean}$ | `VAL_SP` | admissions | covariate |
| V_K07C | MeanICUCost | $\Psi_{mean}$ | `VAL_UTI` | admissions | covariate |
| V_K07D | MeanTotalBillingCost | $\Psi_{mean}$ | `VAL_TOT` | admissions | covariate |
| V_K08 | InvalidFacilityFlagShare | RN | invalid CNES boolean flags | facilities | observer |
| V_K09 | ZeroCNPJShare | RN | nullified all-zero CNPJ records | facilities | observer |

### 3.10.6 Observer-Process Fields

| ID | Field | Operator | Definition |
|---|---|---|---|
| V_O01 | ReportingDelaySIM | $\Psi_{median}$ | median SIM reporting delay |
| V_O02 | ReportingDelaySINASC | $\Psi_{median}$ | median SINASC reporting delay |
| V_O03 | CauseAlterationShare | RN | cause altered / all deaths |
| V_O04 | InvestigationShare | RN | investigated / all deaths |
| V_O05 | IllDefinedCauseShare | RN | ill-defined ICD group / all deaths |
| V_O06 | InvalidICDShare | RN | invalid or unparseable ICD / all records |
| V_O07 | MissingRaceShare_source | RN | missing/unknown race over records, by source |
| V_O08 | CivilHealthDivergence | Bridge | SIM/SINASC versus SIDRA Civil Registry divergence |
| V_O09 | ZeroVarianceDroppedColumns | audit | excluded constant or all-missing source fields |

### 3.10.7 Context Fields

| ID | Field | Operator | Definition |
|---|---|---|---|
| V_X01 | RealGDPPerCapita | RN/deflated/stitch | real GDP / population, longitudinally stitched if table split exists |
| V_X02 | IncomeContext | contextual | income distributions or admissible medians |
| V_X03 | SanitationCoverage | RN | domiciles with sanitation / domiciles |
| V_X04 | EducationContext | contextual | literacy and schooling fields |
| V_X05 | LaborContext | contextual/stitch | labor and economic activity fields with table-segment stitching when required |
| V_X06 | IndividualAggregateContextGradient | Bridge | DATASUS socioeconomic marks against SIDRA context |

### 3.10.8 Curated SIDRA Compendium as the Context Ingestion Target

The context families $V_X$ (§3.10.7) and the SIDRA contextual routing of §2.9–§2.12 are
not open-ended: their concrete national-coverage input is a **curated compendium of
epidemiologically relevant SIDRA/IBGE tables** with near-complete coverage of all 5,570
Brazilian municipalities. The compendium is the operational catalogue that populates the
$V_X$ context space and feeds the regime classifier (§2.9), classification projection
(§2.12.2), longitudinal stitching (§2.12.1), and — where the ST-DFM gate (§2.9) is met —
latent reconstruction (§2.10).

Contract:

- The compendium is maintained as a machine-readable registry
  (`config/registries/sidra_compendium.json`, built from `SIDRA_COMPENDIUM.md`) listing
  each table with tier (`T1_CORE`…), variables (id, unit, type, default-keep), and
  classifications (axis, categories). Ingestion is **driven by this registry**, by tier
  and default-keep flags — never by hardcoded table lists.
- Each ingested SIDRA field enters SHE as a `ContextCells` carrier, is assigned a regime
  $R(q)$ (§2.9), has its classifications projected to canonical axes (§2.12.2), and is
  admitted to the EFG as a `context_gradient` / `latent_context` field. Latent-derived
  fields carry the §3.6 latent-provenance quarantine (dashboard-unsafe by default).
- A field is ST-DFM reconstructed (§2.10) **only** when its regime is
  `bounded_interpolate` and the gate holds (bounded anchors, concept compatibility,
  ≥3 temporal points, admissible dynamics). Single-period or concept-broken fields remain
  `cross_sectional` / `direct` and are never interpolated. This makes the compendium the
  source of a large, statistically-certified context variable set that the EFG and PIRS
  dissect — the latent-reconstruction philosophy of the system — rather than a fixed
  hand-picked set of covariates.

> Implementation status (2026-06-26): the compendium registry and the §2.9/§2.10/§2.12
> machinery exist as modules but are not yet wired into the live SHE→EFG ingestion path,
> and require the curated context tables (and multi-year/disaggregated data for ST-DFM and
> the §2.8 demographic tensor) to be acquired before end-to-end execution.

## 3.11 ICD and Health-Event Ontology

The health-event ontology is:

$$
\mathfrak{I}
=
(
\mathcal{C}_{ICD},
\preceq,
\mathcal{G}_{native},
\mathcal{G}_{curated},
\mathcal{G}_{quality},
\mathcal{G}_{explore},
\mathcal{T}_{diagnostic}
)
$$

where $\mathcal{T}_{diagnostic}$ is the diagnostic-topology registry.

A health diagnostic object is not only a code set. It is:

$$
H
=
(
\mathcal{C}_H,
source,
diagnostic\_role,
topology,
position,
parse\_state,
quality\_class
)
$$

with:

$$
topology
\in
\{
single\_underlying,
single\_principal,
ordered\_terminal\_chain,
unordered\_associated\_set,
unordered\_secondary\_set,
single\_anomaly
\}
$$

and:

$$
diagnostic\_role
\in
\{
underlying\_cause,
principal\_diagnosis,
terminal\_chain,
associated\_condition,
secondary\_comorbidity,
congenital\_anomaly
\}
$$

### 3.11.1 Code-Set Ontology

Each ICD group $G$ maps to:

$$
\mathcal{C}_G\subseteq\mathcal{C}_{ICD}
$$

Overlap is:

$$
Overlap(G_1,G_2)
=
\frac{
|\mathcal{C}_{G_1}\cap\mathcal{C}_{G_2}|
}{
|\mathcal{C}_{G_1}\cup\mathcal{C}_{G_2}|
}
$$

If:

$$
Overlap\ge0.95
$$

the groups may collapse as code-set equivalents only when their diagnostic topology is also compatible.

If:

$$
0.05<Overlap<0.95
$$

the groups split into disjoint difference sets before traversal.

### 3.11.2 Diagnostic Topology Compatibility

Health-axis legality is:

$$
\Delta_{\mathcal{H}}
=
\Delta_{\mathcal{C}_{ICD}}
\Delta_{role}
\Delta_{topology}
\Delta_{position}
$$

where:

$$
\Delta_{\mathcal{C}_{ICD}}=1
$$

for identical code sets, legal ancestor collapse, or curated group membership.

The role term is:

$$
\Delta_{role}=1
$$

iff the diagnostic roles are identical or a registered bridge permits the comparison.

The topology term is:

$$
\Delta_{topology}=1
$$

iff:

$$
topology(H_1)=topology(H_2)
$$

or a topology-erasing projection is explicitly declared.

The position term is:

$$
\Delta_{position}=1
$$

for ordered terminal-chain fields only when position is identical or the operator explicitly marginalizes chain position:

$$
\pi^{chain\to mention}_*
$$

with the resulting field downgraded to observer or exploratory role.

### 3.11.3 Forbidden Health-Axis Equivalences

The following direct equivalences are illegal:

$$
H^{SIM}_{underlying}
\equiv
H^{SIM}_{terminal\_chain}
$$

$$
H^{SIM}_{underlying}
\equiv
H^{SIM}_{associated}
$$

$$
H^{SIM}_{terminal\_chain}
\equiv
H^{SIH}_{secondary}
$$

$$
H^{SIM}_{underlying}
\equiv
H^{SIH}_{principal}
$$

$$
H^{SINASC}_{anomaly}
\equiv
H^{SIM}_{underlying}
$$

unless a bridge grammar explicitly defines the estimand.

Thus:

$$
\Delta_{\mathcal{H}}=0
$$

for underlying-cause mortality versus principal-diagnosis hospitalization as a direct same-axis comparison. Such relations must route through morbidity–mortality bridge grammars.

### 3.11.4 Ordered Terminal Chain Rules

For SIM-DO Part I chain fields:

$$
H^{SIM}_{chain}(e)
=
(h_A,h_B,h_C,h_D)
$$

with:

$$
A\prec B\prec C\prec D
$$

The EFG may construct:

$$
ChainPositionRate_G^{p}
$$

for:

$$
p\in\{A,B,C,D\}
$$

only when the field remains position-specific.

The EFG may construct:

$$
ChainMentionShare_G
=
\pi^{chain\to mention}_*
H^{SIM}_{chain}
$$

only as:

$$
Role\in\{observer,exploratory,covariate\}
$$

It must not label this as cause-specific mortality.

### 3.11.5 Unordered Comorbidity and Associated-Condition Rules

For SIH secondary diagnoses:

$$
H^{SIH}_{secondary}(e)=\{h_1,\ldots,h_K\}
$$

and SIM associated conditions:

$$
H^{SIM}_{associated}(e)=\{h_{II,1},\ldots,h_{II,K}\}
$$

the EFG may construct:

$$
ComorbidityMention_G(e)
=
\mathbf{1}
\{\exists h\in H_{unordered}(e):h\in G\}
$$

These fields may be covariates, observer fields, or multimorbidity descriptors. They are not incident disease outcomes unless a registry-defined estimand states otherwise.

### 3.11.6 Quality Rerouting

If:

$$
G\in\mathcal{G}_{quality}
$$

then:

$$
Role(v)=observer\_proxy
$$

and:

$$
DiseaseOutcomeSafe=0,
\quad
ObserverFieldSafe=1
$$

Quality groups are allowed as observer-process rates, not biological disease outcomes.

### 3.11.7 Traversal Gate

Diagnostic descent is allowed when:

$$
Descend(G)=1
$$

iff:

$$
n_{events}(G)\ge50
\land
cov_S(G)\ge0.10
\land
cov_T(G)\ge0.80
\land
Frag(G)\le0.05
$$

or:

$$
G\in\mathcal{H}_0
\cup
\mathcal{R}_{force}
\cup
\mathcal{G}_{curated}
$$

Traversal must be topology-specific. A group that passes support thresholds as a secondary-comorbidity mention does not automatically pass as an underlying-cause mortality field.

## 3.12 State Tensor

Every field has a state tensor:

$$
Q(v)
=
(
n_{events},
n_{denom},
n_{eff},
cov_S,
cov_T,
missingness,
zero\_inflation,
denom\_fragility,
CV,
MoranI,
temporal\_roughness,
spatial\_entropy,
provenance\_risk,
state
)
$$

### 3.12.1 Event Count

$$
n_{events}(v)
=
\sum_{s,t}\nu_{num}(s,t)
$$

For marked functionals, this is the number of valid contributing records.

### 3.12.2 Denominator Count

$$
n_{denom}(v)
=
\sum_{s,t}\mu_{den}(s,t)
$$

For marked functionals:

$$
n_{denom}=n_{events}
$$

### 3.12.3 Effective Sample Size

$$
n_{eff}(v)
=
\frac{
\left(\sum_{s,t}w_{s,t}\right)^2
}{
\sum_{s,t}w_{s,t}^2
}
\cdot
\frac{1}{1+\max(0,MoranI(v))}
$$

For counts:

$$
w_{s,t}=\nu_{num}(s,t)
$$

For rates/proportions:

$$
w_{s,t}=\mu_{den}(s,t)
$$

### 3.12.4 Spatial Coverage

$$
cov_S(v)
=
\frac{1}{|S^*|}
\sum_s
\mathbf{1}
\left\{
\sum_t valid(v(s,t))>0
\right\}
$$

### 3.12.5 Temporal Coverage

$$
cov_T(v)
=
\frac{1}{|T|}
\sum_t
\mathbf{1}
\left\{
\sum_s valid(v(s,t))>0
\right\}
$$

### 3.12.6 Missingness

$$
missingness(v)
=
\frac{
\sum_{s,t}\mathbf{1}\{v(s,t)=null\}
}{
|S^*||T|
}
$$

Invalid decoded flags, nullified linkage identifiers, and unusable sentinel values contribute to missingness or observer-state penalties under their respective registry rules.

### 3.12.7 Zero Inflation

$$
\zeta(v)
=
\frac{
\sum_{s,t}\mathbf{1}\{v(s,t)=0\}
}{
|S^*||T|
}
$$

### 3.12.8 Denominator Fragility

$$
denom\_fragility(v)
=
\frac{
\sum_{s,t}\mathbf{1}\{\mu_{den}(s,t)<\mu_{min}\}
}{
|S^*||T|
}
$$

Default:

$$
\mu_{min}=50
$$

for highly stratified cells, configurable by family.

### 3.12.9 Coefficient of Variation

For Poisson counts:

$$
CV(Y)=\frac{1}{\sqrt{Y+\epsilon}}
$$

For rates:

$$
Var(Y/\mu)\approx\frac{Y}{\mu^2}
$$

and:

$$
CV=
\frac{
\sqrt{Y/\mu^2}
}{
Y/\mu+\epsilon
}
$$

For binomial proportions:

$$
Var(\hat p)=\frac{\hat p(1-\hat p)}{n}
$$

For latent fields:

$$
Var^*(v)=Var_{observed}(v)+Var_{latent}(v)
$$

### 3.12.10 Moran’s I

$$
MoranI(v)
=
\frac{|S^*|}{\sum_{i,j}W_{ij}}
\frac{
\sum_{i,j}W_{ij}(v_i-\bar v)(v_j-\bar v)
}{
\sum_i(v_i-\bar v)^2
}
$$

### 3.12.11 Temporal Roughness

$$
R_T(v)
=
\frac{1}{|S^*|(|T|-2)}
\sum_{s,t}
\left[
v(s,t)-2v(s,t-1)+v(s,t-2)
\right]^2
$$

### 3.12.12 Spatial Entropy

$$
H_S(v)
=
-\sum_s p_s\log p_s
$$

where:

$$
p_s
=
\frac{\sum_t\nu_{num}(s,t)}
{\sum_{i,t}\nu_{num}(i,t)}
$$

### 3.12.13 Provenance Risk

$$
P_{risk}(v)
=
\begin{cases}
0.0 & official \\
0.2 & harmonized/deflated/AMC\_contracted \\
0.3 & classification\_projected/longitudinally\_stitched \\
0.4 & reconstructed/geneallocated \\
0.5 & SIM\_informed\_denominator\_prior/facility\_linkage\_filtered \\
0.8 & latent \\
1.0 & synthetic
\end{cases}
$$

### 3.12.14 State Assignment

$$
State(v)=Classify(Q(v),\mathcal{R}_{force})
$$

Default state rules:

$$
verified
\iff
n_{eff}\ge100
\land
denom\_fragility<0.05
\land
missingness<0.10
\land
P_{risk}<0.5
$$

$$
fragile
\iff
n_{eff}\ge30
\land
denom\_fragility<0.20
$$

$$
forced\_fragile
\iff
v\in\mathcal{R}_{force}
\land
State(v)\notin\{verified,fragile\}
$$

$$
quarantined\_descriptive
\iff
n_{eff}<30
\lor
denom\_fragility\ge0.20
$$

$$
illegal\_excluded
\iff
\Delta=0
\lor
critical\ missingness
\lor
noncommensurable
$$

## 3.13 Quarantine Permissions

Field classes are:

$$
Class(v)
\in
\{
verified,
fragile,
forced\_fragile,
quarantined\_descriptive,
quarantined\_nochildren,
illegal\_excluded
\}
$$

| Class | Model outcome | Model covariate | Generate RN | ICD descend | Dashboard safe |
|---|---:|---:|---:|---:|---:|
| verified | 1 | 1 | 1 | 1 | 1 |
| fragile | 1 | 1 | 0 | 1 | warning |
| forced_fragile | 1 | 1 | 0 | 0 | 0 |
| quarantined_descriptive | 0 | 1 | 0 | 0 | 0 |
| quarantined_nochildren | 0 | 0 | 0 | 0 | 0 |
| illegal_excluded | 0 | 0 | 0 | 0 | 0 |

Zero-variance fields cannot be promoted beyond audit-only status.

## 3.14 Utility Score

The intent-conditioned utility score is:

$$
U_{\mathcal{I}}(v)
=
\alpha_1 Core(v)
+
\alpha_2 Rel_{\mathcal{I}}(v)
+
\alpha_3 SourceWeight(v)
+
\alpha_4 Bridge(v)
+
\alpha_5 Quality(v)
-
\alpha_6 Complexity(v)
-
\alpha_7 Cost(v)
$$

Default weights:

$$
\alpha=(10,5,3,4,2,1.5,0.5)
$$

Quality is:

$$
Quality(v)=1-denom\_fragility-CV-P_{risk}
$$

Nodes below:

$$
\theta_{compile}=12.5
$$

are not eagerly materialized unless they are in:

$$
V_{core},
\quad
\mathcal{V}_0,
\quad
\mathcal{R}_{force}
$$

## 3.15 Two-Stage Equivalence Compression

EFG compression is:

$$
Compression
=
Compression^{pre}_{topological}
+
Compression^{post}_{empirical}
$$

### 3.15.1 Stage 1 — Topological Pre-Compression

Before materialization, EFG evaluates:

$$
Equiv_{topo}(X_i,X_j)
$$

using metadata only:

$$
Lineage,
Operator,
ParentIDs,
Axes,
Support,
Provenance,
Carrier,
Unit,
RegistryVersion
$$

Topological equivalence holds when lineage is identical or exact algebraic equivalence is provable.

Topological compression includes:

- nested ICD redundancy
- exact projection redundancy
- shared denominator identity
- compositional closure identity
- zero-variance exclusion before DAG creation
- cost-component identity checks
- capacity-vector index identity checks

No numerical field arrays are required.

### 3.15.2 Stage 2 — Post-Selection Empirical Compression

Empirical compression is allowed only after candidate pruning by utility, budget, TopK, forced fields, and active health seeds.

For materialized candidates:

$$
\mathbb{C}_{emp}(X_i,X_j)
=
\max
\left(
|\rho_{Pearson}(X_i,X_j)|,
|\rho_{Spearman}(X_i,X_j)|
\right)
$$

If:

$$
\mathbb{C}_{emp}\ge0.98
$$

the lower-utility node is compressed into the dominant equivalence class.

No data are destroyed. Compressed nodes remain represented in $E_{DAG}$ and variable dictionaries.

# 4. Measurement-Process Epistemology and Race/Color Bridge

## 4.1 Race/Color as Measurement-Process-Indexed Axis

Race/color is not a bare categorical axis.

The invalid simplification is:

$$
\mathcal{R}
=
\{\text{Branca},\text{Preta},\text{Parda},\text{Amarela},\text{Indígena}\}
$$

The correct representation is:

$$
\mathcal{R}^{(source)}
=
(categories,\ declaration\ process,\ missingness\ process,\ source,\ period,\ protocol)
$$

Thus:

$$
\mathcal{R}^{IBGE}_{self}
\neq
\mathcal{R}^{SIM}_{admin}
\neq
\mathcal{R}^{SIH}_{billing}
\neq
\mathcal{R}^{SINASC}_{mixed}
$$

Direct numerator/denominator operations across non-commensurable race axes are mathematically illegal unless mediated by a declared bridge.

## 4.2 Race-Axis Registry

Default source race-axis labels are:

| Source | Axis type | Meaning |
|---|---|---|
| IBGE Census/population denominators | self_declared | Census self-declared race/color. |
| SIM-DO | administrative_death_declaration | Race/color recorded in death-declaration workflow. |
| SIH-RD | billing_record | Administrative/billing race/color. |
| SINASC | administrative_mixed | Birth-declaration workflow race/color. |
| CNES | none/not applicable | Facility registry; not a person-race axis by default. |
| SIDRA contextual tables | table-specific | Classified by source metadata. |

## 4.3 Illegal Direct Race-Specific Division

The compiler must reject:

$$
RN
\left(
\nu^{SIM}_{R_{admin}=r},
\mu^{POP}_{R_{self}=r}
\right)
$$

unless:

$$
\nu^{SIM}_{R_{admin}}
$$

is explicitly transformed by:

$$
Bridge_{\mathcal{R}}
$$

Therefore:

$$
\Delta_{declaration}^{race}=0
$$

for direct administrative health numerator divided by self-declared IBGE denominator.

The system may compute raw administrative distributions:

$$
\nu^{SIM}_{R_{admin}=r},
\quad
\nu^{SIH}_{R_{billing}=r},
\quad
\nu^{SINASC}_{R_{mixed}=r}
$$

but labels them as administrative race distributions, not self-declared race-specific risks.

## 4.4 Race/Color Estimand Taxonomy

PegaSUS distinguishes four race/color estimand families.

### 4.4.1 Raw Administrative Race Count

$$
\nu^{raw}_j(k)
$$

where:

$$
j\in\mathcal{R}_{health}
$$

and:

$$
k=(s,t,a,x,G,source)
$$

Role:

$$
Role=descriptive
$$

Dashboard display is allowed with measurement-process label.

### 4.4.2 Missing/Unknown Race Observer Field

$$
MissingRaceShare^{source}(s,t)
=
\frac{
\nu^{source}_{R=Missing/Unknown}(s,t)
}{
\nu^{source}_{all}(s,t)
}
$$

Role:

$$
Role=observer\_proxy
$$

This field is always preserved and never destroyed by redistribution.

### 4.4.3 Self-Aligned Bayesian Race Estimate

$$
\widehat{\nu}^{self}_i(k)
$$

generated by $Bridge_{\mathcal{R}}$.

This is a model-derived numerator aligned to IBGE self-declared denominator axis:

$$
Prov=bayesian\_race\_axis\_bridge
$$

Default:

$$
State\le fragile
$$

unless local calibration exists.

### 4.4.4 Sensitivity Interval Race Estimate

$$
\rho_i^{lower}(k),
\quad
\rho_i^{upper}(k)
$$

computed over a plausible uncertainty set:

$$
C\in\mathcal{S}_C
$$

High-stakes inequality claims must report sensitivity intervals or posterior credible intervals, not point estimates alone.

## 4.5 Bayesian Ecological Race-Axis Bridge

The race bridge is not deterministic correction to “true race.” It is a probabilistic sensitivity operator.

$$
\boxed{
Bridge_{\mathcal{R}}
\text{ estimates denominator-aligned race/color numerators under explicit misclassification assumptions.}
}
$$

It must not overwrite raw race/color counts.

### 4.5.1 Indexing

Let:

$$
k=(s,t,a,x,G,source)
$$

Let self-declared categories be:

$$
i\in\mathcal{R}_{self}
=
\{Branca,Preta,Parda,Amarela,Indígena\}
$$

Let observed administrative categories be:

$$
j\in\mathcal{R}_{obs}
=
\{Branca,Preta,Parda,Amarela,Indígena,Missing/Unknown\}
$$

Observed administrative numerator:

$$
\widetilde{\nu}^{(k)}_j
$$

Self-declared population denominator:

$$
\widehat{P}^{(k)}_i
$$

Local self-declared baseline:

$$
\pi_i^{(k)}
=
\frac{
\widehat{P}^{(k)}_i
}{
\sum_{m\in\mathcal{R}_{self}}\widehat{P}^{(k)}_m
}
$$

### 4.5.2 Source-Specific Emission Matrix

For each source:

$$
q\in\{SIM,SIH,SINASC\}
$$

define:

$$
C^{(q)}\in[0,1]^{5\times6}
$$

where:

$$
C^{(q)}_{i,j}
=
P(R_{obs}=j\mid R_{self}=i,source=q)
$$

Rows sum to one:

$$
\sum_j C^{(q)}_{i,j}=1
$$

The emission matrix may vary by:

- source
- region
- period
- age
- sex
- facility type

Default prior:

$$
C^{(q)}\sim Dirichlet(\alpha^{(q)})
$$

The prior object is versioned:

$$
\mathcal{R}_{race\_bridge}
=
(source,region,period,\alpha,C_{mean},provenance,uncertainty)
$$

Hardcoded matrices are allowed only as priors, never as ground truth.

### 4.5.3 Bayesian Local Posterior Crosswalk

Given $C^{(q)}$ and $\pi^{(k)}$, define:

$$
W^{(k,q)}_{j,i}
=
P(R_{self}=i\mid R_{obs}=j,k,q)
$$

By Bayes’ theorem:

$$
W^{(k,q)}_{j,i}
=
\frac{
C^{(q)}_{i,j}\pi_i^{(k)}
}{
\sum_{m\in\mathcal{R}_{self}}
C^{(q)}_{m,j}\pi_m^{(k)}
}
$$

The denominator-aligned numerator estimate is:

$$
\widehat{\nu}^{(k)}_i
=
\sum_{j\in\mathcal{R}_{obs}}
\widetilde{\nu}^{(k)}_j
W^{(k,q)}_{j,i}
$$

The aligned race-specific rate is:

$$
\widehat{\rho}^{(k)}_i
=
\frac{
\widehat{\nu}^{(k)}_i
}{
\widehat{P}^{(k)}_i
}
$$

This field receives:

$$
Prov=bayesian\_race\_axis\_bridge
$$

## 4.6 Bridge Uncertainty Propagation

The bridge propagates uncertainty from:

$$
\widetilde{\nu},
\quad
C,
\quad
\pi
$$

The fixed-$W$ approximation is allowed only in fast mode:

$$
Var(\widehat{\nu}_i)
=
\sum_jW_{j,i}^2\widetilde{\nu}_j
$$

Standard and deep modes use posterior simulation.

Draw:

$$
C^{(b)}\sim Dirichlet(\alpha)
$$

$$
\pi^{(b)}\sim Distribution(\widehat{P},U_{POP})
$$

$$
\widetilde{\nu}^{(b)}_j
\sim
Poisson(\widetilde{\nu}_j)
$$

or a source-appropriate count model.

Then:

$$
W^{(b)}_{j,i}
=
\frac{
C^{(b)}_{i,j}\pi^{(b)}_i
}{
\sum_m C^{(b)}_{m,j}\pi^{(b)}_m
}
$$

$$
\widehat{\nu}^{(b)}_i
=
\sum_j
\widetilde{\nu}^{(b)}_jW^{(b)}_{j,i}
$$

$$
\widehat{\rho}^{(b)}_i
=
\frac{
\widehat{\nu}^{(b)}_i
}{
\widehat{P}^{(b)}_i
}
$$

Outputs are:

$$
E(\widehat{\rho}_i),
\quad
Median(\widehat{\rho}_i),
\quad
CI_{95\%}(\widehat{\rho}_i),
\quad
Var(\widehat{\rho}_i)
$$

Race-bridge coefficient of variation is:

$$
CV_{racebridge}
=
\frac{
SD_b(\widehat{\rho}^{(b)}_i)
}{
E_b(\widehat{\rho}^{(b)}_i)+\epsilon
}
$$

Then:

$$
Q(v)
\leftarrow
Q(v)+CV_{racebridge}+P_{risk}(Bridge_{\mathcal{R}})
$$

## 4.7 Partial Identification and Sensitivity Bounds

When the emission prior is weak, contested, or geographically mismatched, the bridge emits sensitivity bounds.

Define:

$$
\mathcal{S}_C
=
\left\{
C:
C_{i,j}\in[l_{i,j},u_{i,j}],
\sum_jC_{i,j}=1
\right\}
$$

Then:

$$
\rho^{lower}_i(k)
=
\inf_{C\in\mathcal{S}_C}
\frac{
\sum_j\widetilde{\nu}^{(k)}_jW^{(k)}_{j,i}(C)
}{
\widehat{P}^{(k)}_i
}
$$

and:

$$
\rho^{upper}_i(k)
=
\sup_{C\in\mathcal{S}_C}
\frac{
\sum_j\widetilde{\nu}^{(k)}_jW^{(k)}_{j,i}(C)
}{
\widehat{P}^{(k)}_i
}
$$

High-stakes inequality claims report:

$$
[\rho^{lower}_i,\rho^{upper}_i]
$$

or posterior credible intervals.

## 4.8 Race Missingness Preservation

The Missing/Unknown race category is included in the bridge but never destroyed.

The bridge emits:

$$
MissingRaceShare^{source}(s,t)
$$

as an observer field.

The module output is:

$$
M_{\mathcal{R}}
=
(
\nu^{raw}_{obs},
MissingRaceShare,
\widehat{\nu}^{self\_aligned},
Var(\widehat{\nu}),
RaceBridgeWarnings
)
$$

Missingness contributes to posterior allocation and remains visible as data-quality process.

## 4.9 Race Bridge State Rules

Race-bridged rates are never silently promoted to verified state.

Default:

$$
State(Bridge_{\mathcal{R}}(v))\le fragile
$$

Promotion to verified requires all conditions:

$$
n_{eff}\ge100
$$

$$
CV_{racebridge}<\theta_{raceCV}
$$

$$
MissingRaceShare<\theta_{missingR}
$$

$$
PriorStrength(C)\ge\theta_C
$$

$$
LocalCalibration=1
$$

If no local calibration exists:

$$
DashboardSafe=0
$$

unless explicitly configured as sensitivity display.

Default thresholds:

$$
\theta_{raceCV}=0.25,
\quad
\theta_{missingR}=0.20
$$

## 4.10 Race-Adjusted State Tensor Extension

For fields generated by $Bridge_{\mathcal{R}}$, $Q(v)$ extends to:

$$
Q_{\mathcal{R}}(v)
=
(
RaceAxisSource,
RaceAxisTarget,
MissingRaceShare,
EmissionPriorStrength,
RaceBridgeCV,
SensitivityWidth,
BridgeMode
)
$$

where:

$$
SensitivityWidth_i
=
\frac{
\rho^{upper}_i-\rho^{lower}_i
}{
E(\rho_i)+\epsilon
}
$$

Downgrade triggers:

$$
MissingRaceShare>\theta_{missingR}
\Rightarrow
State\le fragile
$$

$$
SensitivityWidth>\theta_{sens}
\Rightarrow
State\le quarantined\_descriptive
$$

## 4.11 Reporting Rules for Race/Color Outputs

Allowed estimand labels:

| Label | Meaning |
|---|---|
| administrative_race_count | Raw health-system race/color counts. |
| administrative_race_distribution | Distribution of recorded race/color within health records. |
| self_aligned_race_rate_posterior | Bayesian bridge estimate aligned to IBGE self-declared denominator. |
| self_aligned_race_rate_sensitivity_interval | Partial-identification/sensitivity interval. |
| race_missingness_observer_field | Data-quality/observer-process field. |

Forbidden labels:

$$
true\ race\ rate,
\quad
corrected\ race\ rate,
\quad
real\ race\ distribution,
\quad
bias\text{-}free\ race\ estimate
$$

The bridge attenuates numerator-denominator mismatch. It does not eliminate it.

# 5. Bridge Grammar Registry

## 5.1 Bridge Grammar Definition

Bridge grammars instantiate cross-system relations as typed fields or bridge modules.

A bridge module is:

$$
M:L\to\mathbb{R}^k
$$

but each component must materialize as a scalar node:

$$
v_j:L\to\mathbb{R}
$$

The bridge grammar registry is:

$$
\mathcal{B}_{grammar}
=
\{
MM,
MC,
HO,
DQ,
CapacityOutcome,
BirthContext,
ContextGradient,
FacilityFlow,
ObserverCapacity,
CivilHealthDivergence,
Bridge_{\mathcal{R}}
\}
$$

## 5.2 Morbidity–Mortality Divergence

For group $G$:

$$
v_{MM,G}(s,t)
=
\log
\frac{
\epsilon+\rho^{SIM}_{G,underlying}(s,t)
}{
\epsilon+\rho^{SIH}_{G,principal}(s,t)
}
$$

Interpretation:

$$
morbidity\text{-}mortality\ divergence
$$

It is not case fatality.

## 5.3 Maternal-Child Stratum Bridge

$$
v_{MC,K,G}(s,t)
=
\frac{
InfantDeaths_G^{SIM}(s,t)
}{
Births_K^{SINASC}(s,t-\ell)
}
$$

Legal only if:

$$
Births_K>0
$$

$$
cov_S(Births_K)\ge0.85
$$

$$
K\in\{LowBirthWeight,Prematurity,VeryLowBirthWeight\}
$$

Warnings:

$$
ecological\_stratum\_proxy,
\quad
not\_individual\_linkage
$$

## 5.4 Hospital Outcome Bridge

$$
v_{HO,G}(s,t)
=
\frac{
HospitalDeaths_G^{SIH}(s,t)
}{
Admissions_G^{SIH}(s,t)
}
$$

Label:

$$
hospital\_mortality\_per\_admission
$$

## 5.5 Diagnostic Quality Bridge

Diagnostic-quality module:

$$
M_{DQ}
=
(
median\_reporting\_delay,
illdefined\_share,
invalid\_ICD\_share,
investigation\_share
)
$$

Optional quality index:

$$
v_{DQ,index}
=
z(median\_delay)
+
z(illdefined\_share)
+
z(invalid\_ICD\_share)
-
z(investigation\_share)
$$

The centered log-ratio transform is reserved for true simplex vectors, not arbitrary two-feature products.

## 5.6 Capacity-Outcome Bridge

Capacity-outcome bridges must use explicit capacity-vector indices.

$$
M_{CapacityOutcome,G,k}
=
(
\rho_G^{SIM},
C_k/\mu^{POP},
\log(\rho_G^{SIM}+\epsilon),
\log(C_k/\mu^{POP}+\epsilon)
)
$$

where:

$$
k\in\mathcal{K}_{capacity}
$$

Derived contrast:

$$
v_{capacity\_contrast,G,k}
=
\log(\rho_G^{SIM}+\epsilon)
-
\beta_k\log(C_k/\mu^{POP}+\epsilon)
$$

$\beta_k$ is estimated in PIRS, not hardcoded, unless exploratory mode explicitly allows it.

Generic bed capacity is illegal:

$$
M_{CapacityOutcome,G,Beds}
\Rightarrow
\Delta_{carrier}=0
$$

unless `Beds` is a declared composite of capacity-vector indices.

## 5.7 Birth Context Bridge

$$
M_{BirthContext,K,X}
=
(
\pi_K^{SINASC},
X^{SIDRA/CNES}
)
$$

This supports birth outcomes versus sanitation, income, schooling, primary-care capacity, and obstetric capacity.

When $X$ is CNES-derived, it must be vector-indexed:

$$
X=C_k
\quad
k\in\mathcal{K}_{capacity}
$$

## 5.8 Context Gradient Bridge

$$
M_{ContextGradient,Y,X}
=
(
Y^{DATASUS},
X^{SIDRA},
\nabla_SX,
\nabla_TX
)
$$

Individual-level DATASUS socioeconomic marks may be contrasted against aggregate SIDRA context only through this bridge or a registered extension. This does not transform event-level marks into denominators.

## 5.9 Facility Flow Bridge

$$
M_{FacilityFlow,G,k}
=
(
Admissions_G^{SIH}(residence),
Admissions_G^{SIH}(movement),
Capacity_k^{CNES}(movement)
)
$$

with:

$$
k\in\mathcal{K}_{capacity}
$$

Facility-level linkage is permitted only when:

$$
SanitizedCNPJ=1
$$

or a validated direct facility crosswalk exists.

If facility linkage is unavailable, the bridge falls back to movement-municipality support:

$$
(s_{movement},t)
$$

with:

$$
warning=facility\_linkage\_fragile
$$

Any bridge using all-zero CNPJ records is illegal:

$$
Bridge_{FacilityFlow}
\land
id=\text{"00000000000000"}
\Rightarrow
\Delta_{carrier}=0
$$

## 5.10 Observer Capacity Bridge

$$
M_{ObserverCapacity}
=
(
ObserverFields,
CNESCapacityVector,
SIDRAContext,
FacilityLinkageQuality,
InvalidFlagShare
)
$$

This bridge detects whether observer-process quality varies with health-system capacity, administrative flag quality, facility linkage fragility, and socioeconomic context.

## 5.11 Civil–Health Divergence Bridge

Civil–health divergence fields compare DATASUS microdata streams against SIDRA Civil Registry aggregates as observer-process measurements, not replacements.

Death divergence and birth divergence are defined in Section 2.11.

# 6. Parametric Inference & Residual Scanner

## 6.1 PIRS Definition

The Parametric Inference & Residual Scanner receives legal EFG fields, selects admissible outcomes and covariates according to $Q(v)$ and $U_{\mathcal{I}}(v)$, fits parametric or semiparametric models, extracts residual fields, and scans residual non-linear dependence.

PIRS operates after EFG. It does not create raw epidemiological fields except model-derived residual fields.

Zero-variance fields, illegal economic composites, unbounded high-dimensional fields, invalid boolean outlier fields, and unsanitized facility-linkage fields cannot enter PIRS.

## 6.2 Model Selection

If an intensive density is modeled, PIRS decomposes it into numerator count and exposure offset whenever the carrier registry permits.

For count outcomes:

- zero-inflated support routes to hurdle or zero-inflated models.
- deep budget allows GAM or negative binomial when support is adequate.
- otherwise negative binomial or quasi-Poisson is used.

For bounded binomial proportions:

- binomial or beta-binomial models are used.

For positive skewed continuous outcomes:

- Gamma, lognormal, or two-part models are used depending on zero mass.

For SIH economic outcomes:

- `VAL_SH`, `VAL_SP`, `VAL_UTI`, and `VAL_TOT` are modeled as distinct economic estimands unless a registered composite cost operator is used.
- zero-inflated or constant cost fields are excluded or modeled with two-part methods only when nonzero support exists.

For simplex outcomes:

- Dirichlet or multinomial-logit models are used.

Otherwise:

- Gaussian or Student-t spatial panel models are used.

## 6.3 Spatial Effect Mode

Allowed spatial effect modes are:

$$
SpatialEffectMode
\in
\{
none,
UF\_FE,
municipality\_FE,
ICAR
\}
$$

Precedence:

$$
B=fast\Rightarrow UF\_FE
$$

Otherwise:

$$
|T|\ge10\land \zeta_Y\le0.10
\Rightarrow municipality\_FE
$$

and:

$$
|T|<10\lor \zeta_Y>0.10
\Rightarrow ICAR
$$

If:

$$
MoranI(Y)\approx0
$$

then:

$$
SpatialEffectMode=none
$$

## 6.4 Generic Count Model

A generic count model is:

$$
Y_i\sim Family(\mu_i,\theta)
$$

with linear predictor:

$$
g(\mu_i)
=
\log E_i
+
\alpha
+
FE/Smooth/ICAR
+
\sum_{j=1}^{p}\beta_jX_{ij}
$$

The exposure $E_i$ comes from the carrier registry. It is not always population.

## 6.5 Residual Registry

Generic deviance residual:

$$
e_i
=
sign(y_i-\hat\mu_i)
\sqrt{
2[
\ell(y_i;y_i)-\ell(\hat\mu_i;y_i)
]
}
$$

Residual types by family:

| Family | Residual |
|---|---|
| Poisson/NB/quasi-Poisson | deviance or Pearson |
| Hurdle/ZINB | randomized quantile / Dunn–Smyth |
| Binomial/Beta-binomial | randomized quantile |
| Gamma/lognormal | deviance or normalized log residual |
| Gaussian/Student-t | standardized residual |
| Dirichlet/simplex | clr/ilr residual vector |
| Multinomial | deviance residual vector |

Residual fields inherit:

$$
Prov(e_Y)=model\_derived
$$

## 6.6 Residual Uncertainty Modes

Residual modes are:

$$
ResidualMode
\in
\{
in\text{-}sample,
cross\text{-}fitted,
parametric\_bootstrap,
posterior\_predictive
\}
$$

Default by budget:

$$
ResidualMode(B)
=
\begin{cases}
in\text{-}sample & B=fast \\
cross\text{-}fitted & B=standard \\
cross\text{-}fitted+parametric\_bootstrap & B=deep
\end{cases}
$$

For standard and deep runs, in-sample residual HSIC is not the default.

### 6.6.1 Cross-Fitted Residuals

Partition support into folds:

$$
\mathcal{F}_1,\ldots,\mathcal{F}_K
$$

preserving spatial and temporal blocks.

For observation $i\in\mathcal{F}_k$:

$$
e_i^{cf}
=
Res
\left(
Y_i,
\widehat{\mathcal{M}}_{-k}(X_i)
\right)
$$

HSIC consumes:

$$
e_Y^{cf}
$$

rather than in-sample residuals.

### 6.6.2 Bootstrap-Adjusted HSIC

For deep budget, draw:

$$
Y^{(b)}\sim\widehat{\mathcal{M}}
$$

Compute:

$$
D_b=HSIC(X,e_Y^{(b)})
$$

The uncertainty-adjusted nonlinear score is:

$$
D^*
=
\frac{
\mathbb{E}_b[D_b]
}{
SD_b(D_b)+\epsilon
}
$$

The hypothesis table records:

$$
ResidualMode,
\quad
FoldScheme,
\quad
BootstrapCount,
\quad
ResidualUncertainty
$$

### 6.6.3 Kernel Variance Penalty Fallback

If cross-fitting is infeasible, PIRS may use an uncertainty-weighted residual kernel:

$$
K_e^{adj}
=
K_e\odot W
$$

where:

$$
W_{ij}
=
\frac{1}
{
\sqrt{
(1+\widehat{Var}(e_i))
(1+\widehat{Var}(e_j))
}
}
$$

This mode is labeled:

$$
HSICMode=uncertainty\_penalized\_exploratory
$$

## 6.7 HSIC Residual Scanner

HSIC is disabled when support is insufficient:

$$
n_{eff}<100
\Rightarrow
HSIC_{mode}=disabled
$$

Otherwise:

$$
HSIC_{mode}
=
\begin{cases}
exact & N\le5000\land B=deep \\
Nyström & N>5000\land B\in\{standard,deep\} \\
RFF & N>5000\land B=fast \\
exact & N\le5000\land B\in\{fast,standard\}\land memory\ permits \\
disabled & user\_disabled
\end{cases}
$$

The scanner operates on residuals:

$$
D_Z(X,e_Y)
$$

not raw outcomes, unless explicitly configured.

## 6.8 Null and FDR Registry

The scan regime maps support to null strategy, permutation unit, false-discovery method, and residual mode.

| Support | Null strategy | Permutations | Unit | FDR | Residual mode |
|---|---|---:|---|---|---|
| Annual municipal panel | spatial block + cyclic time shift | 1000 | cross-fitted residual | BY | cross-fitted |
| Monthly seasonal panel | season-preserving moving-block circular shift | 2000 | cross-fitted residual | BY | cross-fitted |
| Cross-sectional census | geo-adjacency shuffle | 1000 | raw or residual | BH | model-dependent |
| Facility stock | restricted intra-UF spatial swap | 5000 | raw field | Storey q | not required |
| Sparse stratified | bootstrap within strata | 1000 | residual | BY | cross-fitted if feasible |

If fewer than five spatial or temporal blocks exist:

$$
HSIC_{residual}
\to
descriptive\_association\_only
$$

## 6.9 Monthly Seasonal Null Correction

For monthly seasonal panels, arbitrary within-season shuffling is forbidden.

The null is:

$$
NullStrategy
=
season\_preserving\_moving\_block\_circular\_shift
$$

It preserves:

$$
seasonality,
\quad
lag\text{-}1\ autocorrelation,
\quad
spatial\ block\ dependence
$$

Let block length be:

$$
L=\max(12,\widehat{\ell}_{AR})
$$

Residual sequences are shifted by contiguous blocks while respecting seasonal boundaries.

## 6.10 Budget Matrix

| Budget | $k_{max}$ | TopK predictors | ICD depth | HSIC | Covariates | Latent reconstruction |
|---|---:|---:|---|---|---:|---|
| fast | 2 | 10 | chapter/curated | RFF or disabled | 10 | none |
| standard | 4 | 30 | block | Nyström | 30 | bounded sociological |
| deep | 7 | 100 | leaf if support adequate | exact/Nyström | 100 | full coregionalized contextual reconstruction |

# 7. Materialization Contract

Materialization is:

$$
Materialize(v)
=
\begin{cases}
eager & v\in V_{core}\cup\mathcal{V}_0 \\
cached & v\in Ancestors(V_{active\_models}) \\
lazy & v\in V_{exploratory}\land U_{\mathcal{I}}<\theta_{compile} \\
view\text{-}only & otherwise
\end{cases}
$$

Illegal fields are not materialized except in audit logs.

Each node has a materialization state:

$$
materialization(v)
\in
\{
unmaterialized,
metadata\_only,
planned,
materialized,
cached,
blocked,
failed,
quarantined
\}
$$

Empirical equivalence compression cannot run on metadata-only or planned nodes. It runs only after candidate selection and materialization.

SHE may preserve raw long-form source fact stores outside $V_{fields}$. The EFG may not materialize unbounded high-dimensional SIDRA, all-missing SIH, constant SIH, generic CNES bed, generic cost, or unsanitized CNPJ fields.

# 8. Output Contract

## 8.1 Immutable 17-Key Run Bundle

Every run emits exactly 17 first-class keys.

The output bundle is:

```json
{
  "$schema": "https://pegasus.famed.ufal.br/schemas/run_contract_v1.json",
  "OutputBundle": {
    "V_fields": "Arrow/Parquet field tensor database split by Class status",
    "E_DAG": "Adjacency/edge table with lineage, operators, and derivations",
    "Q_tensor": "Diagnostic state tensor for every field node",
    "P_vector": "Provenance vectors and propagated provenance labels",
    "UserIntent": "Frozen input tuple I",
    "Warnings": "Field-level and run-level warnings with severity and inheritance",
    "ModelAssociations": "Adjusted parametric association table",
    "ResidualAssociations": "Residual diagnostics and residual explanatory summaries",
    "Hypotheses": "HSIC/residual nonlinear hypothesis table",
    "Tables": "Summaries, decompositions, rankings, and trends",
    "Maps": "Support-resolved spatial layers",
    "VariableDictionary": "Field definitions, formulas, units, roles, and provenance",
    "FailedBranches": "Rejected EFG expressions and failure causes",
    "QuarantinedFields": "Fields blocked or limited by Q(v)",
    "ForcedFields": "User-forced unstable fields and inherited warnings",
    "RunConfig": "Budget, geospatial mode, registry versions, and thresholds",
    "ReproducibilityManifest": "SHA-256 hashes, source versions, and random seeds"
  }
}
```

No production run is valid unless all 17 keys exist.

Run profiles add a populated-content assertion without changing the key set:
profile-required keys must be non-empty, and any profile-optional empty key must carry an explicit `empty_by_profile` warning row naming the key and run profile.

## 8.2 Field Dictionary Requirements

Every field in $V_{fields}$ must have:

- field identifier
- display name
- technical name
- definition
- estimand label
- source systems
- carrier
- unit
- support description
- axis description
- provenance description
- state
- dashboard-safety status
- interpretation warning

Race-related fields must use only allowed race estimand labels.

Facility-capacity fields must declare:

$$
capacity\_index=k
$$

Cost fields must declare:

$$
cost\_component
\in
\{VAL\_SH,VAL\_SP,VAL\_UTI,VAL\_TOT,declared\_composite\}
$$

Diagnostic fields must declare:

$$
diagnostic\_role,
\quad
topology,
\quad
position
$$

SIDRA fields must declare:

$$
projection\_matrix\_id,
\quad
stitch\_metadata
$$

when applicable.

## 8.3 Hypothesis Row Requirements

Every nonlinear hypothesis row must include:

- Outcome_Field_ID
- Covariate_Field_ID
- Residual_Field_ID
- HSIC_Mode
- ResidualMode
- FoldScheme
- BootstrapCount
- ResidualUncertainty
- NullStrategy
- FDR_Method
- Statistic
- P_Value
- Adjusted_Q_Value
- Warnings

## 8.4 Population Tensor Output Requirements

Population tensor artifacts must record:

- PopulationTensorMode
- SolverBackend
- SparseJacobian
- DenominatorFeedbackWarning
- reconstruction uncertainty
- provenance
- warnings
- state tensor diagnostics

## 8.5 Geneallocated Field Output Requirements

Geneallocated fields must record:

- GeoProvenance = geneallocated
- AllocatedFrom = AMC
- IntensiveFieldRule = recomputed_from_allocated_numerator_denominator
- EcologicalBackprojectionWarning = true

## 8.6 Race Bridge Metadata Requirements

Every race-bridged field must add:

- numerator_axis_source
- denominator_axis_target
- bridge_operator
- emission_matrix_registry_version
- bridge_mode
- missing_race_share
- race_bridge_cv
- sensitivity_width
- dashboard_safe
- race-axis warnings
- Bayesian ecological bridge warning

## 8.7 Civil–Health Divergence Metadata Requirements

Civil–health divergence fields must add:

- comparison_type
- health_source
- civil_source
- year_basis_health
- year_basis_civil
- divergence_value
- system_measurement_divergence warning

## 8.8 Source-Reality Metadata Requirements

The output bundle must include warnings and metadata for:

- composite-decoded DATASUS fields;
- invalid boolean flag shares;
- nullified corporate identifiers;
- zero-variance dropped columns;
- SIH economic component separation;
- CNES capacity-vector indexing;
- diagnostic-topology distinctions;
- SIDRA longitudinal stitching;
- SIDRA classification projection;
- high-dimensional bounded pushforward.

These metadata are part of the scientific audit trail, not optional diagnostics.

# 9. End-to-End Compilation Algorithm

## 9.1 Input

The compiler receives:

$$
\mathcal{I}
=
(G,T,\mathcal{H}_0,\mathcal{V}_0,\mathbf{w}_{systems},\mathcal{C}_{policy},B,\mathcal{M}_{geo},\mathcal{R}_{force},\mathcal{D}_{exclude})
$$

## 9.2 Phase A — SHE

1. Parse $\mathcal{M}_{geo}$ and instantiate $S^*(\mathcal{M}_{geo})$.
2. Ingest raw source fields through the source registry.
3. Preserve raw and processed provenance where available.
4. Apply composite decoders to structural age, physical scalar, count, categorical, boolean, and identifier fields.
5. Apply $Clamp_{bool}$ to boolean-like administrative flags.
6. Apply $Filter_{CNPJ}$ to CNES/SIH corporate identifiers.
7. Apply zero-variance exclusion to all-missing and all-constant columns.
8. Route diagnostic strings through the topological six-state ICD parser.
9. Build event streams, facility stocks, capacity vectors, economic burden components, and context cubes.
10. Apply geospatial harmonization.
11. Apply CNES capacity-vector aggregation views.
12. Apply monetary deflation separately by SIH cost component.
13. Apply age, time, race-axis, diagnostic-topology, facility, and context alignment guards.
14. Build independent population denominators by default.
15. Build SIM-informed denominators only when explicitly enabled.
16. For `contextual` and `full` profiles, route SIDRA fields through $R(q)$; for `core_vital`, SIDRA context keys may be empty only with `empty_by_profile` warnings.
17. For `contextual` and `full` profiles, execute longitudinal stitching when a concept is split across tables.
18. For `contextual` and `full` profiles, execute $\Pi_{Clsf\to Axis}$ for SIDRA classifications.
19. For `contextual` and `full` profiles, execute mandatory high-dimensional bounded pushforward before EFG exposure.
20. For `contextual` and `full` profiles, run ST-DFM only when gated and certifiable.
21. Emit $\mathcal{B}_{official}$, $\mathcal{B}_{harmonized}$, $\mathcal{B}_{deflated}$, $\mathcal{B}_{reconstructed}$, $\mathcal{B}_{latent}$, $\mathcal{B}_{cross-sectional}$, and $\mathcal{B}_{excluded}$.

## 9.3 Phase B — EFG

1. Seed fields with $V_{seed}=V_{core}\cup\mathcal{V}_0\cup\mathcal{H}_0$.
2. For each candidate operator:
   - align axes;
   - evaluate $\Delta_{struct}$;
   - evaluate $\Delta_{declaration}$;
   - enforce carrier specificity for capacity vectors and cost components;
   - enforce diagnostic-topology compatibility;
   - apply provenance policy;
   - evaluate $Q(v)$;
   - apply quarantine and force permissions;
   - run topology-specific ICD traversal gates;
   - route quality codes to observer processes;
   - instantiate bridge grammars;
   - perform topological compression.
3. Materialize selected candidates.
4. Perform empirical post-selection compression.
5. Emit fields, lineage, warnings, failed branches, and variable dictionary entries.

## 9.4 Phase C — PIRS Parametric Modeling

1. Score fields using $U_{\mathcal{I}}(v)$.
2. Exclude zero-variance fields, illegal composites, invalid flag outlier fields, and unsanitized linkage fields.
3. Select admissible outcomes and covariates under $Q(v)$ and budget $B$.
4. Select model family and spatial effect mode.
5. Fit models using exposures from the carrier registry.
6. Emit ModelAssociations.
7. Extract residual fields.
8. Emit ResidualAssociations.

## 9.5 Phase D — Residual Nonlinear Scan

1. Disable HSIC when $n_{eff}<100$.
2. Select exact, Nyström, or RFF HSIC mode.
3. Use cross-fitted residuals for standard and deep budgets.
4. Restrict to mutually observed supports.
5. Apply support-appropriate null regime.
6. Apply FDR correction.
7. Emit Hypotheses with uncertainty and warnings.

## 9.6 Phase E — Serialization

1. Write all field, graph, state, provenance, model, residual, hypothesis, table, and map artifacts.
2. Validate the 17-key output schema.
3. Hash source artifacts, registry versions, configurations, and random seeds.
4. Emit $\mathcal{O}_{run}$.

# 10. Hard Abort Conditions

The compiler aborts when any of the following occur:

- invalid user intent;
- missing required registry;
- unknown source system;
- source ingestion failure without cached valid artifact;
- missing required source field for a mandatory core variable;
- required composite decoder missing for a structural field;
- `contextual` or `full` SIDRA request exceeding 49,900 cells after planning;
- `contextual` or `full` unsupported SIDRA table, variable, period, category, or locality;
- dense national population optimization requested above scale threshold;
- population solver band or explicit solver request resolves only to a non-executable scaffold;
- direct administrative race numerator divided by IBGE self-declared denominator;
- race bridge requested without valid emission-prior object;
- direct allocation of an intensive geneallocated field;
- `contextual` or `full` ST-DFM proportion field promoted to verified without denominator or survey uncertainty;
- standard/deep residual HSIC using in-sample residuals;
- monthly seasonal null using arbitrary within-season shuffle;
- SIDRA Civil Registry used to replace SIM/SINASC event streams;
- unsanitized all-zero CNPJ used in facility-flow linkage;
- generic CNES `Beds` carrier requested without capacity-vector index;
- generic SIH cost field requested when the estimand requires `VAL_SH`, `VAL_SP`, or `VAL_UTI`;
- `contextual` or `full` high-dimensional SIDRA field exposed to EFG without legal bounded pushforward;
- diagnostic topology erased without explicit projection;
- output bundle fails the 17-key schema contract.

# 11. Downgrade and Quarantine Conditions

The compiler downgrades state instead of aborting when an object is legal but unstable.

Downgrade triggers include:

- sparse support;
- high missingness;
- fragile denominator;
- SIM-informed denominator feedback risk;
- uncertified latent reconstruction;
- high ST-DFM factor instability;
- system measurement divergence;
- race bridge high coefficient of variation;
- wide race sensitivity interval;
- missing local calibration for race bridge;
- facility linkage uncertainty;
- nullified corporate identifiers;
- invalid CNES boolean flag shares;
- geneallocated support used for inference;
- SIDRA stitching without overlap but with declared methodological continuity;
- fractional classification projection;
- insufficient HSIC blocks.

Downgraded fields remain auditable. They do not silently disappear.

# 12. Final Scope Statement

PegaSUS is:

$$
\boxed{
\textbf{an intent-conditioned, autonomous, stratified epidemiological field compiler}
}
$$

with:

$$
\boxed{
\textbf{provenance-aware substrate harmonization, typed field-graph generation, measurement-process-aware race-axis epistemology, source-realistic DATASUS decoding, SIDRA longitudinal stitching, classification projection, cross-system bridge grammars, and residual non-linear hypothesis scanning.}
}
$$

The system generates epidemiological fields and causal-hypothesis candidates. It does not certify causality without a separate identification layer.

The architecture locks:

- canonical event-substrate boundaries;
- composite administrative decoding semantics;
- zero-variance exclusion semantics;
- denominator independence semantics;
- race/color measurement-process legality;
- SIH diagnostic-topology distinctions;
- SIH economic component separation;
- CNES capacity-vector indexing;
- CNES/SIH corporate linkage gates;
- SIDRA longitudinal stitching;
- SIDRA classification projection;
- high-dimensional bounded pushforward;
- ST-DFM gating and certification;
- EFG node metadata and lineage;
- seven-part structural legality plus declaration-process compatibility;
- $Q(v)$ state semantics;
- bridge grammar outputs;
- residual-scanner null regimes;
- and the immutable 17-key output bundle.

Numerical solvers remain mutable only inside their declared contracts. The compiler must never replace mathematical legality with convenience, missingness with silence, source structure with scalar fiction, or measurement-process uncertainty with false precision.


