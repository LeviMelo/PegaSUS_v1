# DATASUS Technical Variable Compendium

This compendium details the structural, semantic, and empirical characteristics of the DATASUS datasets profiled for this project. It is compiled to serve as an exact, exhaustive schema map for developers and data scientists working on the data substrate. 

The scope is restricted strictly to **DATASUS** datasets (excluding SIDRA). Profile metrics are derived from a scan of up to 200,000 records per file, representing empirical data from Alagoas (AL) with a baseline competency around 2022.

---

## 1. Global Dataset Metadata

| System | Stage | Grain / Record Unit | Files | Rows Profiled | Columns | Profiled File Path / Pattern |
| :--- | :--- | :--- | :---: | :---: | :---: | :--- |
| **CNES-ST** | processed | Registro de estabelecimento/atributo cadastral | 1 | 3,862 | 217 | `...\data\raw\datasus\CNES-ST\uf=AL\2022_01__2022_01\...\processed.csv` |
| **CNES-ST** | raw | Registro de estabelecimento/atributo cadastral | 1 | 3,862 | 209 | `...\data\raw\datasus\CNES-ST\uf=AL\2022_01__2022_01\...\raw.csv` |
| **SIH-RD** | processed | Internação/autorização hospitalar reduzida (AIH) | 1 | 12,854 | 122 | `...\data\raw\datasus\SIH-RD\uf=AL\2022_01__2022_01\...\processed.csv` |
| **SIH-RD** | raw | Internação/autorização hospitalar reduzida (AIH) | 1 | 12,854 | 114 | `...\data\raw\datasus\SIH-RD\uf=AL\2022_01__2022_01\...\raw.csv` |
| **SIM-DO** | processed | Óbito / declaração de óbito | 1 | 23,122 | 101 | `...\data\raw\datasus\SIM-DO\uf=AL\2022_01__2022_12\...\processed.csv` |
| **SIM-DO** | raw | Óbito / declaração de óbito | 1 | 23,122 | 88 | `...\data\raw\datasus\SIM-DO\uf=AL\2022_01__2022_12\...\raw.csv` |
| **SINASC** | processed | Nascido vivo / declaração de nascido vivo | 1 | 45,742 | 70 | `...\data\raw\datasus\SINASC\uf=AL\2022_01__2022_12\...\processed.csv` |
| **SINASC** | raw | Nascido vivo / declaração de nascido vivo | 1 | 45,742 | 62 | `...\data\raw\datasus\SINASC\uf=AL\2022_01__2022_12\...\raw.csv` |

---

## 2. CNES-ST (Cadastro Nacional de Estabelecimentos de Saúde)

This dataset captures registry metrics, structural capacities, administrative ties, and operational constraints of health facilities.

### 2.1 Core Registry & Geographical Variables (Processed vs. Raw)

| Variable Name | Stage | Observed Kind | Type | Null % | Unique | Empirical Range / Values | Code-Shape & Formatting |
| :--- | :--- | :--- | :--- | :---: | :---: | :--- | :--- |
| **CNES** | Both | municipality_code | str | 0.00% | 3,862 | `12343` $\rightarrow$ `9.99953e+06` | 7-digit string (100% unique ID) |
| **CODUFMUN** | Both | municipality_code | str | 0.00% | 102 | `270010` $\rightarrow$ `270940` | 6-digit IBGE municipality code |
| **COD_CEP** | Both | numeric | str | 0.00% | 738 | `5.701e+07` $\rightarrow$ `1.0e+08` | 8-digit postal code (e.g., `57080000`) |
| **CPF_CNPJ** | Both | numeric | str | 0.00% | 2,115 | `0` $\rightarrow$ `7.3472e+13` | 14-digit string; modal: `00000000000000` (1741) |
| **PF_PJ** | Processed | low_cardinality_cat | str | 0.00% | 2 | `"Pessoa jurídica"` (2957), `"Pessoa física"` (905) | Decoded string |
| | Raw | numeric | str | 0.00% | 2 | `1` $\rightarrow$ `3` | `"3"` (2957), `"1"` (905); digit_code_like:1 |
| **NIV_DEP** | Processed | low_cardinality_cat | str | 0.00% | 2 | `"Individual"` (2101), `"Mantida"` (1761) | Decoded string |
| | Raw | numeric | str | 0.00% | 2 | `1` $\rightarrow$ `3` | `"1"` (2101), `"3"` (1761); digit_code_like:1 |
| **CNPJ_MAN** | Both | numeric | str | 0.00% | 111 | `0` $\rightarrow$ `6.99778e+13` | 14-digit string; modal: `0` or `00000000000000` (2101) |
| **REGSAUDE** | Processed | medium_cardinality | str | 78.33% | 48 | `1` $\rightarrow$ `12` | Contains encodings like `"7"` (170), `"6"` (67), `"8"` (45) |
| | Raw | medium_cardinality | str | 78.33% | 67 | `1` $\rightarrow$ `12` | Contains encodings like `"6,,,,1"` (39), `"8,,,,1"` (38) |
| **MICR_REG** | Processed | medium_cardinality | str | 90.78% | 45 | `1` $\rightarrow$ `89` | Modal: `"5"` (54), `"08"` (33), `"10"` (22) |
| | Raw | medium_cardinality | str | 87.93% | 38 | `1` $\rightarrow$ `89` | Modal: `"M"` (175), `"5"` (54), `"10"` (22) |
| **DISTRSAN** | Processed | low_cardinality_cat | str | 100.0% | 0 | None | Fully missing in processed |
| | Raw | low_cardinality_cat | str | 99.82% | 2 | `"M"` (6), `"PAB"` (1) | Highly sparse administrative text |
| **DISTRADM** | Processed | low_cardinality_cat | str | 99.87% | 4 | `0` $\rightarrow$ `1` | `"PAB"` (2), `"0000"` (1), `"000"` (1), `"1"` (1) |
| | Raw | numeric | str | 98.21% | 5 | `0` $\rightarrow$ `1` | `"1"` (62), `"0"` (4), `"PAB"` (1) |
| **VINC_SUS** | Processed | low_cardinality_cat | str | 0.00% | 2 | `"Sim"` (2080), `"Não"` (1782) | Decoded boolean string |
| | Raw | numeric | str | 0.16% | 4 | `0` $\rightarrow$ `4` | `"1"` (1872), `"0"` (1744), `"04"` (175), `"M"` (65) |
| **TPGESTAO** | Processed | low_cardinality_cat | str | 0.00% | 2 | `"Municipal"` (3721), `"Estadual"` (141) | Decoded string |
| | Raw | low_cardinality_cat | str | 6.21% | 3 | `4` $\rightarrow$ `4` | `"M"` (3475), `"E"` (141), `"04"` (6) |
| **ATIVIDAD** | Processed | numeric | str | 0.00% | 4 | `1` $\rightarrow$ `4` | `"04"` (3761), `"03"` (87), `"01"` (7), `"02"` (7); digit:2 |
| | Raw | numeric | str | 1.68% | 9 | `1` $\rightarrow$ `73` | `"04"` (3519), `"03"` (232), `"01"` (18); digit:2 |
| **CLIENTEL** | Processed | numeric | str | 2.95% | 3 | `1` $\rightarrow$ `3` | `"03"` (2343), `"01"` (1113), `"02"` (292); digit:2 |
| | Raw | numeric | str | 2.98% | 14 | `1` $\rightarrow$ `99` | `"03"` (2224), `"01"` (1011), `"02"` (305); digit:2 |
| **TP_UNID** | Processed | medium_cardinality | str | 0.00% | 34 | Categorical names | `"Consultório isolado"` (1210), `"Centro de saúde"` (870) |
| | Raw | numeric | str | 4.01% | 41 | `1` $\rightarrow$ `126` | `"22"` (1179), `"02"` (771), `"36"` (540); digit_code:2,3 |
| **TURNO_AT** | Processed | numeric | str | 0.57% | 7 | `1` $\rightarrow$ `7` | `"03"` (3082), `"04"` (267), `"06"` (229); digit:2 |
| | Raw | numeric | str | 6.40% | 17 | `0` $\rightarrow$ `33,324` | `"03"` (2876), `"04"` (259), `"06"` (216); digit:2 to 5 |
| **TP_PREST** | Processed | numeric | str | 0.00% | 1 | `99` $\rightarrow$ `99` | `"99"` (3862); digit:2 |
| | Raw | municipality_code | str | 6.24% | 3 | `1` $\rightarrow$ `329,320` | `"99"` (3616), `"001"` (4), `"329320"` (1) |
| **ALVARA** | Both | high_cardinality | str | ~50% | ~1750 | `1` $\rightarrow$ `2.0212e+13` | Sanitary permit registry text/numbers |
| **DT_EXPED** | Processed | date | str | 48.45% | 1,385 | `1987-07-06` $\rightarrow$ `2021-12-15` | ISO standard date string |
| | Raw | date | str | 49.79% | 1,345 | `1987-07-06` $\rightarrow$ `2021-12-15` | raw shapes like `"20170626"`; digit:1,8 |
| **ORGEXPED** | Processed | low_cardinality_cat | str | 46.87% | 2 | `"SMS"` (1966), `"SES"` (86) | Issuing authority |
| | Raw | numeric | str | 49.20% | 2 | `1` $\rightarrow$ `2` | `"2"` (1888), `"1"` (74); digit_code_like:1 |
| **NAT_JUR** | Processed | medium_cardinality | str | 0.00% | 22 | `3301` $\rightarrow$ `4000` | `"Município"` (1446), `"4000"` (903), limitadas, etc. |
| | Raw | numeric | str | 6.45% | 22 | `1015` $\rightarrow$ `4000` | `"1244"` (1270), `"4000"` (872), `"2062"` (648); digit:4 |

### 2.2 Facility Programmatic, Infrastructure & Commission Flags (Exhaustive Block)

This block groups specialized binary indicators, program flags, and commissions in `CNES-ST`. These indicators are structurally numeric flags in **Raw** and boolean-like text flags in **Processed**.

#### 2.2.1 Programmatic Flags (`GESPRG` series)
* **GESPRG1M** to **GESPRG6M** (Municipal Programs) and **GESPRG1E** to **GESPRG6E** (State Programs):
  * **Processed**: Boolean categories (`"Não"` / `"Sim"`).
    * `GESPRG1M` (Atenção Básica): `"Não"` (2214), `"Sim"` (1648).
    * `GESPRG2M` (Urgência): `"Sim"` (2807), `"Não"` (1055).
    * `GESPRG2E` (Urgência): `"Não"` (3725), `"Sim"` (137).
    * `GESPRG3E` (Atenção Psicossocial): `"Não"` (3862).
    * `GESPRG3M` (Atenção Psicossocial): `"Não"` (3858), `"Sim"` (4).
    * `GESPRG4M` (Atenção Especializada): `"Não"` (3759), `"Sim"` (103).
    * `GESPRG5M` (Vigilância em Saúde): `"Não"` (3747), `"Sim"` (115).
    * `GESPRG6M` (Apoio Diagnóstico): `"Não"` (3848), `"Sim"` (14).
  * **Raw**: Numeric flags (`0` or `1`).
    * `GESPRG1M` is `"0"` (2385), `"1"` (1477).
    * Anomalous Raw Values: `GESPRG5M` has outliers up to `32` (`15` (1), `32` (1)); `GESPRG6E` has outliers up to `131` (`131` (1), `47` (1)); `GESPRG6M` has outliers up to `97` (`97` (1), `23` (1)).

#### 2.2.2 Service & Care Complexity Levels
* **NIVATE_A** (Atenção Básica Level): Processed: `"Sim"` (3788), `"Não"` (74). Raw: `"1"` (3545), `"0"` (317).
* **NIVATE_H** (Hospital Level): Processed: `"Não"` (3721), `"Sim"` (141). Raw: `"0"` (3722), `"1"` (138) with outliers up to `63`.
* **ATENDAMB** (Ambulatory Care): Processed: `"Sim"` (3338), `"Não"` (524). Raw: `"1"` (3128), `"0"` (733).
* **ATENDHOS** (Hospital Care): Processed: `"Não"` (3759), `"Sim"` (103). Raw: `"0"` (3711), `"1"` (151).
* **URGEMERG** (Urgency/Emergency): Processed: `"Não"` (3579), `"Sim"` (283). Raw: `"0"` (3483), `"1"` (355) with outliers up to `5`.

#### 2.2.3 Physical Structure & Rooms (`QTINST` series)
All `QTINST` variables behave as non-negative integer string values (range $0 \rightarrow N$):
* `QTINST01` to `QTINST13` (Consultório rooms, basic diagnostics): Modal is `0` (typically ~3,800 records). Max ranges: `QTINST01` (0 to 3), `QTINST04` (0 to 4, 66 records = 1), `QTINST08` (0 to 4), `QTINST12` (0 to 2/3), `QTINST13` (0 to 21 in raw).
* `QTINST14` (Salas de cirurgia): Range 0 to 11. Modal: `0` (3713 processed).
* `QTINST15` (Salas de ginecologia): Range 0 to 15. Modal: `0` (2574 processed).
* `QTINST16` (Salas de odontologia): Range 0 to 56. Modal: `0` (3035 processed).
* `QTINST17` (Salas de curativos): Range 0 to 27. Modal: `0` (3299 processed).
* `QTINST18` (Salas de vacina): Range 0 to 47. Modal: `0` (2563 processed).
* `QTINST22` (Sutura/Inalação): Range 0 to 13.
* `QTINST23` (Salas de triagem): Range 0 to 45. Modal `1` occurs in over 1,200 records.
* `QTINST25` (Salas de observação): Range 0 to 10.
* `QTINST26` (Salas de esterilização): Range 0 to 2.
* `QTINST31` (Salas de gesso): Range 0 to 13.
* `QTINST34` (Salas de raios-X): Range 0 to 2.
* `QTINST35` (Salas de ultrassom): Range 0 to 3.
* `CENTRCIR` (Centro Cirúrgico flag): Range 0 to 1. Modal: `0` (3775).
* `CENTROBS` (Centro Obstétrico flag): Processed: `"Não"` (3799), `"Sim"` (63). Raw: range 0 to 11, modal `0` (3801).

#### 2.2.4 Specialized Beds (`QTLEIT` series)
Categorized beds (e.g., Pediatria, Obstetrícia, Isolamento) with structural code representations:
* `QTLEIT05` (Beds type 05): Range 0 to 7. Modal: `0` (3801).
* `QTLEIT08` (Beds type 08): Range 0 to 24. Modal: `0` (3781).
* `QTLEIT22` (Beds type 22): Range 0 to 22. Modal: `0` (3625).
* `QTLEIT40` (Beds type 40): Range 0 to 33. Modal: `0` (3826).
* `QTLEITP1` (Clinical beds): Range 0 to 223. Modal: `0` (3793).
* `QTLEITP2` (Surgical beds): Range 0 to 160. Modal: `0` (3783).
* `QTLEITP3` (Obstetric beds): Range 0 to 134. Modal: `0` (3813).
* `LEITHOSP` (Total hospital beds indicator): Range 0 to 1. Modal: `0` (3763).

#### 2.2.5 Support Services (`SERAP` series)
Processed features `"Sim"`/`"Não"` categorizations; Raw features numeric `1` (Yes) / `0` (No):
* `SERAP01P` to `SERAP11P` (Own services, "Próprio"): e.g., `SERAP01P` (Radiology): `"Sim"` (2691), `"Não"` (1171). `SERAP03P` (Laboratório): `"Não"` (3011), `"Sim"` (851).
* `SERAP01T` to `SERAP11T` (Third-party services, "Terceirizado"): e.g., `SERAP09T` (Hemoterapia): `"Não"` (3484), `"Sim"` (378).
* `SERAPOIO` (Support structure): Processed: `"Sim"` (2863), `"Não"` (999). Raw: `"1"` (2776), `"0"` (1086).

#### 2.2.6 Waste Management & Environmental Controls
* `RES_BIOL` (Biological waste): Processed: `"Não"` (2347), `"Sim"` (1515). Raw: `"0"` (2327), `"1"` (1535).
* `RES_QUIM` (Chemical waste): Processed: `"Não"` (3553), `"Sim"` (309). Raw: `"0"` (3564), `"1"` (298).
* `RES_RADI` (Radioactive waste): Processed: `"Não"` (3790), `"Sim"` (72). Raw: `"0"` (3752), `"1"` (110).
* `COLETRES` (Waste collection): Processed: `"Sim"` (2842), `"Não"` (1020). Raw: `"1"` (2704), `"0"` (1158).

#### 2.2.7 Internal Committees (`COMISS` series)
Presence of regulatory internal committees (e.g., Ethics, Infection Control, Medical Records):
* `COMISSAO` (Any committee flag): Processed: `"Não"` (3426), `"Sim"` (436). Raw: `"0"` (3472), `"1"` (390).
* `COMISS01` (Infection Control): Processed: `"Não"` (3820), `"Sim"` (42). Raw: `"0"` (3821), `"1"` (41).
* `COMISS10` (Death review): Processed: `"Não"` (3598), `"Sim"` (264). Raw: `"0"` (3591), `"1"` (271).
* `COMISS11` (Medical records review): Processed: `"Não"` (3487), `"Sim"` (375). Raw: `"0"` (3526), `"1"` (336).

#### 2.2.8 Specific Ambulatory Convênios (`AP` series)
Maps billing eligibility of facilities to various health agreements (SUS, health plans, out-of-pocket):
* `AP01CV01` to `AP07CV07` matrices:
  * Modal value is `"Não"` / `"0"` (representing over 3,700 records in most cases).
  * Notable exception: `AP02CV01` (SUS services): Processed: `"Não"` (2001), `"Sim"` (1861). Raw: `"0"` (2179), `"1"` (1683).
  * `AP02CV02` (SUS contracts): Processed: `"Não"` (2062), `"Sim"` (1800). Raw: `"0"` (2081), `"1"` (1781).
  * `AP02CV06` (Ambulatory SUS capacity): Processed: `"Não"` (2434), `"Sim"` (1428). Raw: `"0"` (2456), `"1"` (1406).
  * Highly sparse raw columns: `AP07CV05`, `AP07CV06`, `AP07CV07` preserve raw timestamp integers representing processed update months (e.g., `202006` or `202201`) alongside structural zeros.

### 2.3 Processed-Only Derived Geography Columns
These columns do not exist in the Raw layer and are compiled via lookup tables. All show **0.00% missingness** across the Alagoas dataset:
* **munResStatus**: `str` | 1 unique value: `"ATIVO"` (3862).
* **munResTipo**: `str` | 1 unique value: `"MUNIC"` (3862).
* **munResNome**: `str` | 102 unique values | e.g., `"Maceió"` (1628), `"Arapiraca"` (347), `"Penedo"` (96).
* **munResUf**: `str` | 1 unique value: `"Alagoas"` (3862).
* **munResLat**: `str` | 102 unique values | Range: `-1.04074e+06` $\rightarrow$ `-9234` (Latitude coordinates multiplied/encoded).
* **munResLon**: `str` | 102 unique values | Range: `-3.80098e+06` $\rightarrow$ `-35461` (Longitude coordinates multiplied/encoded).
* **munResAlt**: `str` | 88 unique values | Range: `2` $\rightarrow$ `686` (Altitude in meters).
* **munResArea**: `str` | 102 unique values | Range: `16927` $\rightarrow$ `917858` (Area value representation).

---

## 3. SIH-RD (Sistema de Informações Hospitalares — AIH Reduzida)

Captures billing records and clinical abstractions of hospitalizations under the SUS network.

### 3.1 SIH-RD Variables Catalog (Processed vs. Raw)

| Variable Name | Stage | Observed Kind | Type | Null % | Unique | Empirical Range / Values | Code-Shape & Formatting |
| :--- | :--- | :--- | :--- | :---: | :---: | :--- | :--- |
| **UF_ZI** | Both | municipality_code | str | 0.00% | 37 | `270000` $\rightarrow$ `270940` | 6-digit IBGE code; e.g., `"270430"` (4295) |
| **ANO_CMPT** | Both | numeric | str | 0.00% | 1 | `2022` $\rightarrow$ `2022` | 4-digit processing year |
| **MES_CMPT** | Processed | numeric | str | 0.00% | 1 | `1` $\rightarrow$ `1` | `"1"`; single-digit month representation |
| | Raw | numeric | str | 0.00% | 1 | `1` $\rightarrow$ `1` | `"01"`; 2-digit month string |
| **ESPEC** | Processed | numeric | str | 0.00% | 7 | `1` $\rightarrow$ `7` | `"02"` (3936), `"01"` (3558), `"Saúde Mental (Clínico)"` (54) |
| | Raw | numeric | str | 0.00% | 7 | `1` $\rightarrow$ `87` | `"02"` (3936), `"01"` (3558), `"87"` (54); digit:2 |
| **CGC_HOSP** | Both | numeric | str | 24.26% | 35 | `1.22256e+11` $\rightarrow$ `4.11613e+13` | 14-digit CNPJ; modal: `12200259000246` (1294) |
| **N_AIH** | Both | numeric | str | 0.00% | 12,775 | `2.7081e+12` $\rightarrow$ `2.72211e+12` | 13-digit transaction key; digit_code_like:13 |
| **IDENT** | Processed | low_cardinality_cat | str | 0.00% | 2 | `"Principal"` (12544), `"Longa permanência"` (310) | Decoded string |
| | Raw | numeric | str | 0.00% | 2 | `1` $\rightarrow$ `5` | `"1"` (12544), `"5"` (310); digit_code_like:1 |
| **CEP** | Both | numeric | str | 0.00% | 2,357 | `5.756e+06` $\rightarrow$ `8.822e+07` | 8-digit patient postal code |
| **MUNIC_RES** | Both | municipality_code | str | 0.00% | 133 | `110020` $\rightarrow$ `530010` | 6-digit IBGE code of patient residence |
| **NASC** | Processed | date | str | 0.00% | 9,095 | `1913-05-17` $\rightarrow$ `2022-01-30` | ISO standard date of birth |
| | Raw | date | str | 0.00% | 9,095 | `1913-05-17` $\rightarrow$ `2022-01-30` | `YYYYMMDD` string shape; digit:8 |
| **SEXO** | Processed | low_cardinality_cat | str | 0.00% | 2 | `"Feminino"` (8357), `"Masculino"` (4497) | Decoded string |
| | Raw | numeric | str | 0.00% | 2 | `1` $\rightarrow$ `3` | `"3"` (8357), `"1"` (4497); digit_code_like:1 |
| **UTI_MES_TO**| Both | numeric | str | 0.00% | 56 | `0` $\rightarrow$ `110` | Total ICU stay days this month; modal: `0` (12106) |
| **MARCA_UTI**| Processed | low_cardinality_cat | str | 0.00% | 13 | ICU descriptions | `"00"` (12106), `"UTI adulto - tipo II"` (329) |
| | Raw | numeric | str | 0.00% | 13 | `0` $\rightarrow$ `99` | `"00"` (12106), `"75"` (329), `"81"` (141); digit:2 |
| **UTI_INT_TO**| Both | numeric | str | 0.00% | 36 | `0` $\rightarrow$ `65` | Total ICU stay days in hospitalization |
| **DIAR_ACOM**| Both | numeric | str | 0.00% | 57 | `0` $\rightarrow$ `94` | Companion stay days; modal: `0` (5026) |
| **QT_DIARIAS**| Both | numeric | str | 0.00% | 72 | `0` $\rightarrow$ `155` | Total hospital days billed; modal: `1` (3522) |
| **PROC_SOLIC**| Both | procedure_code | str | 0.00% | 513 | `2.0101e+08` $\rightarrow$ `5.0502e+08` | 10-digit procedure code; modal: `0310010039` |
| **PROC_REA** | Both | procedure_code | str | 0.00% | 492 | `2.0101e+08` $\rightarrow$ `5.0502e+08` | 10-digit performed code; modal: `0310010039` |
| **VAL_SH** | Both | numeric | str | 0.00% | 5,618 | `0` $\rightarrow$ `5.84995e+06` | Hospital cost share (decimal values) |
| **VAL_SP** | Both | numeric | str | 0.00% | 2,123 | `0` $\rightarrow$ `1.62758e+06` | Professional cost share (decimal values) |
| **VAL_TOT** | Both | numeric | str | 0.00% | 5,737 | `0` $\rightarrow$ `1.13995e+07` | Total billing value; modal: `688.11` (257) |
| **VAL_UTI** | Both | numeric | str | 0.00% | 126 | `0` $\rightarrow$ `4.84403e+06` | Cost spent inside the ICU; modal: `0` (12106) |
| **US_TOT** | Both | numeric | str | 0.00% | 5,406 | `0` $\rightarrow$ `2.22213e+06` | Total stay units / structural index |
| **DT_INTER** | Processed | date | str | 0.00% | 209 | `2008-01-01` $\rightarrow$ `2022-01-31` | ISO standard admission date |
| | Raw | date | str | 0.00% | 209 | `2008-01-01` $\rightarrow$ `2022-01-31` | `YYYYMMDD` string shape; digit:8 |
| **DT_SAIDA** | Processed | date | str | 0.00% | 170 | `2021-08-03` $\rightarrow$ `2022-01-31` | ISO standard discharge date |
| | Raw | date | str | 0.00% | 170 | `2021-08-03` $\rightarrow$ `2022-01-31` | `YYYYMMDD` string shape; digit:8 |
| **DIAG_PRINC**| Both | ICD10_code | str | 0.00% | 1,482 | ICD-10 keys | Modal: `"O800"` (1534), `"O821"` (311), `"O808"` (213) |
| **COBRANCA** | Processed | medium_cardinality | str | 0.00% | 24 | Categorical names | `"Alta melhorado"` (5814), `"Alta da mãe"` (3286) |
| | Raw | numeric | str | 0.00% | 24 | `11` $\rightarrow$ `66` | `"12"` (5814), `"61"` (3286), `"31"` (897); digit:2 |
| **NAT_JUR** | Processed | low_cardinality_cat | str | 0.00% | 9 | Decoded string | `"Associação Privada"` (3757), `"Órgão Público"` (2692) |
| | Raw | numeric | str | 0.00% | 9 | `1023` $\rightarrow$ `3999` | `"3999"` (3757), `"1023"` (2692), `"2062"` (2131); digit:4 |
| **GESTAO** | Processed | low_cardinality_cat | str | 0.00% | 2 | `"Municipal plena assist"` (9559), `"Estadual plena"` (3295) | Decoded string |
| | Raw | numeric | str | 0.00% | 2 | `1` $\rightarrow$ `2` | `"1"` (9559), `"2"` (3295); digit_code_like:1 |
| **IND_VDRL** | Processed | low_cardinality_cat | str | 0.00% | 2 | `"Não"` (9884), `"Sim"` (2970) | Decoded syphilis test indicator |
| | Raw | numeric | str | 0.00% | 2 | `0` $\rightarrow$ `1` | `"0"` (9884), `"1"` (2970); digit_code_like:1 |
| **MUNIC_MOV** | Both | municipality_code | str | 0.00% | 40 | `270010` $\rightarrow$ `270940` | 6-digit transaction municipality code |
| **COD_IDADE** | Processed | low_cardinality_cat | str | 0.00% | 4 | Unit of age descriptor | `"Anos"` (12033), `"Dias"` (641), `"Meses"` (166), etc. |
| | Raw | numeric | str | 0.00% | 4 | `2` $\rightarrow$ `5` | `"4"` (12033), `"2"` (641), `"3"` (166), `"5"` (14); digit:1 |
| **IDADE** | Both | numeric | str | 0.00% | 100 | `0` $\rightarrow$ `99` | Patient age; modal: `23` (347), `20` (336) |
| **DIAS_PERM** | Both | numeric | str | 0.00% | 81 | `0` $\rightarrow$ `155` | Patient stay length; modal: `1` (3112), `2` (3049) |
| **MORTE** | Processed | low_cardinality_cat | str | 0.00% | 2 | `"Não"` (12271), `"Sim"` (583) | Decoded boolean death flag |
| | Raw | numeric | str | 0.00% | 2 | `0` $\rightarrow$ `1` | `"0"` (12271), `"1"` (583); digit_code_like:1 |
| **NACIONAL** | Processed | low_cardinality_cat | str | 0.00% | 5 | Resolved country name | `"Brasil"` (12848), `"Ilhas virgens"` (3), etc. |
| | Raw | numeric | str | 0.00% | 5 | `10` $\rightarrow$ `109` | `"010"` (12848), `"082"` (3); digit_code_like:3 |
| **CAR_INT** | Both | numeric | str | 0.00% | 4 | `1` $\rightarrow$ `6` | `"02"` (10295), `"01"` (2363); Character of stay; digit:2 |
| **HOMONIMO** | Processed | low_cardinality_cat | str | 0.00% | 2 | `"Não"` (12813), `"Sim"` (41) | Flag indicating patient has a homonym |
| | Raw | numeric | str | 0.00% | 2 | `0` $\rightarrow$ `2` | `"0"` (12813), `"2"` (41); digit_code_like:1 |
| **NUM_FILHOS**| Both | numeric | str | 0.00% | 5 | `0` $\rightarrow$ `6` | Count of children; modal: `0` (12816) |
| **INSTRU** | Both | numeric | str | 0.00% | 4 | `0` $\rightarrow$ `3` | Education level code; modal: `0` (12816) |
| **CID_NOTIF** | Both | ICD10_code | str | 99.70% | 1 | ICD-10 keys | Highly sparse; only value is `"Z302"` (38) |
| **CONTRACEP1**| Processed | numeric | str | 0.00% | 5 | `0` $\rightarrow$ `9` | `"00"` (12816), `"08"` (27), `"Hormônio oral"` (7) |
| | Raw | numeric | str | 0.00% | 5 | `0` $\rightarrow$ `11` | `"00"` (12816), `"08"` (27), `"10"` (7), `"11"` (3); digit:2 |
| **CONTRACEP2**| Processed | numeric | str | 0.00% | 4 | `0` $\rightarrow$ `8` | `"00"` (12823), `"Hormônio oral"` (19), `"Hormônio injetável"` (9) |
| | Raw | numeric | str | 0.00% | 4 | `0` $\rightarrow$ `11` | `"00"` (12823), `"10"` (19), `"11"` (9), `"08"` (3); digit:2 |
| **GESTRISCO** | Processed | low_cardinality_cat | str | 0.00% | 2 | `"Sim"` (12852), `"Não"` (2) | High-risk pregnancy boolean string |
| | Raw | numeric | str | 0.00% | 2 | `0` $\rightarrow$ `1` | `"1"` (12852), `"0"` (2); digit_code_like:1 |
| **INSC_PN** | Both | numeric | str | 0.00% | 9 | `0` $\rightarrow$ `11` | Prenatal registry ID; modal: `000000000000` (12827) |
| **GESTOR_COD**| Both | numeric | str | 0.00% | 15 | `0` $\rightarrow$ `143` | Code of manager; modal: `00000` (11836); digit:5 |
| **GESTOR_TP** | Both | numeric | str | 0.00% | 2 | `0` $\rightarrow$ `1` | Manager type; `"0"` (10731), `"1"` (2123); digit_code_like:1 |
| **GESTOR_CPF**| Processed | procedure_code | str | 0.00% | 24 | `0` $\rightarrow$ `9.59529e+10`| Modal: `0` (10731), `5841331469` (912); digit:1,9,10,11 |
| | Raw | numeric | str | 0.00% | 24 | `0` $\rightarrow$ `9.59529e+10`| Modal: `000000000000000` (10731); digit:15 |
| **CNPJ_MANT**| Both | numeric | str | 57.87% | 29 | `8.43955e+12` $\rightarrow$ `2.44641e+13` | CNPJ of supporting entity; modal: `12200259000165` (2692) |
| **COMPLEX** | Both | numeric | str | 0.00% | 2 | `2` $\rightarrow$ `3` | Complexity level; `"02"` (12151), `"03"` (703); digit:2 |
| **FINANC** | Both | numeric | str | 0.00% | 2 | `4` $\rightarrow$ `6` | Funding block; `"06"` (12794), `"04"` (60); digit:2 |
| **FAEC_TP** | Both | municipality_code | str | 99.53% | 2 | `40032` $\rightarrow$ `40066` | `"040066"` (47), `"040032"` (13); digit:6 |
| **REGCT** | Processed | low_cardinality_cat | str | 0.00% | 3 | Descriptors | `"0000"` (11899), `"ESTABELECIMENTO SEM GERACAO..."` (648) |
| | Raw | numeric | str | 0.00% | 3 | `0` $\rightarrow$ `7,109` | `"0000"` (11899), `"7109"` (648), `"7102"` (307); digit:4 |
| **RACA_COR** | Processed | numeric | str | 28.38% | 5 | `1` $\rightarrow$ `5` | `"03"` (8364), `"01"` (489), `"04"` (212); digit:2 |
| | Raw | numeric | str | 0.00% | 6 | `1` $\rightarrow$ `99` | `"03"` (8364), `"99"` (3648), `"01"` (489); digit:2 |
| **ETNIA** | Processed | numeric | str | 0.00% | 9 | `0` $\rightarrow$ `0` | `"0000"` (12837), `"XUKURU KARIRI"` (5), `"JIRIPANCO"` (4) |
| | Raw | numeric | str | 0.00% | 9 | `0` $\rightarrow$ `252` | `"0000"` (12837), `"0252"` (5), `"0069"` (4); digit:4 |
| **SEQUENCIA**| Both | numeric | str | 0.00% | 6,133 | `1` $\rightarrow$ `6,683` | Billing sequence; modal: `1` (34); digit:1,2 |
| **REMESSA** | Both | medium_cardinality | str | 0.00% | 37 | Delivery batch | e.g., `"HM27043001N202201.DTS"` (4295) |
| **AUD_JUST** | Both | low_cardinality_cat | str | 97.05% | 10 | Audit justifications | Highly sparse text (e.g., `"PACIENTE NAO APRESENTOU CNS"`) |
| **SIS_JUST** | Both | medium_cardinality | str | 97.05% | 35 | System justifications | Highly sparse text (e.g., `"ATENDIMENTO DE EMERGENCIA..."`) |
| **VAL_SH_FED**| Both | numeric | str | 0.00% | 8 | `0` $\rightarrow$ `52,127` | Federal hospital cost share; modal: `0` (12807) |
| **VAL_SP_FED**| Both | numeric | str | 0.00% | 8 | `0` $\rightarrow$ `30,173` | Federal professional cost share; modal: `0` (12807) |
| **VAL_UCI** | Both | numeric | str | 0.00% | 42 | `0` $\rightarrow$ `11,700` | Intermediate stay costs; modal: `0` (12402) |
| **MARCA_UCI**| Both | numeric | str | 0.00% | 3 | `0` $\rightarrow$ `2` | `"00"` (12402), `"01"` (443), `"02"` (9); digit:2 |
| **DIAGSEC1** | Both | ICD10_code | str | 87.01% | 185 | ICD-10 keys | `"W199"` (270), `"Y349"` (211); secondary diagnosis |
| **DIAGSEC2** | Both | ICD10_code | str | 99.69% | 11 | ICD-10 keys | `"E11"` (19), `"B238"` (7) |
| **DIAGSEC3** | Both | ICD10_code | str | 99.98% | 1 | ICD-10 keys | Only value is `"B238"` (2) |
| **TPDISEC1** | Processed | low_cardinality_cat | str | 87.01% | 2 | `"Adquirido"` (906), `"Pré-existente"` (764) | Decoded clinical state indicator |
| | Raw | numeric | str | 0.00% | 3 | `0` $\rightarrow$ `2` | `"0"` (11184), `"2"` (906), `"1"` (764); digit_code_like:1 |
| **TPDISEC2** | Both | numeric | str | 0.00% | 2 | `0` $\rightarrow$ `1` | `"0"` (12814), `"1"` (40); digit_code_like:1 |
| **TPDISEC3** | Both | numeric | str | 0.00% | 2 | `0` $\rightarrow$ `1` | `"0"` (12852), `"1"` (2); digit_code_like:1 |

### 3.2 100% Constant / Empty Columns in SIH-RD
* **VAL_SADT**, **VAL_RN**, **VAL_ACOMP**, **VAL_ORTP**, **VAL_SANGUE**, **VAL_SADTSR**, **VAL_TRANSP**, **VAL_OBSANG**, **VAL_PED1AC**, **DIAG_SECUN** (Fixed `"0000"`), **NATUREZA** (Fixed `"00"`), **RUBRICA**, **VAL_SH_GES**, **VAL_SP_GES**, **TPDISEC4** to **TPDISEC9** (Fixed `"0"`).
* **100% NA variables**: `NUM_PROC`, `CPF_AUT`, `GESTOR_DT`, `INFEHOSP`, `DIAGSEC4` through `DIAGSEC9`.

---

## 4. SIM-DO (Sistema de Informações sobre Mortalidade)

Tracks individual mortality records, timing parameters, and socio-demographic indicators.

### 4.1 SIM-DO Variables Catalog (Processed vs. Raw)

| Variable Name | Processed Encodings | Raw Encodings (where distinct) | Null % | Unique | observed kind / Notes |
| :--- | :--- | :--- | :---: | :---: | :--- |
| **ORIGEM** | `"1"` (23121), `"2"` (1) | Same | 0.00% | 2 | numeric |
| **TIPOBITO** | `"Não Fetal"` (23122) | `"2"` (23122) | 0.00% | 1 | low_cardinality_cat / numeric |
| **DTOBITO** | `"2022-01-04"` | `"04012022"`; digit:8 | 0.00% | 365 | date |
| **HORAOBITO**| `"0500"` (284), `"0600"` (279) | Same | 4.00% | 1,430 | numeric-like; military hour |
| **NATURAL** | `"ALAGOAS"` (19010) | `"827"` (19010); digit:3 | 6.6% | 35 | State of birth |
| **CODMUNNATU**| `"270430"` (3624) | Same | 6.87% | 608 | municipality_code |
| **DTNASC** | `"1954-06-06"` | `"06061954"`; digit:8 | 0.06% | 14,932| date |
| **IDADE** | `474` (556) | `474` (556); digit:3 | 0.00% | 196 | Structural age code |
| **SEXO** | `"Masculino"` (12736) | `"1"` (12736), `"2"` (10378), `"0"` (8) | 0.03% | 3 | low_cardinality_cat / numeric |
| **RACACOR** | `"Parda"` (15707) | `"4"` (15707), `"1"` (4949), `"2"` (1131)| 5.16% | 5 | low_cardinality_cat / numeric |
| **ESTCIV** | `"Solteiro"` (6642) | `"1"` (6642), `"2"` (6086), `"3"` (4527)| 17.62%| 6 | low_cardinality_cat / numeric |
| **ESC** | `"Nenhuma"` (6417) | `"1"` (6417), `"3"` (3341), `"9"` (2951)| 30.09%| 6 | Education level |
| **ESC2010** | `"0"` (6387), `"1"` (4653) | Same | 17.51%| 7 | Education (2010 structure) |
| **SERIESCFAL**| `"4"` (1866), `"8"` (793) | Same | 80.90%| 8 | School year of decedent |
| **OCUP** | `"Trabalhador volante..."` | `"999993"` (4632); CBO-2002 code | 31.40%| 514 | decedent occupation |
| **CODMUNRES**| `"270430"` (7203) | Same | 0.00% | 103 | municipality_code |
| **LOCOCOR** | `"Hospital"` (13755) | `"1"` (13755), `"3"` (5970) | 0.05% | 6 | Event setting |
| **CODESTAB** | `"2006510"` (2548) | Same | 34.48%| 251 | CNES code of event facility |
| **CODMUNOCOR**| `"270430"` (9866) | Same | 0.00% | 193 | Occurrence municipality code |
| **IDADEMAE** | `20` (39), `19` (31) | Same | 97.83%| 34 | Maternal age in infant death |
| **ESCMAE** | `"8 a 11 anos"` (222) | `"4"` (222), `"3"` (117) | 97.94%| 6 | Maternal education in infant death |
| **ESCMAE2010**| `3` (176), `2` (155) | Same | 97.94%| 7 | Maternal education scale |
| **SERIESCMAE**| `8` (46), `3` (33) | Same | 99.41%| 8 | Maternal school grade |
| **OCUPMAE** | `"Trabalhador volante..."` | `"999992"` (187); CBO-2002 code | 98.30%| 48 | Maternal occupation |
| **QTDFILVIVO**| `1` (160), `0` (115) | `"01"` (160), `"00"` (115); digit:2 | 97.87%| 14 | Living children count |
| **QTDFILMORT**| `0` (343), `1` (81) | `"00"` (343), `"01"` (81); digit:2 | 97.99%| 7 | Deceased children count |
| **GRAVIDEZ** | `"única"` (488), `"Dupla"` (33) | `"1"` (488), `"2"` (33); digit:1 | 97.74%| 4 | Pregnancy type |
| **SEMAGESTAC**| `39` (41), `38` (36) | Same | 97.91%| 32 | Weeks of gestation |
| **GESTACAO** | `"37 a 41 semanas"` (147) | `"5"` (147), `"2"` (118); digit:1 | 97.95%| 7 | Gestation brackets |
| **PARTO** | `"Vaginal"` (291) | `"1"` (291), `"2"` (230) | 97.75%| 3 | Delivery type |
| **OBITOPARTO**| `"Depois"` (505) | `"3"` (505), `"9"` (11) | 97.82%| 2 | Death timeline relative to delivery |
| **PESO** | `3100` (5), `2500` (5) | Same | 97.82%| 376 | Birth weight in grams |
| **TPMORTEOCO**| `8` (1034), `9` (342) | Same | 93.85%| 7 | Maternal death scenario indicator |
| **OBITOGRAV** | `"Não"` (1068), `"Sim"` (15) | `"2"` (1068), `"9"` (342), `"1"` (15)| 93.84%| 3 | Death during pregnancy |
| **OBITOPUERP**| `"Não"` (1052) | `"3"` (1052), `"9"` (342), `"1"` (19)| 95.32%| 4 | Death in puerperium |
| **ASSISTMED**| `"Sim"` (11565), `"Não"` (2795)| `"1"` (11565), `"2"` (2795), `"9"` (732)| 34.73%| 3 | Medical assistance flag |
| **EXAME** | `"Não"` (18), `"Sim"` (1) | `"2"` (18), `"9"` (2), `"1"` (1) | 99.91%| 3 | Examination performed flag |
| **CIRURGIA** | `"Não"` (20), `"Sim"` (1) | `"2"` (20), `"1"` (1) | 99.91%| 2 | Surgery flag |
| **NECROPSIA**| `"Não"` (12187), `"Sim"` (2282)| `"2"` (12187), `"1"` (2282), `"9"` (394)| 35.72%| 3 | Autopsy flag |
| **LINHAA** | `"*A419"` (3079) | Same | 2.14% | 889 | Underlying diagnostic chain line A |
| **LINHAB** | `"*A419"` (1299) | Same | 21.15%| 1479| Diagnostic chain line B |
| **LINHAC** | `"*I10X"` (868) | Same | 51.19%| 1315| Diagnostic chain line C |
| **LINHAD** | `"*I10X"` (603) | Same | 78.78%| 951 | Diagnostic chain line D |
| **LINHAII** | `"*I10X"` (1342) | Same | 58.06%| 2765| Diagnostic chain line II |
| **CAUSABAS** | `"I219"` (1848), `"R99"` (988) | Same | 0.00% | 1470| Underlying cause of death (ICD-10) |
| **COMUNSVOIM**| `"270430"` (2930) | Same | 82.59%| 38 | SVO/IML municipality code |
| **DTATESTADO**| `"2022-01-31"` | `"31012022"`; digit:8 | 0.00% | 387 | Date of death certificate |
| **CIRCOBITO**| `"Homicídio"` (1086) | `"3"` (1086), `"1"` (937), `"9"` (209) | 89.29%| 5 | Circumstances of death |
| **ACIDTRAB** | `"Não"` (259), `"Sim"` (54) | `"2"` (259), `"9"` (121), `"1"` (54) | 98.12%| 3 | Work accident flag |
| **FONTE** | `"Boletim de Ocorrência"` | `"1"` (893), `"2"` (185), `"9"` (104) | 94.52%| 5 | Information source |
| **NUMEROLOTE**| `20230001` (644) | Same | 0.00% | 142 | Processing batch ID; digit:8 |
| **TPPOS** | `"Investigado"` (3882) | `"S"` (3882), `"N"` (3038) | 70.07%| 2 | Post-mortem investigation status |
| **DTINVESTIG**| `"2023-07-24"` | `"24072023"`; digit:8 | 83.29%| 468 | Date of investigation |
| **CAUSABAS_O**| `"I219"` (1763) | Same | 0.01% | 1469| Original underlying cause code |
| **DTCADASTRO**| `"2022-08-04"` | `"04082022"`; digit:8 | 0.00% | 402 | Date of database entry |
| **ATESTANTE**| `"Sim"` (7711) | `"1"` (7711), `"2"` (5951) | 0.24% | 5 | Attending physician status |
| **STCODIFICA**| `"S"` (23118), `"N"` (2) | Same | 0.01% | 2 | Code generation system status flag |
| **CODIFICADO**| `"S"` (23120), `"N"` (2) | Same | 0.00% | 2 | Coding flag status |
| **VERSAOSIST**| `"3.2.30"` (22976) | Same | 0.00% | 4 | System version descriptor |
| **VERSAOSCB**| `"3.4"` (22714) | Same | 0.05% | 3 | Verification block version |
| **FONTEINV** | `"Múltiplas fontes"` (2362) | `"8"` (2362), `"3"` (826); digit:1 | 83.88%| 9 | Source of investigation information |
| **DTRECEBIM**| `"2022-10-19"` | `"19102022"`; digit:8 | 0.00% | 389 | Ingestion date |
| **ATESTADO** | `"R99"` (965) | Same | 0.00% | 16300| Direct certificate text |
| **DTRECORIGA**| `"2022-10-19"` | `"19102022"`; digit:8 | 0.00% | 308 | Date of original file collection |
| **ESCMAEAGR1**| `12` (119), `11` (85) | `"12"` (119), `"11"` (85); digit:2 | 97.94%| 13 | Aggregated maternal education |
| **ESCFALAGR1**| `00` (6387), `09` (2949) | Same | 17.51%| 13 | Aggregated decedent education |
| **STDOEPIDEM**| `0` (23122) | Same | 0.00% | 1 | Epidem-related notification status flag |
| **STDONOVA** | `1` (23122) | Same | 0.00% | 1 | Entry novelty validation code |
| **DIFDATA** | `010` (523), `008` (512) | Same | 0.00% | 657 | Processing delay index in days |
| **NUDIASOBCO**| `57` (18), `44` (18) | Same | 94.98%| 218 | Days elapsed since birth |
| **DTCADINV** | `"2022-12-05"` | `"05122022"`; digit:8 | 94.98%| 294 | Date of investigation entry |
| **TPOBITOCOR**| `9` (1107), `5` (24) | Same | 94.98%| 8 | Event location type |
| **DTCONINV** | `"2022-12-05"` | `"05122022"`; digit:8 | 94.98%| 306 | Investigation completion date |
| **FONTES** | `"SXXSXX"` (221) | Same | 97.44%| 22 | Investigation data verification matrix |
| **TPRESGINFO**| `1` (45), `2` (4) | Same | 99.78%| 3 | Information source type |
| **TPNIVELINV**| `"M"` (1161) | Same | 94.98%| 1 | Level of post-mortem investigation |
| **DTCADINF** | `"2022-09-05"` | `"05092022"`; digit:8 | 97.44%| 225 | System update date |
| **MORTEPARTO**| `3` (583), `1` (5) | Same | 97.44%| 4 | Obstetric death time categorization |
| **DTCONCASO** | `"2022-09-05"` | `"05092022"`; digit:8 | 97.50%| 226 | Case lock date |
| **ALTCAUSA** | `2` (549), `1` (43) | Same | 97.44%| 2 | Code change status following investigation |
| **CONTADOR** | `529` (1) | Same | 0.00% | 23122| Database internal row sequence key |

### 4.2 100% Constant / Empty Columns in SIM-DO
* **CAUSAMAT**, **CB_PRE**, **NUDIASOBIN**, **NUDIASINF**, **FONTESINF**.

---

## 5. SINASC (Sistema de Informações sobre Nascidos Vivos)

This dataset registers detailed structural and maternal variables for every live birth.

### 5.1 SINASC Variables Catalog (Processed vs. Raw)

| Variable Name | Processed Encodings | Raw Encodings (where distinct) | Null % | Unique | Observed Kind & Description |
| :--- | :--- | :--- | :---: | :---: | :--- |
| **ORIGEM** | `1` (45742) | Same | 0.00% | 1 | numeric |
| **CODESTAB** | `"2006340"` (4602) | Same | 0.43% | 154 | municipality_code |
| **CODMUNNASC**| `"270430"` (20416) | Same | 0.00% | 123 | municipality_code |
| **LOCNASC** | `"Hospital"` (45276) | `"1"` (45276), `"2"` (269), `"9"` (1)| 0.00% | 5 | low_cardinality_cat / numeric |
| **IDADEMAE** | `22` (2623), `21` (2584) | Same | 0.00% | 44 | Maternal age in years |
| **ESTCIVMAE** | `"Solteira"` (19556) | `"1"` (19556), `"5"` (12819) | 0.64% | 6 | low_cardinality_cat / numeric |
| **ESCMAE** | `"8 a 11 anos"` (29145) | `"4"` (29145), `"3"` (8285), `"9"` (7) | 0.03% | 6 | Maternal education classification |
| **CODOCUPMAE**| `"Trabalhador volante..."` | `"999992"` (18617) | 3.54% | 455 | CBO-2002 occupation code of mother |
| **QTDFILVIVO**| `0` (18092), `1` (15015) | `"00"` (18092), `"01"` (15015) | 0.12% | 17 | Living children count |
| **QTDFILMORT**| `0` (36248), `1` (7469) | `"00"` (36248), `"01"` (7469) | 0.16% | 11 | Deceased children count |
| **CODMUNRES**| `"270430"` (13027) | Same | 0.00% | 102 | municipality_code |
| **GESTACAO** | `"37 a 41 semanas"` (38605) | `"5"` (38605), `"4"` (4700) | 0.05% | 6 | Gestational age groups |
| **GRAVIDEZ** | `"única"` (44895) | `"1"` (44895), `"2"` (827), `"3"` (20) | 0.00% | 3 | Birth plurality |
| **PARTO** | `"Cesáreo"` (25536) | `"2"` (25536), `"1"` (20204) | 0.00% | 2 | Delivery type |
| **CONSULTAS** | `"7 ou mais vezes"` (32815)| `"4"` (32815), `"3"` (9637), `"9"` (270)| 0.59% | 5 | Prenatal visits |
| **DTNASC** | `"2022-04-18"` | `"18042022"`; digit:8 | 0.00% | 365 | date |
| **HORANASC** | `"1130"` (119) | Same | 0.00% | 1,440| Birth hour (military format) |
| **SEXO** | `"Masculino"` (23445) | `"1"` (23445), `"2"` (22289), `"0"` (8)| 0.02% | 3 | low_cardinality_cat / numeric |
| **APGAR1** | `9` (21082) | `"09"` (21082), `"99"` (1); digit:2 | 0.80% | 12 | APGAR 1 minute score |
| **APGAR5** | `9` (32101) | `"09"` (32101), `"99"` (2); digit:2 | 0.78% | 12 | APGAR 5 minute score |
| **RACACOR** | `"Parda"` (41158) | `"4"` (41158), `"1"` (3205) | 1.14% | 5 | Newborn race/color |
| **PESO** | `3400` (257) | Same | 0.00% | 2255 | Weight in grams |
| **IDANOMAL** | `"Não"` (45184), `"Sim"` (404) | `"2"` (45184), `"1"` (404), `"9"` (84) | 0.34% | 3 | Congenital anomaly flag |
| **DTCADASTRO**| `"2023-07-06"` | `"06072023"`; digit:8 | 0.00% | 448 | Registry entry date |
| **CODANOMAL** | `"Q699"` (58) | Same | 99.12%| 167 | Congenital anomaly (ICD-10 code) |
| **NUMEROLOTE**| `20230029` (1746) | Same | 0.00% | 109 | Batch ID |
| **VERSAOSIST**| `"3.2.50"` (41441) | Same | 0.00% | 3 | System software version |
| **DTRECEBIM**| `"2022-10-19"` | `"19102022"`; digit:8 | 0.00% | 330 | Processing receipt date |
| **DIFDATA** | `010` (1127), `011` (1115) | Same | 0.00% | 643 | Delay metrics |
| **NATURALMAE**| `827` (41371) | Same | 0.09% | 27 | Mother's state of birth |
| **CODMUNNATU**| `"270430"` (13067) | Same | 0.09% | 787 | Mother's municipality of birth code |
| **CODUFNATU** | `27` (41371), `26` (1649) | Same | 0.09% | 27 | Mother's birth state code |
| **ESCMAE2010**| `3` (22570), `2` (13466) | Same | 0.02% | 7 | Maternal education scale |
| **SERIESCMAE**| `3` (15554), `8` (6575) | Same | 16.20%| 8 | Maternal school grade level |
| **DTNASCMAE** | `"1998-06-01"` | `"01061998"`; digit:8 | 0.22% | 10034| Maternal birth date |
| **RACACORMAE**| `"Parda"` (41158) | `"4"` (41158), `"1"` (3205) | 1.14% | 5 | Maternal race/color |
| **QTDGESTANT**| `0` (15813), `1` (13830) | `"00"` (15813), `"01"` (13830); digit:2 | 0.07% | 17 | Count of prior pregnancies |
| **QTDPARTNOR**| `0` (28223), `1` (9136) | `"00"` (28223), `"01"` (9136); digit:2 | 0.08% | 16 | Count of normal deliveries |
| **QTDPARTCES**| `0` (32764), `1` (9386) | `"00"` (32764), `"01"` (9386); digit:2 | 0.09% | 8 | Count of cesarean deliveries |
| **IDADEPAI** | `30` (214), `31` (212) | Same | 90.40%| 54 | Paternal age |
| **DTULTMENST**| `"2021-05-05"` | `"05052021"`; digit:8 | 15.11%| 470 | Date of last menstruation |
| **SEMAGESTAC**| `39` (11810), `40` (9199) | Same | 0.05% | 27 | Calculated gestation weeks |
| **TPMETESTIM**| `8` (38832), `1` (5161) | Same | 0.05% | 4 | Gestational estimate methodology |
| **CONSPRENAT**| `8` (7032), `7` (6122) | `"08"` (7032), `"07"` (6122); digit:2 | 0.39% | 37 | Count of prenatal consultations |
| **MESPRENAT** | `2` (18260), `1` (10956) | `"02"` (18260), `"01"` (10956); digit:2 | 0.59% | 10 | Prenatal start month |
| **TPAPRESENT**| `1` (44481), `2` (1210) | Same | 0.00% | 4 | Presentation type |
| **STTRABPART**| `2` (42145), `1` (3356) | Same | 0.21% | 3 | Labor laboring status |
| **STCESPARTO**| `3` (20204), `2` (17015) | Same | 0.15% | 4 | Cesarean section delivery status |
| **TPNASCASSI**| `1` (35459), `2` (9924) | Same | 0.07% | 5 | Attendant type |
| **TPFUNCRESP**| `2` (26123), `5` (19113) | Same | 0.43% | 5 | Responsible professional function |
| **TPDOCRESP** | `3` (18978), `0` (10036) | Same | 0.00% | 6 | Document registrar type |
| **DTDECLARAC**| `"2022-05-11"` | `"11052022"`; digit:8 | 0.09% | 399 | Declaration issuance date |
| **ESCMAEAGR1**| `6` (15060), `5` (7374) | `"06"` (15060), `"05"` (7374); digit:2 | 0.02% | 13 | Maternal education level (aggregated) |
| **STDNEPIDEM**| `0` (32348), `1` (13394) | Same | 0.00% | 2 | Epidem-related notification status flag |
| **STDNNOVA** | `1` (45742) | Same | 0.00% | 1 | Entry novelty validation code |
| **CODPAISRES**| `1` (45742) | Same | 0.00% | 1 | Country of residence code |
| **TPROBSON** | `3` (12020), `5` (11120) | `"03"` (12020), `"05"` (11120); digit:2 | 0.00% | 11 | Robson Classification code |
| **PARIDADE** | `1` (29995), `0` (15747) | Same | 0.00% | 2 | Parity binary flag |
| **KOTELCHUCK**| `5` (30028), `2` (7374) | Same | 0.00% | 6 | Prenatal adequacy index |
| **CONTADOR** | `115206` (1) | Same; digit:6 | 0.00% | 45742| Sequential row ID |

### 5.2 100% Constant / Empty Columns in SINASC
* **DTRECORIGA** (100% missing in both Raw and Processed).

---

