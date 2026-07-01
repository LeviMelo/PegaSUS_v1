# SIDRA / IBGE METADATA REGISTRY: DEVELOPER SYSTEM SPECIFICATION

This compendium serves as a comprehensive, structured, and compact specification of the database substrate. It is designed to assist project developers and data analysts in mapping table structures, variable dependencies, and classification hierarchies.

---

## 1. NOMENCLATURE & ABBREVIATIONS KEY

To optimize readability and density, the following abbreviations and acronyms are used systematically throughout this specification:

### General & Structural
*   **Tab**: Table Code (Unique ID)
*   **Res**: Research (Source Survey)
*   **Subj**: Subject Name
*   **Per**: Temporal coverage (Periods available)
*   **Locs**: Localities covered (Municipalities count)
*   **GR / UF**: Grandes Regiões / Unidades da Federação
*   **PC / HH**: Per Capita / Household (Domiciliar)

### Variable Meta
*   **V**: Variable ID
*   **U**: Unit
    *   `Un`: Unidades (Units/Count)
    *   `Px`: Pessoas (Persons/Count)
    *   `R$`: Brazilian Real
    *   `milR$`: Thousand Brazilian Reais (R$ x 1,000)
    *   `SM`: Salários Mínimos (Minimum Wages)
    *   `%`: Percentagem (Percentage)
    *   `Km²`: Quilômetros Quadrados
    *   `Hab/Km²`: Habitantes por Quilômetro Quadrado
    *   `Km`: Quilômetros (Kilometers)
    *   `An`: Anos (Years)
    *   `Rz`: Razão (Ratio)
*   **Type**: Inferred variable data type
    *   `c`: Count (Absolute frequency)
    *   `m`: Monetary or nominal amount
    *   `p`: Percent, share, or distribution
    *   `md`: Median or average value
    *   `r`: Rate or statistical index
    *   `o`: Other geographic/physical measurement
*   **DK**: Default Keep status (Y = True, N = False)

### Classifications & Categories
*   **Clsf**: Classification ID & Name
*   **Axis**: Inferred Axis type (e.g., `sex`, `race_color`, `education`, `age`, `labor`, `sanitation_water`, `sanitation_sewage`, `waste`, `housing`, `income`)
*   **Cats**: Category breakdown (Count & key IDs/names)

---

## 2. METADATA REGISTRY MAP

---

### GROUP 1: ECONOMIC CONTEXT (CEMPRE, PIB)

#### Tab 1685 | Local units, enterprises, employees & wages (Série encerrada 2021)
*   **Tier**: T1_CORE | **Res**: Cadastro Central de Empresas (CEMPRE) | **Subj**: Unidades locais | **Per**: 2006–2021 | **Locs**: 5570
*   **Vars**: 
    *   `706` (Número de unidades locais; U: Un, Type: c, DK: Y)
    *   `367` (Número de empresas e outras organizações atuantes; U: Un, Type: c, DK: Y)
    *   `707` (Pessoal ocupado total; U: Px, Type: c, DK: Y)
    *   `708` (Pessoal ocupado assalariado; U: Px, Type: c, DK: Y)
    *   `5944` (Pessoal assalariado médio; U: Px, Type: c, DK: Y)
    *   `662` (Salários e outras remunerações; U: milR$, Type: m, DK: Y)
    *   `1606` (Salário médio mensal; U: SM, Type: m, DK: Y)
    *   `10143` (Salário médio mensal em reais; U: R$, Type: m, DK: Y)
*   **Clsfs**: None

#### Tab 9509 | Local units, enterprises, employees & wages
*   **Tier**: T1_CORE | **Res**: Cadastro Central de Empresas | **Subj**: Unidades locais | **Per**: 2022–2023 | **Locs**: 5570
*   **Vars**: Same as Tab 1685 (`706`, `367`, `707`, `708`, `5944`, `662`, `1606`, `10143`)
*   **Clsfs**: None

#### Tab 21 | Municipal Gross Domestic Product & GVA (Série encerrada 1999–2012)
* **Tier**: T1_CORE | **Res**: PIB dos Municípios | **Subj**: Contas Nacionais e Regionais | **Per**: 1999–2012 | **Locs**: 5565
* **Vars**: 
    * `37` (PIB preços correntes; U: milR$, Type: m, DK: Y)
    * `543` (Impostos, líquidos de subsídios sobre produtos; U: milR$, Type: m, DK: Y)
    * `498` (Valor adicionado bruto (VAB) total; U: milR$, Type: m, DK: Y)
    * `513` (VAB Agropecuária; U: milR$, Type: m, DK: Y)
    * `517` (VAB Indústria; U: milR$, Type: m, DK: Y)
    * `521` (VAB Serviços total, inc. Adm. Pública; U: milR$, Type: m, DK: Y)
    * `525` (VAB Administração, defesa, edu, saúde públicas e seguridade; U: milR$, Type: m, DK: Y)
    * *Shares & Regional Participation (U: %, Type: p, DK: N)*: `553` (PIB microrreg), `552` (PIB mesorreg), `497` (PIB UF), `530` (PIB GR), `496` (PIB BR), `571`/`570`/`545`/`551`/`544` (Taxes share micr/meso/UF/GR/BR), `555`/`554`/`500`/`546`/`499` (VAB Total share), `516`/`557`/`556`/`515`/`547`/`514` (VAB Agro share), `520`/`559`/`558`/`519`/`548`/`518` (VAB Ind share), `524`/`561`/`560`/`523`/`549`/`522` (VAB Serv share), `528`/`563`/`562`/`527`/`550`/`526` (VAB Adm share)
* **Clsfs**: None

#### Tab 5938 | Gross Domestic Product, taxes & Gross Value Added (Ref. 2010)
*   **Tier**: T1_CORE | **Res**: PIB dos Municípios | **Subj**: Contas Nacionais e Regionais | **Per**: 2002–2023 | **Locs**: 5570
*   **Vars**: Same as Tab 21, with updated structure for services:
    *   `6575` (VAB Serviços, exclusive administração pública; U: milR$, Type: m, DK: Y)
    *   `6574` (Participação do VAB serviços exc. adm. pública no total; U: %, Type: p, DK: N)
    *   *Additional regional shares*: `6571` (microrreg), `6570` (mesorreg), `6572` (UF), `6569` (GR), `6573` (BR) (U: %, Type: p, DK: N)
*   **Clsfs**: None

---

### GROUP 2: DEMOGRAPHICS, POPULATION STRUCTURE & GEOGRAPHY

#### Tab 6579 | Resident population estimates (Post-censal series)
* **Tier**: T1_CORE | **Res**: Estimativas de População | **Subj**: Geral | **Per**: 2001–2025 excl. 2007 (IBGE estimation gap), 2010/2022 (census years, superseded by Tab 9606), 2023 (post-2022-census processing lag) — verified live against IBGE SIDRA on 2026-07-01; do not re-hardcode this list in code, trust the table's own live `periods` metadata | **Locs**: 5570
* **Vars**: `9324` (População residente estimada; U: Px, Type: c, DK: Y)
* **Clsfs**: None
* **Cube role**: intercensal total-only population anchor, paired with Tab 9606 (census years, full sex/race/age disaggregation) in the population denominator tensor (MSD §2.8.10). See `config/registries/sidra/sidra_stitching.yaml`.

#### Tab 4714 | Land Area, Resident Population & Demographic Density
*   **Tier**: T1_CORE | **Res**: Censo Demográfico | **Subj**: Território | **Per**: 2022 | **Locs**: 5570
*   **Vars**: 
    *   `93` (População residente; U: Px, Type: c, DK: Y)
    *   `6318` (Área da unidade territorial; U: Km², Type: o, DK: Y)
    *   `614` (Densidade demográfica; U: Hab/Km², Type: o, DK: Y)
*   **Clsfs**: None

#### Tab 9515 | Essential demographic summary indices
*   **Tier**: T2_CONTEXT | **Res**: Censo Demográfico | **Subj**: Pessoas | **Per**: 2022 | **Locs**: 5570
*   **Vars**: 
    *   `10612` (Índice de envelhecimento; U: Rz, Type: r, DK: Y)
    *   `10613` (Idade mediana; U: An, Type: md, DK: Y)
    *   `8845` (Razão de sexo; U: Rz, Type: o, DK: Y)
*   **Clsfs**: None

#### Tab 136 | Resident population by Color or Race (Histórica)
* **Tier**: T1_CORE | **Res**: Censo Demográfico | **Subj**: Cor ou raça | **Per**: 1991 | 2000 | 2010 | **Locs**: 5507 (2000) | 5565 (1991, 2010)
* **Vars**: `93` (U: Px, Type: c, DK: Y), `1000093` (População residente - % do total geral; U: %, Type: p, DK: N)
* **Clsfs**: 
    * `Clsf 86` (Cor ou raça; Axis: race_color, Cats: 7; `0` Total, `2776` Branca, `2777` Preta, `2778` Amarela, `2779` Parda, `2780` Indígena, `2781` Sem declaração)

#### Tab 9605 | Resident population by Color or Race (Comparabilidade)
*   **Tier**: T1_CORE | **Res**: Censo Demográfico | **Subj**: Cor ou raça | **Per**: 2010 | 2022 | **Locs**: 5570
*   **Vars**: `93` (U: Px, Type: c, DK: Y), `1000093` (U: %, Type: p, DK: N)
*   **Clsfs**: 
    *   `Clsf 86` (Cor ou raça; Axis: race_color, Cats: 6; `95251` Total, `2776` Branca, `2777` Preta, `2778` Amarela, `2779` Parda, `2780` Indígena)

#### Tab 9606 | Resident population by Color, Sex & Age (Full matrix)
*   **Tier**: T1_CORE | **Res**: Censo Demográfico | **Subj**: Cor ou raça | **Per**: 2010 | 2022 | **Locs**: 5570
*   **Vars**: `93` (U: Px, Type: c, DK: Y), `1000093` (U: %, Type: p, DK: N)
*   **Clsfs**: 
    *   `Clsf 86` (Cor ou raça; Axis: race_color, Cats: 6)
    *   `Clsf 2` (Sexo; Axis: sex, Cats: 3; `6794` Total, `4` Homens, `5` Mulheres)
    *   `Clsf 287` (Idade; Axis: age, Cats: 134; `100362` Total, `93070` 0-4, `6557` <1 [with detailed months `93071` to `93082`], single-years `1` to `100+`)

#### Tab 200 | Resident population by Sex, Situation & Age (Amostra histórica)
* **Tier**: T1_CORE | **Res**: Censo Demográfico | **Subj**: Pessoas | **Per**: 1970 | 1980 | 1991 | 2000 | 2010 | **Locs**: 5507 (2000) | 5565 (1970-1991, 2010)
* **Vars**: `93` (U: Px, Type: c, DK: Y), `1000093` (U: %, Type: p, DK: N)
* **Clsfs**: 
    * `Clsf 2` (Sexo; Axis: sex, Cats: 3; `0` Total, `4` Homens, `5` Mulheres)
    * `Clsf 1` (Situação do domicílio; Axis: housing|urban_rural, Cats: 3; `0` Total, `1` Urbana, `2` Rural)
    * `Clsf 58` (Grupo de idade; Axis: age, Cats: 50; `0` Total, `1140` 0-4, single-years `2483` to `2487`, five-year cohorts `1141` to `2503`, and single years up to `100+`, plus `3245` Idade ignorada)

#### Tab 1552 | Resident population by Situation, Sex, Age reporting & Age (2000–2010)
* **Tier**: T1_CORE | **Res**: Censo Demográfico | **Subj**: Pessoas | **Per**: 2000 | 2010 | **Locs**: 5507 (2000) | 5565 (2010)
* **Vars**: `93` (U: Px, Type: c, DK: Y), `1000093` (U: %, Type: p, DK: N)
* **Clsfs**: 
    * `Clsf 1` (Situação do domicílio; Axis: housing|urban_rural, Cats: 3; `0` Total, `1` Urbana, `2` Rural)
    * `Clsf 2` (Sexo; Axis: sex, Cats: 3; `0` Total, `92956` Homem, `92957` Mulher)
    * `Clsf 286` (Forma de declaração da idade; Axis: age, Cats: 3; `0` Total, `6555` Data de nascimento, `6556` Idade presumida)
    * `Clsf 287` (Idade; Axis: age, Cats: 132; `0` Total, single years, cohorts, months)

#### Tab 9514 | Resident population by Sex, Age & Age reporting (2022)
*   **Tier**: T1_CORE | **Res**: Censo Demográfico | **Subj**: Pessoas | **Per**: 2022 | **Locs**: 5570
*   **Vars**: `93` (U: Px, Type: c, DK: Y), `1000093` (U: %, Type: p, DK: N)
*   **Clsfs**: 
    *   `Clsf 2` (Sexo; Axis: sex, Cats: 3; `6794` Total, `4` Homens, `5` Mulheres)
    *   `Clsf 287` (Idade; Axis: age, Cats: 134)
    *   `Clsf 286` (Forma de declaração da idade; Axis: age|race_color, Cats: 3; `113635` Total, `6555` Data de nascimento, `6556` Idade presumida)

#### Tab 10211 | Resident population by Localization & Situation
*   **Tier**: T1_CORE | **Res**: Censo Demográfico | **Subj**: Geral | **Per**: 2022 | **Locs**: 5570
*   **Vars**: `93` (U: Px, Type: c, DK: Y), `1000093` (U: %, Type: p, DK: N)
*   **Clsfs**: 
    *   `Clsf 2661` (Localização do domicílio; Axis: housing, Cats: 3; `32776` Total, `50581` Em Unidades de Conservação, `56938` Fora de Unidades de Conservação)
    *   `Clsf 1` (Situação do domicílio; Axis: housing|urban_rural, Cats: 3; `6795` Total, `1` Urbana, `2` Rural)

#### Tab 202 | Population by Sex and Situation (Histórica)
* **Tier**: T1_CORE | **Res**: Censo Demográfico | **Subj**: Pessoas | **Per**: 1970 | 1980 | 1991 | 2000 | 2010 | **Locs**: 5507 (2000) | 5565 (1970-1991, 2010)
* **Vars**: `93` (U: Px, Type: c, DK: Y), `1000093` (U: %, Type: p, DK: N)
* **Clsfs**: 
    * `Clsf 2` (Sexo; Axis: sex, Cats: 3; `0` Total, `4` Homens, `5` Mulheres)
    * `Clsf 1` (Situação do domicílio; Axis: housing|urban_rural, Cats: 3; `0` Total, `1` Urbana, `2` Rural)

#### Tab 9923 | Population by Situation of Household (2022)
*   **Tier**: T1_CORE | **Res**: Censo Demográfico | **Subj**: Pessoas | **Per**: 2022 | **Locs**: 5570
*   **Vars**: `93` (U: Px, Type: c, DK: Y), `1000093` (U: %, Type: p, DK: N)
*   **Clsfs**: 
    *   `Clsf 1` (Situação do domicílio; Axis: housing|urban_rural, Cats: 3; `6795` Total, `1` Urbana, `2` Rural)

---

### GROUP 3: EDUCATION, LITERACY & COGNITIVE SKILLS

#### Tab 10141 | Pop 25+ by Sex, Education & Disability
*   **Tier**: T1_CORE | **Res**: Censo Demográfico | **Subj**: Educação | **Per**: 2022 | **Locs**: 5570
*   **Vars**: `1643` (Pessoas de 25 anos ou mais de idade; U: Px, Type: c, DK: Y), `10270` (Distribuição percentual; U: %, Type: p, DK: N)
*   **Clsfs**: 
    *   `Clsf 2` (Sexo; Axis: sex, Cats: 3; `6794` Total, `4` Homens, `5` Mulheres)
    *   `Clsf 1568` (Nível de instrução; Axis: education, Cats: 5; `120704` Total, `9493` Sem instrução e fund. incompleto, `9494` Fund. completo e médio incompleto, `9495` Médio completo e superior incompleto, `99713` Superior completo)
    *   `Clsf 839` (Existência de deficiência; Axis: disability, Cats: 3; `46583` Total, `58765` Pessoa com deficiência, `58766` Pessoa sem deficiência)

#### Tab 10142 | Pop 25+ by Race, Education & Disability
*   **Tier**: T1_CORE | **Res**: Censo Demográfico | **Subj**: Educação | **Per**: 2022 | **Locs**: 5570
*   **Vars**: `1643` (U: Px, Type: c, DK: Y), `10270` (U: %, Type: p, DK: N)
*   **Clsfs**: 
    *   `Clsf 86` (Cor ou raça; Axis: race_color, Cats: 6; `95251` Total, `2776` Branca, `2777` Preta, `2778` Amarela, `2779` Parda, `2780` Indígena)
    *   `Clsf 1568` (Nível de instrução; Axis: education, Cats: 5; `120704` Total, same categories as 10141)
    *   `Clsf 839` (Existência de deficiência; Axis: disability, Cats: 3; `46583` Total, `58765` Pessoa com deficiência, `58766` Pessoa sem deficiência)

#### Tab 1554 | Pop 10+ by Education (Amostra histórica 2010)
*   **Tier**: T1_CORE | **Res**: Censo Demográfico | **Subj**: Educação | **Per**: 2010 | **Locs**: 5565
*   **Vars**: `140` (Pessoas de 10 anos ou mais; U: Px, Type: c, DK: Y), `1000140` (U: %, Type: p, DK: N)
*   **Clsfs**: 
    *   `Clsf 1568` (Nível de instrução; Axis: education, Cats: 6; `0` Total, `9493` Sem instrução e fund. inc., `9494` Fund. comp. e méd. inc., `9495` Méd. comp. e sup. inc., `99713` Superior completo, `11626` Não determinado)

#### Tab 3547 | Pop 25+ by Sex & Education (2010)
*   **Tier**: T1_CORE | **Res**: Censo Demográfico | **Subj**: Educação | **Per**: 2010 | **Locs**: 5565
*   **Vars**: `1643` (U: Px, Type: c, DK: Y), `1001643` (U: %, Type: p, DK: N)
*   **Clsfs**: 
    *   `Clsf 2` (Sexo; Axis: sex, Cats: 3; `0` Total, `4` Homens, `5` Mulheres)
    *   `Clsf 1568` (Nível de instrução; Axis: education, Cats: 6; `0` Total, categories same as Tab 1554)

#### Tab 9543 | Literacy rate 15+ by Sex, Race & Age
*   **Tier**: T1_CORE | **Res**: Censo Demográfico | **Subj**: Educação | **Per**: 2022 | **Locs**: 5570 | **High Dim**: Yes
*   **Vars**: `2513` (Taxa de alfabetização das pessoas de 15 anos ou mais de idade; U: %, Type: p, DK: N)
*   **Clsfs**: 
    *   `Clsf 2` (Sexo; Axis: sex, Cats: 3; `6794` Total, `4` Homens, `5` Mulheres)
    *   `Clsf 86` (Cor ou raça; Axis: race_color, Cats: 6; `95251` Total, standard 5 races)
    *   `Clsf 287` (Idade; Axis: age, Cats: 60; `100362` Total, detailed age brackets e.g. `93086` 15-19, single ages `6572` 15, `6573` 16, up to `113623` 80+)

#### Tab 10062 | Mean Schooling Years (11+) by Age, Sex & Race
* **Tier**: T2_CONTEXT | **Res**: Censo Demográfico | **Subj**: Educação | **Per**: 2022 | **Locs**: 5570
* **Vars**: `13285` (Número médio de anos de estudo das pessoas com 11 anos ou mais de idade; U: An, Type: md, DK: Y)
* **Clsfs**: 
    * `Clsf 58` (Grupo de idade; Axis: age, Cats: 21; `95253` Total, cohorts: `14433` 11-14, `2792` 15-17, `100052` 18-24, `108866` 25+, five-year/ten-year intervals to `2503` 80+)
    * `Clsf 2` (Sexo; Axis: sex, Cats: 3; `6794` Total, `4` Homens, `5` Mulheres)
    * `Clsf 86` (Cor ou raça; Axis: race_color, Cats: 6; `95251` Total, standard 5 groups)

#### Tab 1699 | Population 10+, Literate Pop & Literacy Rate by Sex (2010)
*   **Tier**: T2_CONTEXT | **Res**: Censo Demográfico | **Subj**: Educação | **Per**: 2010 | **Locs**: 5565
*   **Vars**: 
    *   `140` (Pessoas de 10 anos ou mais de idade; U: Px, Type: c, DK: Y)
    *   `1000140` (Pessoas de 10 anos ou mais de idade - percentual; U: %, Type: p, DK: N)
    *   `1645` (Pessoas de 10 anos ou mais de idade, alfabetizadas; U: Px, Type: c, DK: Y)
    *   `1001645` (Pessoas de 10+ alfabetizadas - percentual; U: %, Type: p, DK: N)
    *   `1646` (Taxa de alfabetização das pessoas de 10 anos ou mais de idade; U: %, Type: p, DK: N)
*   **Clsfs**: 
    *   `Clsf 2` (Sexo; Axis: sex, Cats: 3; `6794` Total, `4` Homens, `5` Mulheres)

#### Tab 2987 | Population 10+, Literate Pop & Literacy Rate by Age (2000)
*   **Tier**: T2_CONTEXT | **Res**: Censo Demográfico | **Subj**: Educação | **Per**: 2000 | **Locs**: 5507
*   **Vars**: Same as Tab 1699 (`140`, `1000140`, `1645`, `1001645`, `1646`)
*   **Clsfs**: 
    *   `Clsf 58` (Grupo de idade; Axis: age, Cats: 4; `95253` Total, `1142` 10-14, `1143` 15-19, `109062` 20+)

---

### GROUP 4: HOUSING CHARACTERISTICS, SANITATION, WATER & SEWAGE

#### Tab 1436 | Households & Residents by Situation & Water Supply (2000)
* **Tier**: T1_CORE | **Res**: Censo Demográfico | **Subj**: Domicílios | **Per**: 2000 | **Locs**: 5507
* **Vars**: 
    * `96` (Domicílios particulares permanentes; U: Un, Type: c, DK: Y)
    * `1000096` (Domicílios particulares permanentes - percentual; U: %, Type: p, DK: N)
    * `137` (Moradores em domicílios particulares permanentes; U: Px, Type: c, DK: Y)
    * `1000137` (Moradores em domicílios - percentual; U: %, Type: p, DK: N)
* **Clsfs**: 
    * `Clsf 1` (Situação do domicílio; Axis: housing|urban_rural, Cats: 3; `0` Total, `1` Urbana, `2` Rural)
    * `Clsf 61` (Forma de abastecimento de água; Axis: sanitation_water, Cats: 12; `0` Total, `92853` Rede geral, `92847` Rede geral-canalizada cômodo, `92848` Rede geral-canalizada prop/terr, `92854` Poço/nascente prop, `92849` Poço/nascente prop-canalizada cômodo, `92850` Poço/nascente-canalizada prop/terr, `92851` Poço/nascente-não canalizada, `92852` Outra, `92866` Outra-canalizada cômodo, `92867` Outra-canalizada prop/terr, `92868` Outra-não canalizada)

#### Tab 1437 | Households & Residents by Situation & Sewage (2000)
* **Tier**: T1_CORE | **Res**: Censo Demográfico | **Subj**: Domicílios | **Per**: 2000 | **Locs**: 5507
* **Vars**: Same as Tab 1436 (`96`, `1000096`, `137`, `1000137`)
* **Clsfs**: 
    * `Clsf 1` (Situação do domicílio; Axis: housing|urban_rural, Cats: 3; `0` Total, `1` Urbana, `2` Rural)
    * `Clsf 11558` (Tipo de esgotamento sanitário; Axis: sanitation_sewage, Cats: 8; `0` Total, `92855` Rede geral ou pluvial, `92856` Fossa séptica, `92857` Fossa rudimentar, `92858` Vala, `92859` Rio/lago/mar, `92860` Outro, `92861` Não tinham banheiro nem sanitário)

#### Tab 1439 | Households & Residents by Situation & Waste Destination (2000)
* **Tier**: T1_CORE | **Res**: Censo Demográfico | **Subj**: Domicílios | **Per**: 2000 | **Locs**: 5507
* **Vars**: Same as Tab 1436 (`96`, `1000096`, `137`, `1000137`)
* **Clsfs**: 
    * `Clsf 1` (Situação do domicílio; Axis: housing|urban_rural, Cats: 3; `0` Total, `1` Urbana, `2` Rural)
    * `Clsf 67` (Destino do lixo; Axis: waste, Cats: 9; `0` Total, `2520` Coletado, `92863` Coletado por serviço limpeza, `92864` Coletado caçamba, `1087` Queimado prop, `1088` Enterrado prop, `1089` Jogado terreno/logradouro, `1090` Jogado rio/lago/mar, `1091` Outro destino)

#### Tab 2065 | Water Piping Existence & Supply (Amostra 2010)
*   **Tier**: T1_CORE | **Res**: Censo Demográfico | **Subj**: Domicílios | **Per**: 2010 | **Locs**: 5565
*   **Vars**: `96` (U: Un, Type: c, DK: Y), `1000096` (U: %, Type: p, DK: N)
*   **Clsfs**: 
    *   `Clsf 471` (Existência de água canalizada e forma de abastecimento; Axis: sanitation_water, Cats: 9; `0` Total, `12212` Tinham, `12213` Tinham-cômodo, `12214` Tinham-cômodo-rede geral, `12215` Tinham-cômodo-outra, `12216` Tinham-terreno/prop, `12217` Tinham-terreno/prop-rede geral, `12218` Tinham-terreno/prop-outra, `12219` Não tinham)

#### Tab 3154 | Toilet Existence & Sewage (Universo Preliminar 2010)
*   **Tier**: T1_CORE | **Res**: Censo Demográfico | **Subj**: Domicílios | **Per**: 2010 | **Locs**: 5565
*   **Vars**: `96` (U: Un, Type: c, DK: Y), `1000096` (U: %, Type: p, DK: N)
*   **Clsfs**: 
    *   `Clsf 299` (Existência de banheiro ou sanitário e esgotamento sanitário; Axis: sanitation_sewage, Cats: 6; `2942` Total, `10930` Tinham banheiro/sanitário, `10941` Tinham-rede geral/pluvial, `10942` Tinham-fossa séptica, `10962` Tinham-outro, `10963` Não tinham)

#### Tab 6326 | Occupied Households by Type
*   **Tier**: T1_CORE | **Res**: Censo Demográfico | **Subj**: Características do domicílio | **Per**: 2022 | **Locs**: 5570
*   **Vars**: `381` (Domicílios particulares permanentes ocupados; U: Un, Type: c, DK: Y), `1000381` (Domicílios ocupados - % total; U: %, Type: p, DK: N)
*   **Clsfs**: 
    *   `Clsf 125` (Tipo de domicílio; Axis: housing, Cats: 7; `2932` Total, `6815` Casa, `121264` Casa de vila ou em condomínio, `3247` Apartamento, `71975` Habitação em casa de cômodos ou cortiço, `71976` Habitação indígena sem paredes ou maloca, `71977` Estrutura residencial permanente degradada ou inacabada)

#### Tab 6803 | Water Grid Connection & Main Supply
*   **Tier**: T1_CORE | **Res**: Censo Demográfico | **Subj**: Características do domicílio | **Per**: 2022 | **Locs**: 5570
*   **Vars**: `381` (U: Un, Type: c, DK: Y), `1000381` (U: %, Type: p, DK: N)
*   **Clsfs**: 
    *   `Clsf 1821` (Ligação à rede e principal forma de abastecimento; Axis: sanitation_water, Cats: 18; `72129` Total, `72144` Possui ligação à rede e utiliza como principal, `72145` Possui ligação mas utiliza outra, `72146` [outra: Poço profundo/artesiano], `72147` [outra: Poço raso/cacimba], `72148` [outra: Fonte/nascente], `72149` [outra: Carro-pipa], `72150` [outra: Água chuva], `72151` [outra: Rios/lagos/igarapés], `72152` [outra: Outra], `72153` Não possui ligação, `72154` [não possui: Poço profundo], `72155` [não possui: Poço raso], `72156` [não possui: Fonte/nascente], `72157` [não possui: Carro-pipa], `72158` [não possui: Água chuva], `72159` [não possui: Rios/lagos], `72160` [não possui: Outra])

#### Tab 6804 | Water Piping & Main Supply (Intercrossed)
*   **Tier**: T1_CORE | **Res**: Censo Demográfico | **Subj**: Características do domicílio | **Per**: 2022 | **Locs**: 5570
*   **Vars**: `381` (U: Un, Type: c, DK: Y), `1000381` (U: %, Type: p, DK: N)
*   **Clsfs**: 
    *   `Clsf 301` (Principal forma de abastecimento de água; Axis: sanitation_water, Cats: 9; `72053` Total, `31471` Rede geral, `72054` Poço profundo/artesiano, `72055` Poço raso/cacimba, `72088` Fonte/nascente, `31472` Carro-pipa, `72089` Água de chuva, `72090` Rios/córregos/lagos, `72091` Outra)
    *   `Clsf 1817` (Existência de canalização de água; Axis: sanitation_water, Cats: 4; `72125` Total, `72126` Canalizada até dentro da habitação, `72127` Canalizada apenas no terreno, `72128` Sem água canalizada)

#### Tab 6805 | Sewage Disposal detailed categories
*   **Tier**: T1_CORE | **Res**: Censo Demográfico | **Subj**: Características do domicílio | **Per**: 2022 | **Locs**: 5570
*   **Vars**: `381` (U: Un, Type: c, DK: Y), `1000381` (U: %, Type: p, DK: N)
*   **Clsfs**: 
    *   `Clsf 11558` (Tipo de esgotamento sanitário; Axis: sanitation_sewage, Cats: 10; `46292` Total, `46290` Rede geral/pluvial/fossa ligada à rede, `72110` Rede geral ou pluvial, `72111` Fossa séptica/filtro ligada à rede, `72112` Fossa séptica/filtro não ligada, `72113` Fossa rudimentar/buraco, `92858` Vala, `72114` Rio/lago/córrego/mar, `72115` Outra forma, `92861` Não tinham banheiro nem sanitário)

#### Tab 6806 | Bathrooms Count (Exclusive) & Sewage (Crossed)
*   **Tier**: T1_CORE | **Res**: Censo Demográfico | **Subj**: Características do domicílio | **Per**: 2022 | **Locs**: 5570
*   **Vars**: `381` (U: Un, Type: c, DK: Y), `1000381` (U: %, Type: p, DK: N)
*   **Clsfs**: 
    *   `Clsf 458` (Existência e número de banheiros exclusivos; Axis: housing|sanitation_sewage, Cats: 9; `72117` Total, `12032` Tinham banheiro exclusivo, `12033` [exclusivo: 1 banheiro], `12034` [exclusivo: 2 banheiros], `12035` [exclusivo: 3 banheiros], `12036` [exclusivo: 4+ banheiros], `72118` Apenas banheiro de uso comum, `72119` Apenas sanitário ou buraco, `12046` Não tinham banheiro nem sanitário)
    *   `Clsf 11558` (Tipo de esgotamento sanitário; Axis: sanitation_sewage, Cats: 10; Same categories as Tab 6805)

#### Tab 6892 | Waste Destination Detailed (2022)
*   **Tier**: T1_CORE | **Res**: Censo Demográfico | **Subj**: Características do domicílio | **Per**: 2022 | **Locs**: 5570
*   **Vars**: `381` (U: Un, Type: c, DK: Y), `1000381` (U: %, Type: p, DK: N)
*   **Clsfs**: 
    *   `Clsf 67` (Destino do lixo; Axis: waste, Cats: 8; `10972` Total, `2520` Coletado, `72120` Coletado no domicílio por serv. limpeza, `72121` Depositado em caçamba de serv. limpeza, `72122` Queimado na propriedade, `72123` Enterrado na propriedade, `72124` Jogado terreno baldio/encosta/área pública, `1091` Outro destino)

#### Tab 3218 | Comprehensive Water, Sewage, Waste & Electricity Cross-matrix (2010)
*   **Tier**: T2_CONTEXT | **Res**: Censo Demográfico | **Subj**: Domicílios | **Per**: 2010 | **Locs**: 5565 | **High Dim**: Yes
*   **Vars**: `96` (U: Un, Type: c, DK: Y), `1000096` (U: %, Type: p, DK: N)
*   **Clsfs**: 
    *   `Clsf 61` (Abastecimento; Axis: sanitation_water, Cats: 8; `0` Total, `92853` Rede geral, `10971` Poço/nascente na prop., `121290` Poço/nascente fora prop., `121294` Rio/açude/lago/igarapé, `121296` Poço/nascente na aldeia, `121297` Poço/nascente fora aldeia, `121295` Outra)
    *   `Clsf 299` (Existência de banheiro/sanitário e esgotamento; Axis: sanitation_sewage, Cats: 8; `0` Total, `2944` Banheiro exclusivo, `9678` Banheiro exclusivo-rede geral/fossa, `2950` Banheiro exclusivo-outro, `2958` Sanitário, `9679` Sanitário-rede geral/fossa, `2964` Sanitário-outro, `10006` Sem banheiro/sanitário)
    *   `Clsf 67` (Lixo; Axis: waste, Cats: 5; `0` Total, `2520` Coletado, `92863` Coletado por limpeza, `92864` Coletado caçamba, `1091` Outro destino)
    *   `Clsf 309` (Energia elétrica; Axis: other_axis, Cats: 3; `0` Total, `3011` Tinham, `3018` Não tinham)

#### Tab 9860 | Double Universe Matrix (Total & Indigenous) by Water, Sewage, Toilets & Geography
*   **Tier**: T2_CONTEXT | **Res**: Censo Demográfico | **Subj**: Domicílios | **Per**: 2010 | 2022 | **Locs**: 5570 | **High Dim**: Yes
*   **Vars**: 
    *   `381` (Dom. part. perm. ocupados; U: Un, Type: c, DK: Y)
    *   `1000381` (Dom. ocupados - % total; U: %, Type: p, DK: N)
    *   `7088` (Dom. ocupados com pelo menos um morador indígena; U: Un, Type: c, DK: Y)
    *   `1007088` (Dom. c/ morador indígena - % total; U: %, Type: p, DK: N)
    *   `382` (Moradores em dom. ocupados; U: Px, Type: c, DK: Y)
    *   `1000382` (Moradores em dom. ocupados - % total; U: %, Type: p, DK: N)
    *   `8691` (Moradores indígenas em dom. ocupados; U: Px, Type: c, DK: Y)
    *   `1008691` (Moradores indígenas - % total; U: %, Type: p, DK: N)
*   **Clsfs**: 
    *   `Clsf 125` (Tipo de domicílio; Axis: housing, Cats: 7)
    *   `Clsf 1821` (Ligação rede & abast. principal; Axis: sanitation_water, Cats: 18)
    *   `Clsf 1817` (Canalização; Axis: sanitation_water, Cats: 4)
    *   `Clsf 458` (Banheiro & número exclusivo; Axis: housing|sanitation_sewage, Cats: 9)
    *   `Clsf 11558` (Tipo de esgotamento; Axis: sanitation_sewage, Cats: 10)
    *   `Clsf 2661` (Localização; Axis: housing, Cats: 3; `32776` Total, `12869` Em terras indígenas, `12870` Fora de terras indígenas)

---

### GROUP 5: HOUSEHOLD SIZE, CROWDING, PROPERTY STATUS & ASSETS

#### Tab 206 | Households by Situation & Room Count (1970–2010)
*   **Tier**: T2_CONTEXT | **Res**: Censo Demográfico | **Subj**: Domicílios | **Per**: 1970–2010 | **Locs**: 5565
*   **Vars**: `96` (U: Un, Type: c, DK: Y), `1000096` (U: %, Type: p, DK: N)
*   **Clsfs**: 
    *   `Clsf 1` (Situação; Axis: housing|urban_rural, Cats: 3; `0` Total, `1` Urbana, `2` Rural)
    *   `Clsf 65` (Número de cômodos; Axis: housing, Cats: 15; `0` Total, `1059` 1 cômodo, `1060` 2 cômodos, ..., `2511` 10 cômodos ou mais, `3391` Sem declaração)

#### Tab 3155 | Bathrooms of Exclusive Use Count (Universo Preliminar 2010)
*   **Tier**: T2_CONTEXT | **Res**: Censo Demográfico | **Subj**: Domicílios | **Per**: 2010 | **Locs**: 5565
*   **Vars**: `96` (U: Un, Type: c, DK: Y), `1000096` (U: %, Type: p, DK: N)
*   **Clsfs**: 
    *   `Clsf 300` (Existência/nº banheiros exclusivos; Axis: housing|sanitation_sewage, Cats: 7; `10929` Total, `2966` Tinham, `2967` 1 banheiro, `2968` 2 banheiros, `2995` 3 banheiros, `2996` 4+ banheiros, `2997` Não tinham)

#### Tab 9931 | Occupied Households by Rooms, Bathrooms & Bedrooms (2022)
*   **Tier**: T2_CONTEXT | **Res**: Censo Demográfico | **Subj**: Características do domicílio | **Per**: 2022 | **Locs**: 5570
*   **Vars**: `381` (U: Un, Type: c, DK: Y), `1000381` (U: %, Type: p, DK: N)
*   **Clsfs**: 
    *   `Clsf 65` (Número de cômodos; Axis: housing, Cats: 8; `95810` Total, `1059` 1, `1060` 2, `1061` 3, `1062` 4, `1063` 5, `112686` 6-9, `2511` 10+)
    *   `Clsf 1973` (Número de banheiros de uso exclusivo; Axis: housing|sanitation_sewage, Cats: 5; `73081` Total, `73082` Nenhum, `73083` 1, `73084` 2, `73085` 3+)
    *   `Clsf 74` (Número de dormitórios; Axis: housing, Cats: 5; `95811` Total, `3361` 1, `3362` 2, `3363` 3, `12199` 4+)

#### Tab 9933 | Bed Density, Occupancy Status & Household Type (2022)
*   **Tier**: T2_CONTEXT | **Res**: Censo Demográfico | **Subj**: Características do domicílio | **Per**: 2022 | **Locs**: 5570
*   **Vars**: `381` (U: Un, Type: c, DK: Y), `1000381` (U: %, Type: p, DK: N)
*   **Clsfs**: 
    *   `Clsf 1975` (Moradores por dormitório; Axis: housing, Cats: 5; `73086` Total, `73087` 1 morador, `73088` >1 a 2, `73089` >2 a 3, `73090` >3 moradores)
    *   `Clsf 63` (Condição de ocupação; Axis: housing|labor, Cats: 10; `95826` Total, `73554` Próprio, `73126` Próprio-pago/herança, `4343` Próprio-pagando, `1055` Alugado, `73553` Cedido, `73127` Cedido por empregador, `73128` Cedido por familiar, `73129` Cedido de outra forma, `1058` Outra)
    *   `Clsf 125` (Tipo de domicílio; Axis: housing, Cats: 7; same categories as Tab 6326)

#### Tab 3156 | Electricity Metering Status (Universo 2010)
*   **Tier**: T2_CONTEXT | **Res**: Censo Demográfico | **Subj**: Domicílios | **Per**: 2010 | **Locs**: 5565
*   **Vars**: `2619` (U: Un, Type: c, DK: Y), `1002619` (U: %, Type: p, DK: N)
*   **Clsfs**: 
    *   `Clsf 421` (Existência de medidor de energia; Axis: other_axis, Cats: 5; `10964` Total, `10965` Tinham, `10966` Tinham-uso exclusivo, `10967` Tinham-comum, `10968` Não tinham)

#### Tab 9935 | Washing Machine Ownership, Occupancy Status & Household Type
*   **Tier**: T2_CONTEXT | **Res**: Censo Demográfico | **Subj**: Características do domicílio | **Per**: 2022 | **Locs**: 5570
*   **Vars**: `381` (U: Un, Type: c, DK: Y), `1000381` (U: %, Type: p, DK: N)
*   **Clsfs**: 
    *   `Clsf 1977` (Presença de máquina de lavar roupas; Axis: other_axis, Cats: 3; `73091` Total, `73092` Sim, `73093` Não)
    *   `Clsf 63` (Condição de ocupação; Axis: housing|labor, Cats: 10)
    *   `Clsf 125` (Tipo de domicílio; Axis: housing, Cats: 7)

#### Tab 9936 | Home Internet Connectivity, Occupancy Status & Household Type
*   **Tier**: T2_CONTEXT | **Res**: Censo Demográfico | **Subj**: Características do domicílio | **Per**: 2022 | **Locs**: 5570
*   **Vars**: `381` (U: Un, Type: c, DK: Y), `1000381` (U: %, Type: p, DK: N)
*   **Clsfs**: 
    *   `Clsf 2072` (Existência de conexão domiciliar à Internet; Axis: other_axis, Cats: 3; `77584` Total, `77585` Sim, `77586` Não)
    *   `Clsf 63` (Condição de ocupação; Axis: housing|labor, Cats: 10)
    *   `Clsf 125` (Tipo de domicílio; Axis: housing, Cats: 7)

---

### GROUP 6: INCOME, LABOUR FORCE & ECONOMIC ACTIVITY

#### Tab 10295 | PC Income Indicators (Mean & Median) by Sex, Race & Age Brackets
*   **Tier**: T1_CORE | **Res**: Censo Demográfico | **Subj**: Rendimento | **Per**: 2022 | **Locs**: 5570 | **High Dim**: Yes
*   **Vars**: 
    *   `13604` (Moradores em dom. ocupados excl. pensionistas/domésticos/parentes; U: Px, Type: c, DK: Y)
    *   `1013604` (Moradores - % total; U: %, Type: p, DK: N)
    *   `13431` (Valor do rendimento nominal médio mensal domiciliar per capita; U: R$, Type: m, DK: Y)
    *   `13534` (Valor do rendimento nominal mediano mensal domiciliar per capita; U: R$, Type: md, DK: Y)
*   **Clsfs**: 
    *   `Clsf 2` (Sexo; Axis: sex, Cats: 3; `6794` Total, `4` Homens, `5` Mulheres)
    *   `Clsf 86` (Cor ou raça; Axis: race_color, Cats: 6)
    *   `Clsf 58` (Grupo de idade; Axis: age, Cats: 23; `95253` Total, cohorts: `33269` 0-17, `6744` 0-9, `118282` 10-13, `114535` 14-17, `100052` 18-24, `2793` 18-19, `1144` 20-24, `108866` 25+, detailed five-year brackets to `2503` 80+)

#### Tab 10296 | Population by PC Household Income Classes (SM), Sex & Race
*   **Tier**: T1_CORE | **Res**: Censo Demográfico | **Subj**: Rendimento | **Per**: 2022 | **Locs**: 5570
*   **Vars**: `13604` (U: Px, Type: c, DK: Y), `1013604` (U: %, Type: p, DK: N)
*   **Clsfs**: 
    *   `Clsf 2` (Sexo; Axis: sex, Cats: 3; `6794` Total)
    *   `Clsf 86` (Cor ou raça; Axis: race_color, Cats: 6)
    *   `Clsf 386` (Classes de rendimento nominal mensal domiciliar per capita; Axis: income, Cats: 12; `9680` Total, `9681` Até 1/4 SM, `9682` >1/4 a 1/2, `9683` >1/2 a 1, `9684` >1 a 2, `9685` >2 a 3, `9686` >3 a 5, `9687` >5 a 10, `9688` >10 a 15, `9689` >15 a 20, `9690` >20 SM, `9692` Sem rendimento)

#### Tab 2035 | Households & Total Household Income (Mean & Median) (2000)
* **Tier**: T1_CORE | **Res**: Censo Demográfico | **Subj**: Rendimento | **Per**: 2000 | **Locs**: 5507
* **Vars**: 
    * `96` (Domicílios; U: Un, Type: c, DK: Y)
    * `884` (Domicílios com rendimento; U: Un, Type: c, DK: Y)
    * `847` (Rendimento nominal médio mensal dos domicílios; U: R$, Type: m, DK: Y)
    * `878` (Rendimento nominal médio mensal com rendimento; U: R$, Type: m, DK: Y)
    * `848` (Rendimento nominal mediano dos domicílios; U: R$, Type: md, DK: Y)
    * `879` (Rendimento nominal mediano com rendimento; U: R$, Type: md, DK: Y)
* **Clsfs**: None

#### Tab 2426 | Households with Income (Mean & Median) by Urban/Rural (Amostra 2000–2010)
* **Tier**: T1_CORE | **Res**: Censo Demográfico | **Subj**: Rendimento | **Per**: 2000 | 2010 | **Locs**: 5507 (2000) | 5565 (2010)
* **Vars**: 
    * `884` (U: Un, Type: c, DK: Y), `1000884` (U: %, Type: p, DK: N), `878` (U: R$, Type: m, DK: Y), `879` (U: R$, Type: md, DK: Y)
* **Clsfs**: 
    * `Clsf 1` (Situação; Axis: housing|urban_rural, Cats: 3; `6795` Total, `1` Urbana, `2` Rural)

#### Tab 3563 | Households & PC Income by Urban/Rural & PC Income Classes (2010)
*   **Tier**: T1_CORE | **Res**: Censo Demográfico | **Subj**: Rendimento | **Per**: 2010 | **Locs**: 5565
*   **Vars**: 
    *   `96` (U: Un, Type: c, DK: Y), `1000096` (U: %, Type: p, DK: N)
    *   `2010` (Rendimento nominal médio mensal per capita; U: R$, Type: m, DK: Y)
    *   `2011` (Rendimento nominal mediano mensal per capita; U: R$, Type: md, DK: Y)
*   **Clsfs**: 
    *   `Clsf 1` (Situação; Axis: housing|urban_rural, Cats: 3; `6795` Total, `1` Urbana, `2` Rural)
    *   `Clsf 386` (Classes de rendimento nominal mensal domiciliar per capita; Axis: income, Cats: 11; `9680` Total, `12009` Até 1/8 SM, `12010` >1/8 a 1/4, `9682` >1/4 a 1/2, `9683` >1/2 a 1, `9684` >1 a 2, `9685` >2 a 3, `9686` >3 a 5, `9687` >5 a 10, `9691` >10 SM, `9692` Sem rendimento)

#### Tab 3571 | Households by Absolute Monthly Income Classes (Sample 2010)
*   **Tier**: T1_CORE | **Res**: Censo Demográfico | **Subj**: Rendimento | **Per**: 2010 | **Locs**: 5565
*   **Vars**: `96` (U: Un, Type: c, DK: Y), `1000096` (U: %, Type: p, DK: N)
*   **Clsfs**: 
    *   `Clsf 12065` (Classes de rendimento nominal mensal domiciliar; Axis: income, Cats: 9; `0` Total, `3089` Até 1/2 SM, `99831` >1/2 a 1, `99832` >1 a 2, `12086` >2 a 5, `99835` >5 a 10, `99922` >10 a 20, `99923` >20 SM, `99839` Sem rendimento)

#### Tab 3578 | PC Residents & PC Income by Urban/Rural & PC Income Classes (2010)
*   **Tier**: T1_CORE | **Res**: Censo Demográfico | **Subj**: Rendimento | **Per**: 2010 | **Locs**: 5565
*   **Vars**: 
    *   `2035` (Pessoas moradoras em domicílios particulares permanentes [excl. pensionistas/domésticos/parentes]; U: Px, Type: c, DK: Y)
    *   `1002035` (Pessoas moradoras - % total; U: %, Type: p, DK: N)
    *   `2012` (Rendimento nominal médio mensal domiciliar per capita; U: R$, Type: m, DK: Y)
    *   `2013` (Rendimento nominal mediano mensal domiciliar per capita; U: R$, Type: md, DK: Y)
*   **Clsfs**: 
    *   `Clsf 1` (Situação; Axis: housing|urban_rural, Cats: 3; `6795` Total)
    *   `Clsf 386` (Classes de rendimento nominal mensal domiciliar per capita; Axis: income, Cats: 11; same categories as Tab 3563)

#### Tab 3572 | Pop 10+ by Activity Status, Sex, Urban/Rural, Race, Education & Age (2010)
*   **Tier**: T1_CORE | **Res**: Censo Demográfico | **Subj**: Atividade | **Per**: 2010 | **Locs**: 5565 | **High Dim**: Yes
*   **Vars**: `140` (U: Px, Type: c, DK: Y), `1000140` (U: %, Type: p, DK: N)
*   **Clsfs**: 
    *   `Clsf 1` (Situação; Axis: housing|urban_rural, Cats: 3; `0` Total)
    *   `Clsf 2` (Sexo; Axis: sex, Cats: 3; `0` Total, `4` Homens, `5` Mulheres)
    *   `Clsf 12056` (Condição de atividade na semana de referência; Axis: age|labor, Cats: 3; `0` Total, `99566` Economicamente ativas, `99567` Não economicamente ativas)
    *   `Clsf 86` (Cor ou raça; Axis: race_color, Cats: 7; `0` Total, standard 5 groups, `2781` Sem declaração)
    *   `Clsf 1568` (Nível de instrução; Axis: education, Cats: 6; `0` Total, categories same as Tab 1554)
    *   `Clsf 58` (Grupo de idade; Axis: age, Cats: 19; `0` Total, `1142` 10-14, detailed brackets, `3244` 70+)

#### Tab 616 | Pop 10+ Activity Status by Age, Sex & Situation (1991–2010)
*   **Tier**: T1_CORE | **Res**: Censo Demográfico | **Subj**: Atividade | **Per**: 1991 | 2000 | 2010 | **Locs**: 5565 | **High Dim**: Yes
*   **Vars**: `140` (U: Px, Type: c, DK: Y), `1000140` (U: %, Type: p, DK: N)
*   **Clsfs**: 
    *   `Clsf 90` (Condição de atividade; Axis: age|labor, Cats: 3; `0` Total, `3287` Economicamente ativa, `3288` Não economicamente ativa)
    *   `Clsf 2` (Sexo; Axis: sex, Cats: 3; `0` Total, `4` Homens, `5` Mulheres)
    *   `Clsf 1` (Situação; Axis: housing|urban_rural, Cats: 3; `0` Total)
    *   `Clsf 58` (Grupo de idade; Axis: age, Cats: 25; `0` Total, standard cohorts to `2503` 80+)

#### Tab 9517 | Pop 14+ by Labour Force status, Sex, Race & Education (2022)
*   **Tier**: T1_CORE | **Res**: Censo Demográfico | **Subj**: Geral | **Per**: 2022 | **Locs**: 5570
*   **Vars**: `1641` (Pessoas de 14 anos ou mais de idade; U: Px, Type: c, DK: Y), `1001641` (Pessoas de 14+ - % total; U: %, Type: p, DK: N)
*   **Clsfs**: 
    *   `Clsf 629` (Condição em relação à força de trabalho; Axis: labor, Cats: 5; `32385` Total, `32386` Força de trabalho, `32387` Força-ocupada, `32446` Força-desocupada, `32447` Fora da força de trabalho)
    *   `Clsf 2` (Sexo; Axis: sex, Cats: 3; `6794` Total, `4` Homens, `5` Mulheres)
    *   `Clsf 86` (Cor ou raça; Axis: race_color, Cats: 6; `95251` Total)
    *   `Clsf 1568` (Nível de instrução; Axis: education, Cats: 5; `120704` Total, standard categories same as Tab 10141)

#### Tab 10280 | Mean/Median Income of Occupied Pop 14+ by Sex & Occupational Position
*   **Tier**: T2_CONTEXT | **Res**: Censo Demográfico | **Subj**: Rendimento | **Per**: 2022 | **Locs**: 5570
*   **Vars**: 
    *   `13535` (Pessoas de 14 anos ou mais, ocupadas e com rendimento; U: Px, Type: c, DK: Y)
    *   `1013535` (Pessoas 14+ ocupadas com rendimento - % total; U: %, Type: p, DK: N)
    *   `13536` (Rendimento nominal médio de todos os trabalhos; U: R$, Type: m, DK: Y)
    *   `13537` (Rendimento nominal mediano de todos os trabalhos; U: R$, Type: md, DK: Y)
*   **Clsfs**: 
    *   `Clsf 2` (Sexo; Axis: sex, Cats: 3; `6794` Total)
    *   `Clsf 11913` (Posição na ocupação no trabalho principal; Axis: labor, Cats: 22; `96165` Total, `31721` Empregado setor privado, `31722` Privado-com carteira, `31723` Privado-sem carteira, `31724` Trabalhador doméstico, `79367` Doméstico-com carteira, `79368` Doméstico-sem carteira, `79369` Militar, `31727` Empregado setor público, `79370` Público-estatutário, `79371` Público-celetista, `79372` Público-com carteira, `79373` Público-sem carteira, `79374` Estatal, `79375` Estatal-com carteira, `79376` Estatal-sem carteira, `79377` Empregador, `45934` Empregador-CNPJ, `45935` Empregador-sem CNPJ, `79378` Conta própria, `45936` Conta própria-CNPJ, `45937` Conta própria-sem CNPJ)

#### Tab 3579 | Pop 10+ Occupation status by Urban/Rural, Sex, Schooling & Age (2010)
*   **Tier**: T2_CONTEXT | **Res**: Censo Demográfico | **Subj**: Atividade | **Per**: 2010 | **Locs**: 5565 | **High Dim**: Yes
*   **Vars**: `140` (U: Px, Type: c, DK: Y), `1000140` (U: %, Type: p, DK: N)
*   **Clsfs**: 
    *   `Clsf 12085` (Situação de ocupação na semana de referência; Axis: labor, Cats: 3; `0` Total, `100440` Ocupadas, `100441` Não ocupadas)
    *   `Clsf 1` (Situação; Axis: housing|urban_rural, Cats: 3; `0` Total)
    *   `Clsf 2` (Sexo; Axis: sex, Cats: 3; `0` Total)
    *   `Clsf 204` (Frequência à escola; Axis: other_axis, Cats: 3; `0` Total, `100462` Frequentavam, `100463` Não frequentavam)
    *   `Clsf 58` (Grupo de idade; Axis: age, Cats: 16; `0` Total, standard cohorts up to `3302` 60+)

#### Tab 10289 | Work Income Mass & Mean/Median Income 14+ by Sex & Race
*   **Tier**: T2_CONTEXT | **Res**: Censo Demográfico | **Subj**: Rendimento | **Per**: 2022 | **Locs**: 5570
*   **Vars**: 
    *   `13424` (Massa de rendimento nominal mensal de todos os trabalhos; U: R$, Type: m, DK: Y)
    *   `13536` (Rendimento nominal médio de todos os trabalhos; U: R$, Type: m, DK: Y)
    *   `13537` (Rendimento nominal mediano de todos os trabalhos; U: R$, Type: md, DK: Y)
*   **Clsfs**: 
    *   `Clsf 2` (Sexo; Axis: sex, Cats: 3; `6794` Total)
    *   `Clsf 86` (Cor ou raça; Axis: race_color, Cats: 6; `95251` Total)

#### Tab 10292 | Occupied Pop 14+ Distribution by Sex & Monthly Work Income Classes (SM)
*   **Tier**: T2_CONTEXT | **Res**: Censo Demográfico | **Subj**: Rendimento | **Per**: 2022 | **Locs**: 5570
*   **Vars**: 
    *   `4090` (Pessoas de 14 anos ou mais, ocupadas na semana de referência; U: Px, Type: c, DK: Y)
    *   `1004090` (Pessoas 14+ ocupadas - % total; U: %, Type: p, DK: N)
    *   `13600` (Distribuição das pessoas de 14 anos ou mais, ocupadas; U: %, Type: p, DK: N)
*   **Clsfs**: 
    *   `Clsf 2` (Sexo; Axis: sex, Cats: 3; `6794` Total)
    *   `Clsf 11915` (Classes de rendimento nominal mensal de todos os trabalhos; Axis: income|labor, Cats: 12; `96177` Total, `99822` Até 1/4 SM, `99823` >1/4 a 1/2, `99824` >1/2 a 1, `96179` >1 a 2, `96180` >2 a 3, `96181` >3 a 5, `96182` >5 a 10, `99825` >10 a 15, `99828` >15 a 20, `96184` >20 SM, `96185` Sem rendimento)

#### Tab 10300 | PC HH Residents (14+) by Labor Force Status & PC Income Classes (SM)
*   **Tier**: T2_CONTEXT | **Res**: Censo Demográfico | **Subj**: Rendimento | **Per**: 2022 | **Locs**: 5570
*   **Vars**: `13432` (Moradores 14+ [excl. pensionistas/domésticos/parentes]; U: Px, Type: c, DK: Y), `1013432` (Moradores 14+ - % total; U: %, Type: p, DK: N)
*   **Clsfs**: 
    *   `Clsf 629` (Condição em relação à força de trabalho; Axis: labor, Cats: 5; `32385` Total)
    *   `Clsf 386` (Classes de rendimento nominal mensal domiciliar per capita; Axis: income, Cats: 12; `9680` Total, standard categories same as Tab 10296)

#### Tab 10381 | Occupied Pop 10+ Distribution by Race & Monthly Work Income Classes (SM)
*   **Tier**: T2_CONTEXT | **Res**: Censo Demográfico | **Subj**: Rendimento | **Per**: 2022 | **Locs**: 5570
*   **Vars**: 
    *   `696` (Pessoas de 10 anos ou mais, ocupadas; U: Px, Type: c, DK: Y)
    *   `1000696` (Pessoas 10+ ocupadas - % total; U: %, Type: p, DK: N)
    *   `13423` (Distribuição das pessoas de 10 anos ou mais, ocupadas; U: %, Type: p, DK: N)
*   **Clsfs**: 
    *   `Clsf 86` (Cor ou raça; Axis: race_color, Cats: 6; `95251` Total)
    *   `Clsf 11915` (Classes de rendimento nominal mensal de todos os trabalhos; Axis: income|labor, Cats: 12; `96177` Total, standard categories same as Tab 10292)

---

### GROUP 7: PHYSICAL & COGNITIVE PERMANENT DISABILITIES

#### Tab 1495 | Population by Detailed Permanent Disability Type (Amostra 2010)
*   **Tier**: T2_CONTEXT | **Res**: Censo Demográfico | **Subj**: Deficiência | **Per**: 2010 | **Locs**: 5565
*   **Vars**: `93` (U: Px, Type: c, DK: Y), `1000093` (U: %, Type: p, DK: N)
*   **Clsfs**: 
    *   `Clsf 134` (Tipo de deficiência permanente; Axis: disability, Cats: 14; `0` Total, `7815` Pelo menos uma, `12152` Def. visual-não consegue de modo algum, `12153` Def. visual-grande dificuldade, `12154` Def. visual-alguma, `12155` Def. auditiva-não consegue de modo algum, `12156` Def. auditiva-grande dificuldade, `12157` Def. auditiva-alguma, `12158` Def. motora-não consegue de modo algum, `12159` Def. motora-grande dificuldade, `12160` Def. motora-alguma, `12161` Mental/intelectual, `95295` Nenhuma, `2791` Sem declaração)

#### Tab 2111 | Population by Disability, Situation, Sex & Age (2000)
* **Tier**: T2_CONTEXT | **Res**: Censo Demográfico | **Subj**: Deficiência | **Per**: 2000 | **Locs**: 5507 | **High Dim**: Yes
* **Vars**: `93` (U: Px, Type: c, DK: Y), `1000093` (U: %, Type: p, DK: N)
* **Clsfs**: 
    * `Clsf 134` (Tipo de deficiência permanente; Axis: disability, Cats: 10; `0` Total, `95747` Pelo menos uma, `95279` Def. mental permanente, `95280` Def. física-tetra/para/hemiplegia, `95281` Def. física-falta de membro, `96031` Def. visual, `96032` Def. auditiva, `96033` Def. motora, `95295` Nenhuma, `2791` Sem declaração)
    * `Clsf 1` (Situação; Axis: housing|urban_rural, Cats: 3; `0` Total, `1` Urbana, `2` Rural)
    * `Clsf 2` (Sexo; Axis: sex, Cats: 3; `0` Total, `4` Homens, `5` Mulheres)
    * `Clsf 58` (Grupo de idade; Axis: age, Cats: 28)

#### Tab 3425 | Population by Disability, Situation, Sex & Age (Amostra 2010)
*   **Tier**: T2_CONTEXT | **Res**: Censo Demográfico | **Subj**: Deficiência | **Per**: 2010 | **Locs**: 5565 | **High Dim**: Yes
*   **Vars**: `93` (U: Px, Type: c, DK: Y), `1000093` (U: %, Type: p, DK: N)
*   **Clsfs**: 
    *   `Clsf 134` (Tipo de deficiência permanente; Axis: disability, Cats: 14; same categories as Tab 1495)
    *   `Clsf 1` (Situação; Axis: housing|urban_rural, Cats: 3; `0` Total)
    *   `Clsf 2` (Sexo; Axis: sex, Cats: 3; `0` Total)
    *   `Clsf 58` (Grupo de idade; Axis: age, Cats: 24; cohorts: `100402` 0-14, single cohorts to `2503` 80+)

#### Tab 3426 | Population by Disability, Sex & Color/Race (Amostra 2010)
*   **Tier**: T2_CONTEXT | **Res**: Censo Demográfico | **Subj**: Deficiência | **Per**: 2010 | **Locs**: 5565
*   **Vars**: `93` (U: Px, Type: c, DK: Y), `1000093` (U: %, Type: p, DK: N)
*   **Clsfs**: 
    *   `Clsf 134` (Tipo de deficiência permanente; Axis: disability, Cats: 14; same categories as Tab 1495)
    *   `Clsf 2` (Sexo; Axis: sex, Cats: 3; `0` Total)
    *   `Clsf 86` (Cor ou raça; Axis: race_color, Cats: 7; `0` Total, standard 5 groups, `2781` Sem declaração)

---

### GROUP 8: FERTILITY, MOTHERHOOD & CHILDREN BORN ALIVE

#### Tab 10075 | Mothers (12+) by Children Count, Education & Age Cohorts
*   **Tier**: T2_CONTEXT | **Res**: Censo Demográfico | **Subj**: Fecundidade | **Per**: 2010 | 2022 | **Locs**: 5570 | **High Dim**: Yes
*   **Vars**: `13314` (Mulheres de 12 anos ou mais que tiveram filhos nascidos vivos; U: Px, Type: c, DK: Y), `1013314` (Mulheres de 12+ com filhos - % total; U: %, Type: p, DK: N)
*   **Clsfs**: 
    *   `Clsf 12291` (Número de filhos tidos nascidos vivos; Axis: other_axis, Cats: 7; `58895` Total, `105103` 1 filho, `105104` 2, `105105` 3, `105106` 4, `105107` 5, `105108` 6 filhos ou mais)
    *   `Clsf 1568` (Nível de instrução; Axis: education, Cats: 6; `120704` Total, standard categories same as Tab 1554)
    *   `Clsf 12232` (Grupos de idade das mulheres; Axis: age, Cats: 16; `58896` Total, `58897` 12-14, `104541` 15-19, `104542` 15-17, `104543` 18/19, `104544` 20-24, `104545` 25-29, `104546` 30-34, `104547` 35-39, `104548` 40-44, `104549` 45-49, `105109` 50-54, `105110` 55-59, `105111` 60-64, `105112` 65-69, `105113` 70+)

#### Tab 10076 | Mothers (12+) by Children Count, Color/Race & Age Cohorts
*   **Tier**: T2_CONTEXT | **Res**: Censo Demográfico | **Subj**: Fecundidade | **Per**: 2010 | 2022 | **Locs**: 5570 | **High Dim**: Yes
*   **Vars**: `13314` (U: Px, Type: c, DK: Y), `1013314` (U: %, Type: p, DK: N)
*   **Clsfs**: 
    *   `Clsf 12291` (Número de filhos; Axis: other_axis, Cats: 7; `58895` Total, `105103` to `105108` 6+)
    *   `Clsf 12293` (Cor ou raça das mulheres; Axis: race_color, Cats: 7; `58898` Total, `105167` Branca, `105168` Preta, `105169` Amarela, `105170` Parda, `105171` Indígena, `105172` Sem declaração)
    *   `Clsf 12232` (Grupos de idade das mulheres; Axis: age, Cats: 16; `58896` Total, same categories as Tab 10075)

#### Tab 10077 | Mothers (12+) by Age, Education & Color/Race (2010–2022)
*   **Tier**: T2_CONTEXT | **Res**: Censo Demográfico | **Subj**: Fecundidade | **Per**: 2010 | 2022 | **Locs**: 5570 | **High Dim**: Yes
*   **Vars**: `13314` (U: Px, Type: c, DK: Y), `1013314` (U: %, Type: p, DK: N)
*   **Clsfs**: 
    *   `Clsf 12232` (Grupos de idade das mulheres; Axis: age, Cats: 16)
    *   `Clsf 1568` (Nível de instrução; Axis: education, Cats: 6)
    *   `Clsf 12293` (Cor ou raça das mulheres; Axis: race_color, Cats: 7)

#### Tab 2573 | Mothers (10+) by Children Count, Situation & Age (Histórica)
*   **Tier**: T2_CONTEXT | **Res**: Censo Demográfico | **Subj**: Fecundidade | **Per**: 2000 | 2010 | **Locs**: 5565
*   **Vars**: `1349` (Mulheres de 10 anos ou mais de idade que tiveram filhos; U: Px, Type: c, DK: Y), `1001349` (Mulheres 10+ - % total; U: %, Type: p, DK: N)
*   **Clsfs**: 
    *   `Clsf 12291` (Número de filhos; Axis: other_axis, Cats: 7)
    *   `Clsf 1` (Situação; Axis: housing|urban_rural, Cats: 3; `0` Total, `1` Urbana, `2` Rural)
    *   `Clsf 12232` (Grupos de idade das mulheres; Axis: age, Cats: 16; `0` Total, cohorts `104540` 10-14, up to `105113` 70+)

---

### GROUP 9: INDIGENOUS & QUILOMBOLA SPECIAL POPULATIONS

#### Tab 10089 | Residents & Quilombolas by Sex, Age, Territory Status & Situation
*   **Tier**: T2_CONTEXT | **Res**: Censo Demográfico | **Subj**: População quilombola | **Per**: 2022 | **Locs**: 5570 | **High Dim**: Yes
*   **Vars**: 
    *   `93` (População residente; U: Px, Type: c, DK: Y)
    *   `1000093` (População - % total; U: %, Type: p, DK: N)
    *   `4709` (Pessoas quilombolas; U: Px, Type: c, DK: Y)
    *   `1004709` (Pessoas quilombolas - % total; U: %, Type: p, DK: N)
    *   `4728` (Percentual de pessoas quilombolas na população residente; U: %, Type: p, DK: N)
*   **Clsfs**: 
    *   `Clsf 2` (Sexo; Axis: sex, Cats: 3; `6794` Total)
    *   `Clsf 58` (Grupo de idade; Axis: age, Cats: 22; standard census five-year cohorts)
    *   `Clsf 2661` (Localização do domicílio; Axis: housing, Cats: 3; `32776` Total, `60027` Em territórios quilombolas, `60028` Fora de territórios quilombolas)
    *   `Clsf 1` (Situação; Axis: housing|urban_rural, Cats: 3; `6795` Total)

#### Tab 8176 | Quilombola Population by Location, Age Brackets & Sex (Single-years)
*   **Tier**: T2_CONTEXT | **Res**: Censo Demográfico | **Subj**: População quilombola | **Per**: 2022 | **Locs**: 5570 | **High Dim**: Yes
*   **Vars**: `4709` (U: Px, Type: c, DK: Y), `1004709` (U: %, Type: p, DK: N)
*   **Clsfs**: 
    *   `Clsf 287` (Idade; Axis: age, Cats: 134; standard detailed single years / months)
    *   `Clsf 2` (Sexo; Axis: sex, Cats: 3; `6794` Total)
    *   `Clsf 2661` (Localização; Axis: housing, Cats: 3; same as Tab 10089)

#### Tab 9578 | Quilombola Presence Share by Territory Location
*   **Tier**: T2_CONTEXT | **Res**: Censo Demográfico | **Subj**: População quilombola | **Per**: 2022 | **Locs**: 5570
*   **Vars**: `4709` (U: Px, Type: c, DK: Y), `1004709` (U: %, Type: p, DK: N), `93` (U: Px, Type: c, DK: Y), `4728` (U: %, Type: p, DK: N)
*   **Clsfs**: 
    *   `Clsf 2661` (Localização; Axis: housing, Cats: 3; same as Tab 10089)

#### Tab 9718 | Indigenous Population & Declaration category by Territory Status (2022)
*   **Tier**: T2_CONTEXT | **Res**: Censo Demográfico | **Subj**: População indígena | **Per**: 2022 | **Locs**: 5570
*   **Vars**: 
    *   `350` (Pessoas indígenas; U: Px, Type: c, DK: Y)
    *   `1000350` (Pessoas indígenas - % total; U: %, Type: p, DK: N)
    *   `93` (População residente; U: Px, Type: c, DK: Y)
    *   `1000093` (População - % total; U: %, Type: p, DK: N)
    *   `4727` (Percentual de pessoas indígenas na população residente; U: %, Type: p, DK: N)
*   **Clsfs**: 
    *   `Clsf 1714` (Quesito de declaração indígena; Axis: indigenous_quilombola|race_color, Cats: 4; `60024` Total, `60025` Cor ou raça indígena, `60026` Se considera indígena, `60029` Não se declarou indígena)
    *   `Clsf 2661` (Localização; Axis: housing, Cats: 3; `32776` Total, `12869` Em terras indígenas, `12870` Fora de terras indígenas)

#### Tab 9723 | Population inside officially designated Quilombola Territories
*   **Tier**: T2_CONTEXT | **Res**: Censo Demográfico | **Subj**: Territórios quilombolas | **Per**: 2022 | **Locs**: 5570
*   **Vars**: 
    *   `6559` (Pessoas residentes em territórios quilombolas; U: Px, Type: c, DK: Y)
    *   `7079` (Pessoas quilombolas residentes em territórios quilombolas; U: Px, Type: c, DK: Y)
    *   `7080` (Percentual de quilombolas no total de residentes; U: %, Type: p, DK: N)
*   **Clsfs**: None (Aggregated by official Quilombola Territories names)

#### Tab 9764 | Residents inside designated Indigenous Lands by Age, Sex & Declaration Category
*   **Tier**: T2_CONTEXT | **Res**: Censo Demográfico | **Subj**: Terras indígenas | **Per**: 2010 | 2022 | **Locs**: 5570 | **High Dim**: Yes
*   **Vars**: 
    *   `576` (Pessoas residentes em terras indígenas; U: Px, Type: c, DK: Y)
    *   `1000576` (Pessoas residentes - % total; U: %, Type: p, DK: N)
    *   `1432` (Pessoas indígenas residentes em terras indígenas; U: Px, Type: c, DK: Y)
    *   `1001432` (Indígenas - % total; U: %, Type: p, DK: N)
    *   `8824` (Percentual de indígenas no total de residentes em terras indígenas; U: %, Type: p, DK: N)
*   **Clsfs**: 
    *   `Clsf 287` (Idade; Axis: age, Cats: 134)
    *   `Clsf 2` (Sexo; Axis: sex, Cats: 3; `6794` Total)
    *   `Clsf 1714` (Quesito de declaração; Axis: indigenous_quilombola|race_color, Cats: 4; same as Tab 9718)

#### Tab 9765 | Residents inside Quilombola Territories by Age & Sex
*   **Tier**: T2_CONTEXT | **Res**: Censo Demográfico | **Subj**: Territórios quilombolas | **Per**: 2022 | **Locs**: 5570
*   **Vars**: 
    *   `6559` (U: Px, Type: c, DK: Y), `1006559` (U: %, Type: p, DK: N)
    *   `7079` (U: Px, Type: c, DK: Y), `1007079` (U: %, Type: p, DK: N)
    *   `7080` (U: %, Type: p, DK: N)
*   **Clsfs**: 
    *   `Clsf 287` (Idade; Axis: age, Cats: 134)
    *   `Clsf 2` (Sexo; Axis: sex, Cats: 3; `6794` Total)

---

### GROUP 10: ECONOMIC SECTOR DETAIL (CNAE 2.0)

#### Tab 6450 | Local Units, Employees & Wages by CNAE 2.0 (Série encerrada 2021)
* **Tier**: T3_OPTIONAL | **Res**: Cadastro Central de Empresas | **Subj**: Unidades locais | **Per**: 2006–2021 | **Locs**: 5570 | **High Dim**: Yes
* **Vars**: 
    * `706` (Unidades locais; U: Un, Type: c, DK: Y)
    * `1000706` (Unidades locais - % total; U: %, Type: p, DK: N)
    * `707` (Pessoal ocupado total; U: Px, Type: c, DK: Y)
    * `1000707` (Pessoal ocupado total - % total; U: %, Type: p, DK: N)
    * `708` (Pessoal ocupado assalariado; U: Px, Type: c, DK: Y)
    * `1000708` (Pessoal ocupado assalariado - % total; U: %, Type: p, DK: N)
    * `662` (Salários e outras remunerações; U: milR$, Type: m, DK: Y)
    * `1000662` (Salários - % total; U: %, Type: p, DK: N)
* **Clsfs**: 
    * `Clsf 12762` (CNAE 2.0; Axis: cnae_sector, Cats: 1067; `117897` Total, sections `A` to `U` and sub-divisions, e.g. `116830` Seção A, `116831` Divisão 01, `116832` Grupo 01.1, `116833` Classe 01.11-3 Cultivo cereais, to `117895` Classe 99.00-8)

#### Tab 9528 | Local Units, Employees & Wages by CNAE 2.0 (Ativa)
* **Tier**: T3_OPTIONAL | **Res**: Cadastro Central de Empresas | **Subj**: Unidades locais | **Per**: 2022–2023 | **Locs**: 5570 | **High Dim**: Yes
* **Vars**: Same as Tab 6450 (`706`, `1000706`, `707`, `1000707`, `708`, `1000708`, `662`, `1000662`)
* **Clsfs**: 
    * `Clsf 12762` (CNAE 2.0; Axis: cnae_sector, Cats: 1067)

---

### GROUP 11: MUNICIPAL INFRASTRUCTURE & ENVIRONMENTAL SANITATION (PNSB)

#### Tab 1238 | Basic Sanitation Services Coverage by Type (2000–2008)
*   **Tier**: T3_OPTIONAL | **Res**: Pesquisa Nacional de Saneamento Básico (PNSB) | **Subj**: Gestão municipal | **Per**: 2000 | 2008 | **Locs**: 5564
*   **Vars**: 
    *   `2613` (Municípios com algum serviço de saneamento básico; U: Un, Type: c, DK: Y)
    *   `1002613` (Municípios com serviço - % total; U: %, Type: p, DK: N)
*   **Clsfs**: 
    *   `Clsf 11969` (Tipo de serviço de saneamento básico; Axis: other_axis, Cats: 6; `121191` Total geral de municípios, `98589` Total com algum serviço, `98366` Rede geral de distribuição de água, `98367` Rede coletora de esgoto, `121192` Manejo de resíduos sólidos, `121194` Manejo de águas pluviais)

#### Tab 1241 | Municipalities with Waste Management by Selective Collection Status
*   **Tier**: T3_OPTIONAL | **Res**: PNSB | **Subj**: Gestão municipal | **Per**: 2000 | 2008 | **Locs**: 5564
*   **Vars**: `2594` (Municípios com manejo de resíduos sólidos; U: Un, Type: c, DK: Y), `1002594` (Municípios com manejo - % total; U: %, Type: p, DK: N)
*   **Clsfs**: 
    *   `Clsf 11461` (Situação da coleta seletiva no município; Axis: other_axis, Cats: 7; `121204` Total geral de municípios, `103915` Total, `91763` Em atividade, `118112` Projeto-piloto, `91764` Interrompida, `91765` Não há coleta seletiva, `92950` Sem declaração)

#### Tab 1242 | Municipalities with Waste Management by Service Type
*   **Tier**: T3_OPTIONAL | **Res**: PNSB | **Subj**: Gestão municipal | **Per**: 2008 | **Locs**: 5564
*   **Vars**: `2594` (U: Un, Type: c, DK: Y), `1002594` (U: %, Type: p, DK: N)
*   **Clsfs**: 
    *   `Clsf 2646` (Natureza dos serviços; Axis: other_axis, Cats: 9; `103366` Total geral de municípios, `121205` Total, `121206` Coleta domiciliar regular, `103370` Coleta seletiva, `99987` Limpeza pública, `121207` Triagem, `121208` Coleta resíduos especiais, `539` Tratamento, `540` Disposição no solo)

#### Tab 1365 | Active and Residential Water Connections (PNSB)
*   **Tier**: T3_OPTIONAL | **Res**: PNSB | **Subj**: Abastecimento de água | **Per**: 2008 | 2017 | **Locs**: 5533
*   **Vars**: 
    *   `722` (Número de economias abastecidas; U: Un, Type: c, DK: Y)
    *   `339` (Número de economias ativas abastecidas; U: Un, Type: c, DK: Y)
    *   `340` (Número de economias ativas residenciais; U: Un, Type: c, DK: Y)
    *   `341` (Número de outras economias ativas; U: Un, Type: c, DK: Y)
*   **Clsfs**: None

#### Tab 7460 | Municipalities with Active Water Supply Grid by Operation Status
*   **Tier**: T3_OPTIONAL | **Res**: PNSB | **Subj**: Abastecimento de água | **Per**: 2017 | **Locs**: 5570
*   **Vars**: `2598` (Municípios com abastecimento por rede geral; U: Un, Type: c, DK: Y)
*   **Clsfs**: 
    *   `Clsf 921` (Condição de funcionamento do serviço; Axis: other_axis, Cats: 4; `48278` Total, `48279` Em funcionamento, `48280` Em implantação, `48281` Paralisado)

#### Tab 7461 | Municipalities with Active Sewage Network by Operation Status
*   **Tier**: T3_OPTIONAL | **Res**: PNSB | **Subj**: Esgotamento sanitário | **Per**: 2017 | **Locs**: 5570
*   **Vars**: `2601` (Municípios com serviço de esgotamento sanitário; U: Un, Type: c, DK: Y)
*   **Clsfs**: 
    *   `Clsf 921` (Condição de funcionamento do serviço; Axis: other_axis, Cats: 3; `48278` Total, `48279` Em funcionamento, `48280` Em implantação)

#### Tab 7483 | Operating Sewage Network Extension by Collector Type
*   **Tier**: T3_OPTIONAL | **Res**: PNSB | **Subj**: Esgotamento sanitário | **Per**: 2017 | **Locs**: 5570
*   **Vars**: 
    *   `10656` (Municípios com esgotamento por rede em funcionamento; U: Un, Type: c, DK: Y)
    *   `10757` (Extensão da rede coletora de esgoto; U: Quilômetros, Type: o, DK: Y)
*   **Clsfs**: 
    *   `Clsf 12005` (Tipo de rede coletora de esgoto sanitário; Axis: sanitation_sewage, Cats: 3; `49181` Unitária convencional, `2074` Separadora convencional, `98720` Condominial)

#### Tab 7500 | Operating Water Distribution Network Extension
*   **Tier**: T3_OPTIONAL | **Res**: PNSB | **Subj**: Abastecimento de água | **Per**: 2017 | **Locs**: 5517
*   **Vars**: 
    *   `10485` (Municípios com serviço de rede de distribuição em funcionamento; U: Un, Type: c, DK: Y)
    *   `724` (Extensão da rede distribuidora de água; U: Quilômetros, Type: o, DK: Y)
*   **Clsfs**: None

#### Tab 354 | Sanitation-Associated Disease Outbreaks (Municipal counts)
*   **Tier**: T3_OPTIONAL | **Res**: PNSB | **Subj**: Gestão municipal | **Per**: 2008 | **Locs**: 5564
*   **Vars**: `2597` (Municípios com ocorrência de doenças; U: Un, Type: c, DK: Y), `1002597` (Municípios com doença - % total; U: %, Type: p, DK: N)
*   **Clsfs**: 
    *   `Clsf 12963` (Tipo de doença; Axis: other_axis, Cats: 15; `120931` Total, `120937` Dengue, `120930` Total geral de municípios, `120932` Diarréia, `120933` Leptospirose, `120934` Verminoses, `120935` Cólera, `120936` Difteria, `120938` Tifo, `120939` Malária, `120940` Hepatite, `120941` Febre amarela, `120942` Dermatite, `120943` Doença respiratória, `120944` Outra)

---

### GROUP 12: CIVIL SOCIETY, PRIVATE FOUNDATIONS & NONPROFITS (FASFIL)

#### Tab 3606 | Private Nonprofits Indicators by Sex & Higher Education Status (2010 Methodology)
*   **Tier**: T4_LOW_PRIORITY | **Res**: FASFIL | **Subj**: Entidades sem fins lucrativos | **Per**: 2006 | 2008 | 2010 | **Locs**: 5563
*   **Vars**: 
    *   `2122` (Unidades locais das entidades; U: Un, Type: c, DK: Y)
    *   `2123` (Pessoal ocupado assalariado em 31/12; U: Px, Type: c, DK: Y)
    *   `1541`/`1670`/`1543` (Pessoal ocupado: Masculino / Feminino / Nível Superior; U: Px, Type: c, DK: Y)
    *   `2124` (Salários e outras remunerações; U: milR$, Type: m, DK: Y)
    *   `1544`/`1671`/`1545` (Salários: Masculino / Feminino / Nível Superior; U: milR$, Type: m, DK: Y)
    *   `2125` (Salário médio mensal; U: SM, Type: m, DK: Y)
    *   `1546`/`1672`/`1548` (Salário médio: Masculino / Feminino / Nível Superior; U: SM, Type: m, DK: Y)
*   **Clsfs**: None

#### Tab 3612 | Private Nonprofits General Indicators (2006 Methodology)
*   **Tier**: T4_LOW_PRIORITY | **Res**: FASFIL | **Subj**: Entidades sem fins lucrativos | **Per**: 2006 | **Locs**: 5555
*   **Vars**: Core indicators only (`2122` Unidades locais, `2123` Pessoal ocupado, `2124` Salários, `2125` Salário médio mensal)
*   **Clsfs**: None

#### Tab 6916 | Private Foundations & Nonprofits (FASFIL) by Detailed Activity Classification
*   **Tier**: T4_LOW_PRIORITY | **Res**: FASFIL | **Subj**: FASFIL | **Per**: 2010 | 2013 | 2016 | **Locs**: 5559 | **High Dim**: Yes
*   **Vars**: 
    *   `2129` (Unidades locais; U: Un, Type: c, DK: Y), `1002129` (U: %, Type: p, DK: N)
    *   `2130` (Pessoal ocupado; U: Px, Type: c, DK: Y), `1002130` (U: %, Type: p, DK: N)
    *   `1550` (Pessoal Masc; U: Px, Type: c, DK: Y), `1001550` (U: %, Type: p, DK: N)
    *   `1673` (Pessoal Fem; U: Px, Type: c, DK: Y), `1001673` (U: %, Type: p, DK: N)
    *   `1551` (Pessoal Sup; U: Px, Type: c, DK: Y), `1001551` (U: %, Type: p, DK: N)
    *   `9945` (Pessoal sem nível superior; U: Px, Type: c, DK: Y), `1009945` (U: %, Type: p, DK: N)
    *   `2131` (Salários; U: milR$, Type: m, DK: Y), `1002131` (U: %, Type: p, DK: N)
    *   `1666`/`1674`/`1667`/`9952` (Salários: Masc / Fem / Sup / Sem Sup; U: milR$, Type: m, DK: Y), `1001666`/`1001674`/`1001667`/`1009952` (Shares; U: %, Type: p, DK: N)
*   **Clsfs**: 
    *   `Clsf 510` (Classificação das fundações privadas; Axis: other_axis, Cats: 36; `128557` Total, `128522` Habitação, `128524` Saúde, `128525` Hospitais, `128527` Cultura/recreação, `128530` Educação, `128538` Assistência social, `128540` Religião, `128542` Associações patronais/profissionais, `128546` Meio ambiente, `128548` Defesa de direitos, `128555` Outras)

#### Tab 6917 | All Nonprofits & Foundations (FASFIL Broad) by Activity Classification
*   **Tier**: T4_LOW_PRIORITY | **Res**: FASFIL | **Subj**: Entidades | **Per**: 2010 | 2013 | 2016 | **Locs**: 5570 | **High Dim**: Yes
*   **Vars**: Same as Tab 6916 but using broad definitions: `2122`/`1002122` (Units), `2123`/`1002123` (Employees), `1541`/`1001541` (Masc), `1670`/`1001670` (Fem), `1543`/`1001543` (Sup), `9954`/`1009954` (Sem Sup), `2124`/`1002124` (Salários), `1544`/`1001544` (Salários Masc), `1671`/`1001671` (Salários Fem), `1545`/`1001545` (Salários Sup), `9960`/`1009960` (Salários Sem Sup).
*   **Clsfs**: 
    *   `Clsf 12665` (Classificação broad; Axis: age|institutional, Cats: 46; `114484` Total, standard FASFIL types plus `114431` Partidos políticos, `114432` Sindicatos/federações, `114446` Condomínios, `114447` Cartórios, `114448` Sistema S, `114451` Conselhos municipais, `114452` Cemitérios/funerárias)

---

### GROUP 13: VITAL STATISTICS & CIVIL REGISTRY (REGISTRO CIVIL)

#### Tab 197 | Live Births by Mother's Age & Infant's Sex (1984–2002)
*   **Tier**: T4_LOW_PRIORITY | **Res**: Estatísticas do Registro Civil | **Subj**: Nascidos vivos | **Per**: 1984–2002 | **Locs**: 5079
*   **Vars**: `218` (Nascidos vivos ocorridos no ano; U: Px, Type: c, DK: Y), `1000218` (Live births - % total; U: %, Type: p, DK: N)
*   **Clsfs**: 
    *   `Clsf 239` (Grupos de idade da mãe; Axis: age, Cats: 11; `0` Total, `5355` <15 anos, `5356` 15-19, `5357` 20-24, `5358` 25-29, `5359` 30-34, `5360` 35-39, `5361` 40-44, `5362` 45-49, `5363` 50+, `5364` Idade ignorada)
    *   `Clsf 2` (Sexo; Axis: sex, Cats: 3; `0` Total, `4` Homens, `5` Mulheres)

#### Tab 2609 | Live Births by Birth Year, Mother's Age, Infant's Sex & Residence Location (Active)
*   **Tier**: T4_LOW_PRIORITY | **Res**: Estatísticas do Registro Civil | **Subj**: Nascidos vivos | **Per**: 2003–2024 | **Locs**: 5570 | **High Dim**: Yes
*   **Vars**: `217` (Nascidos vivos registrados no ano; U: Px, Type: c, DK: Y), `1000217` (Registered births - % total; U: %, Type: p, DK: N)
*   **Clsfs**: 
    *   `Clsf 232` (Ano de nascimento; Axis: other_axis, Cats: 62; `0` Total, single years `2024` down to `1966`, `90798` Antes de 1966, `105276` Sem declaração)
    *   `Clsf 240` (Idade da mãe na ocasião do parto; Axis: age, Cats: 46; `0` Total, detailed single ages `15` to `49`, cohorts `<15`, `15-19` to `45-49`, `50+`, `5413` Ignorada)
    *   `Clsf 2` (Sexo; Axis: sex, Cats: 4; `0` Total, `4` Homens, `5` Mulheres, `104539` Ignorado)

#### Tab 2683 | Deaths by Marital Status, Nature, Sex, Age & Occurrence Place (Active)
*   **Tier**: T4_LOW_PRIORITY | **Res**: Estatísticas do Registro Civil | **Subj**: Óbitos | **Per**: 2003–2024 | **Locs**: 5570 | **High Dim**: Yes
*   **Vars**: `343` (Número de óbitos ocorridos no ano; U: Px, Type: c, DK: Y), `1000343` (Deaths - % total; U: %, Type: p, DK: N)
*   **Clsfs**: 
    *   `Clsf 9832` (Estado civil; Axis: other_axis, Cats: 8; `0` Total, `99197` Casado, `99217` Separado jud., `78092` Desquitado, `78093` Divorciado, `78094` Viúvo, `78090` Solteiro, `99195` Ignorado)
    *   `Clsf 1836` (Natureza do óbito; Axis: other_axis, Cats: 5; `0` Total, `26877` Natural, `99818` Não natural, `26881` Outra, `26882` Ignorado)
    *   `Clsf 2` (Sexo; Axis: sex, Cats: 4; `0` Total, `4` Homens, `5` Mulheres, `104539` Ignorado)
    *   `Clsf 260` (Idade do falecido; Axis: age, Cats: 21; `0` Total, `<15` anos, detailed five-year cohorts up to `5996` 100+, `5997` Idade ignorada)
    *   `Clsf 257` (Local de ocorrência; Axis: other_axis, Cats: 6; `0` Total, `5832` Hospital, `5833` Domicílio, `107166` Via pública, `100296` Outro local, `100297` Ignorado)

#### Tab 367 | Deaths Registered by Deceased's Residence, Occurrence Place & Sex (1984–2002)
*   **Tier**: T4_LOW_PRIORITY | **Res**: Estatísticas do Registro Civil | **Subj**: Óbitos | **Per**: 1984–2002 | **Locs**: 5559 | **High Dim**: Yes
*   **Vars**: `224` (Número de óbitos ocorridos e registrados no ano; U: Px, Type: c, DK: Y), `1000224` (Registered deaths - % total; U: %, Type: p, DK: N)
*   **Clsfs**: 
    *   `Clsf 258` (Lugar de residência do falecido; Axis: other_axis, Cats: 68; `0` Total, breakdown of major states e.g. `5834` Rondônia, capital cities e.g. `5835` Porto Velho, Metropolitan Regions, `5896` Ignorado, `5897` Estrangeiro)
    *   `Clsf 257` (Local de ocorrência; Axis: other_axis, Cats: 4; `0` Total, `5832` Hospital, `5833` Domicílio, `100296` Outro local)
    *   `Clsf 2` (Sexo; Axis: sex, Cats: 3; `0` Total, `4` Homens, `5` Mulheres)

#### Tab 368 | Deaths Registered by Deceased's Residence, Age & Sex (1984–2002)
*   **Tier**: T4_LOW_PRIORITY | **Res**: Estatísticas do Registro Civil | **Subj**: Óbitos | **Per**: 1984–2002 | **Locs**: 5560 | **High Dim**: Yes
*   **Vars**: Same as Tab 367 (`224`, `1000224`)
*   **Clsfs**: 
    *   `Clsf 258` (Lugar de residência do falecido; Axis: other_axis, Cats: 68; same as Tab 367)
    *   `Clsf 259` (Grupos de idade da pessoa falecida; Axis: age, Cats: 21; `0` Total, `5898` <1 ano, `5899` 1-4, `5900` 5-9, five-year cohorts up to `5916` 85+, `5921` Idade ignorada)
    *   `Clsf 2` (Sexo; Axis: sex, Cats: 3)

#### Tab 704 | Live Births by Birth Place (1984–2002)
*   **Tier**: T4_LOW_PRIORITY | **Res**: Estatísticas do Registro Civil | **Subj**: Nascidos vivos | **Per**: 1984–2002 | **Locs**: 5079
*   **Vars**: `218` (U: Px, Type: c, DK: Y), `1000218` (U: %, Type: p, DK: N)
*   **Clsfs**: 
    *   `Clsf 237` (Local do nascimento; Axis: other_axis, Cats: 4; `0` Total, `5349` Hospital, `5350` Domicílio, `5351` Outro local)


