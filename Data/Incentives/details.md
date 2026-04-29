# Incentives Datasets

## `incentives_by_state.csv`
- Granularity: State.
- Source: Compiled state incentive assumptions using DSIRE and policy references: https://www.dsireusa.org/
- Columns:
  - `State`: State abbreviation.
  - `Net Upfront Cost (assuming $17,500 system @ $2.5 per W, federal tax credit)`: Modeled net upfront cost.
  - `Adjusted Payback Period (Years, under energy generation assumptions)`: Estimated payback period.
  - `State-level Incentives`: Qualitative incentive label.
  - `Numeric state-level upfront incentive`: Numeric incentive amount.
  - `Notes`: Policy/net-metering assumptions.

## `census_by_zip_complex.csv`
- Granularity: ZCTA.
- Source lineage: Derived from Census ACS plus region/division joins.
- Columns:
  - `Income: < $10,000` through `Income: $200,000+`: Household counts by income bin.
  - `total_households`, `Median_income`, `per_capita_income`, `households_below_poverty_line`: Socioeconomic fields.
  - `black_population`, `white_population`, `asian_population`, `native_population`, `hispanic_population`: Demographic counts.
  - `zcta`, `state_abbr`, `region`, `division`: Geography keys.

## `agent_average_bracket_energy_ratio.csv`
- Granularity: State.
- Source lineage: Model output from incentive simulation.
- Columns: `State`, `Data` (serialized income-bracket energy ratio mapping).

## `agent_average_bracket_energy_burden.csv`
- Granularity: State.
- Source lineage: Model output from incentive simulation.
- Columns: `State`, `Data` (serialized income-bracket energy burden mapping).
