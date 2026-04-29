# EIA Datasets

## `small_scale_solar_by_state_by_month.csv`
- Granularity: State-month.
- Source: EIA Form 861M monthly distributed solar data: https://www.eia.gov/electricity/data/eia861m/
- Columns:
  - `Year`, `Month`, `State`: Time and geography IDs.
  - `Residential_cap`, `Commercial_cap`, `Industrial_cap`, `Total_cap`: Capacity by sector and total.
  - `Residential_gen`, `Commercial_gen`, `Industrial_gen`, `Total_gen`: Generation by sector and total.
  - `gen_per_cap`: Derived generation-per-capita metric.

## `jan_24_25_by_state.csv`
- Granularity: State (Jan 2024 vs Jan 2025 comparison).
- Source lineage: Derived from `small_scale_solar_by_state_by_month.csv` in `Data/data_load_util.py`.
- Columns: `State`, `Residential_cap_24`, `Residential_cap_25`, `Residential_gen_24`, `Residential_gen_25`, `Commercial_cap_24`, `Commercial_cap_25`, `Commercial_gen_24`, `Commercial_gen_25`, `Industrial_cap_24`, `Industrial_cap_25`, `Industrial_gen_24`, `Industrial_gen_25`, `Total_cap_24`, `Total_cap_25`, `Total_gen_24`, `Total_gen_25`, `Residential_cap_prop_24`, `Residential_gen_prop_24`, `Residential_cap_prop_25`, `Residential_gen_prop_25`, `Residential_added_cap`, `Residential_added_gen`, `prop_cap_added`.

## `eia_household_elec_consumption.csv`
- Granularity: Census region by income bracket.
- Source: EIA Residential Energy Consumption Survey (RECS): https://www.eia.gov/consumption/residential/data/
- Columns:
  - `Region`: Census region.
  - `*_elec`: Electricity use estimate for each income bracket.
  - `*_RSE`: Relative standard error of estimate.
  - `*_count_millions`: Household counts (millions) per income bracket.
