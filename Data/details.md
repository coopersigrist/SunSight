# Data Folder Datasets

This file documents datasets stored directly in `Data/` (not in subfolders).

## `census_by_zip.csv`
- Granularity: ZIP Code Tabulation Area (ZCTA), one row per ZCTA.
- Source: U.S. Census Bureau ACS 5-year API (2022): https://api.census.gov/data/2022/acs/acs5
- Columns: `Total_Population`, `total_households`, `Median_income`, `per_capita_income`, `households_below_poverty_line`, `black_population`, `white_population`, `asian_population`, `native_population`, `zcta`.

## `small_scale_solar_by_state_by_month.csv`
- Granularity: State-month, one row per state and month.
- Source: U.S. EIA Form 861M: https://www.eia.gov/electricity/data/eia861m/
- Columns: `Year`, `Month`, `State`, `Residential_cap`, `Commercial_cap`, `Industrial_cap`, `Total_cap`, `Residential_gen`, `Commercial_gen`, `Industrial_gen`, `Total_gen`, `gen_per_cap`.
