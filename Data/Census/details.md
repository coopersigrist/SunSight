# Census Datasets

## `census_by_zip.csv`
- Granularity: ZIP Code Tabulation Area (ZCTA), one row per ZCTA.
- Source: U.S. Census Bureau ACS 5-year API (2022): https://api.census.gov/data/2022/acs/acs5
- Ingestion: `Data/Data_scraping/scrape_util.py` (`get_census_info_by_zip_codes`).
- Columns:
  - `Total_Population`: Total population estimate.
  - `total_households`: Total household count.
  - `Median_income`: Median household income.
  - `per_capita_income`: Per-capita income.
  - `households_below_poverty_line`: Households below poverty level.
  - `black_population`: Black population count.
  - `white_population`: White population count.
  - `asian_population`: Asian population count.
  - `native_population`: American Indian/Alaska Native population count.
  - `hispanic_population`: Hispanic/Latino population count.
  - `zcta`: ZIP Code Tabulation Area code.
