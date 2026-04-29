# Clean / Derived Datasets

All files here are transformed outputs generated in-repo (mainly `Data/data_load_util.py`).

## `sunroof_by_zip.csv`
- Granularity: ZIP.
- Source lineage: Cleaned from `Data/Sunroof/solar_by_zip.csv`.
- Columns: `region_name`, `state_name`, `yearly_sunlight_kwh_kw_threshold_avg`, `yearly_sunlight_kwh_total`, `existing_installs_count`, `percent_covered`, `count_qualified`, `number_of_panels_total`, `estimated_max_panels`.

## `census_by_zip.csv`
- Granularity: ZIP.
- Source lineage: Cleaned from `Data/Census/census_by_zip.csv`.
- Columns: `Total_Population`, `total_households`, `Median_income`, `per_capita_income`, `households_below_poverty_line`, `black_population`, `white_population`, `asian_population`, `native_population`, `hispanic_population`, `zip`.

## `grid_by_zip.csv`
- Granularity: ZIP.
- Source lineage: Cleaned from `Data/Grid/emissions_by_zip.csv`.
- Columns: `region_name`, `count_qualified`, `yearly_sunlight_kwh_total`, `carbon_offset_metric_tons`, `Grid`, `CO2e_nb_kg_per_MWh`, `carbon_offset_metric_tons_new`, `prop_diff`.

## `energy_stats_by_state.csv`
- Granularity: State.
- Source lineage: Aggregated from `Data/Grid/energy_stats_by_state.csv`.
- Columns: `State`, `State code`, `Clean`, `Bioenergy`, `Coal`, `Gas`, `Fossil`, `Solar`, `Hydro`, `Nuclear`, `Wind`, `Other Renewables`, `Other Fossil`, `Total Generation`, and matching `_prop` share columns.

## `election_by_state.csv`
- Granularity: State.
- Source lineage: Aggregated from `Data/Election/election_by_state.csv`.
- Columns: `state`, `Democrat`, `Republican`, `Total`, `Democrat_prop`, `Republican_prop`.

## `installs_by_state.csv`
- Granularity: State.
- Source lineage: Derived from EIA monthly installs table.
- Columns: `State`, `Residential_cap_24`, `Residential_cap_25`, `Residential_gen_24`, `Residential_gen_25`, `Commercial_cap_24`, `Commercial_cap_25`, `Commercial_gen_24`, `Commercial_gen_25`, `Industrial_cap_24`, `Industrial_cap_25`, `Industrial_gen_24`, `Industrial_gen_25`, `Total_cap_24`, `Total_cap_25`, `Total_gen_24`, `Total_gen_25`, `Residential_cap_prop_24`, `Residential_gen_prop_24`, `Residential_cap_prop_25`, `Residential_gen_prop_25`, `Residential_added_cap`, `Residential_added_gen`, `prop_cap_added`.

## `data_by_zip.csv`
- Granularity: ZIP integrated feature table.
- Source lineage: Joined solar + census + grid + coordinates in `make_zip_dataset`.
- Columns:
  - Core solar/census/grid IDs and metrics: `region_name`, `state_name`, `yearly_sunlight_kwh_kw_threshold_avg`, `yearly_sunlight_kwh_total`, `existing_installs_count`, `percent_covered`, `count_qualified`, `number_of_panels_total`, `estimated_max_panels`, `Total_Population`, `total_households`, `Median_income`, `per_capita_income`, `households_below_poverty_line`, `black_population`, `white_population`, `asian_population`, `native_population`, `hispanic_population`, `zip`, `carbon_offset_metric_tons`, `Grid`, `CO2e_nb_kg_per_MWh`, `carbon_offset_metric_tons_new`, `prop_diff`, `Latitude`, `Longitude`, `zip_code`.
  - Derived metrics: `panel_utilization`, `existing_installs_count_per_capita`, `carbon_offset_metric_tons_per_panel`, `carbon_offset_metric_tons_per_capita`, `carbon_offset_kg`, `carbon_offset_kg_per_panel`, `asian_prop`, `white_prop`, `black_prop`, `hispanic_prop`, `percent_below_poverty_line`.

## `data_by_state.csv`
- Granularity: State integrated feature table.
- Source lineage: Combined energy + election + demographics + incentives + install growth in `make_state_dataset`.
- Columns include:
  - IDs: `State`, `State code`.
  - Energy totals and shares: `Clean` through `Total Generation_prop`.
  - Election: `Democrat`, `Republican`, `Total`, `Democrat_prop`, `Republican_prop`.
  - Socioeconomic + solar means: `Total_Population`, `total_households`, `Median_income`, `per_capita_income`, `households_below_poverty_line`, `yearly_sunlight_kwh_kw_threshold_avg`, `existing_installs_count`, `carbon_offset_metric_tons`, `carbon_offset_metric_tons_per_panel`, `carbon_offset_metric_tons_per_capita`, `existing_installs_count_per_capita`, `panel_utilization`, `carbon_offset_kg_per_panel`, `carbon_offset_kg`.
  - Demographic totals/shares: `black_population`, `black_prop`, `white_population`, `white_prop`, `asian_population`, `asian_prop`, `hispanic_population`, `hispanic_prop`, `native_population`, `native_prop`.
  - Incentives and notes: `Net Upfront Cost (assuming $17,500 system @ $2.5 per W, federal tax credit)`, `Adjusted Payback Period (Years, under energy generation assumptions)`, `State-level Incentives`, `Numeric state-level upfront incentive`, `Notes`.
  - Install comparison metrics: all `*_24`, `*_25`, `Residential_*_prop_*`, `Residential_added_cap`, `Residential_added_gen`, `prop_cap_added`.
  - Simulation outputs: `energy_ratio_by_income_bracket`, `energy_burden_by_income_bracket`.

## `data_by_state_sum.csv`
- Granularity: State sum-based variant.
- Source lineage: `make_state_dataset(..., agg='sum')`.
- Columns: same as `data_by_state.csv` except no `energy_ratio_by_income_bracket` and `energy_burden_by_income_bracket`.

## `zips.csv`
- Granularity: ZIP.
- Source lineage: Intersection of ZIPs available in cleaned solar, census, and grid datasets.
- Columns: `zip`, `state_name`, `state_code`.

## `zips_old.csv`
- Granularity: ZIP.
- Source lineage: Legacy ZIP list retained for compatibility.
- Columns: `zip`.
