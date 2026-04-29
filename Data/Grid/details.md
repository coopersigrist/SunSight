# Grid and Electricity Datasets

## `energy_stats_by_state.csv`
- Granularity: State-year-variable (long format).
- Source: U.S. EIA state energy data: https://www.eia.gov/state/
- Columns: `Country`, `Country code`, `State`, `State code`, `State type`, `Year`, `Category`, `Subcategory`, `Variable`, `Unit`, `Value`, `YoY absolute change`, `YoY % change`.

## `elec_rate_states_2022.csv`
- Granularity: State (2022 snapshot).
- Source: EIA electricity rate tables: https://www.eia.gov/electricity/data/state/
- Columns: `State`, `Number of Customers`, `Average Monthly Consumption (kWh)`, `Average Price (cents/kWh)`, `Average Monthly Bill (Dollar and cents)`.

## `elec_rate_zipcodes_2022.csv`
- Granularity: ZIP-utility-service type.
- Source: EIA utility-level electricity data: https://www.eia.gov/electricity/data/eia861/
- Columns: `zip`, `eiaid`, `utility_name`, `state`, `service_type`, `ownership`, `comm_rate`, `ind_rate`, `res_rate`.

## `grid_by_zip.csv`
- Granularity: ZIP to eGRID subregion mapping.
- Source: U.S. EPA eGRID ZIP lookup: https://www.epa.gov/egrid
- Columns: `ZIP (character)`, `ZIP (numeric)`, `State`, `eGRID Subregion #1`, `eGRID Subregion #2`, `eGRID Subregion #3`, `Concatenated`.

## `emissions_by_grid.csv`
- Granularity: eGRID subregion.
- Source: U.S. EPA eGRID data explorer: https://www.epa.gov/egrid/data-explorer
- Columns: `Grid`, `Grid_name`, `CO2`, `CH4`, `N2O`, `CO2e`, `Annual NO?`, `Ozone Season NO?`, `SO2`, `CO2_nb`, `CH4_nb`, `N2O_nb`, `CO2e_nb`, `Annual NO?_nb`, `Ozone Season NO?_nb`, `SO2_nb`, `grid_loss`, `net_gen(MWh)`.

## `emissions_by_zip.csv`
- Granularity: ZIP.
- Source lineage: Derived from grid + Sunroof data in `Data/Grid/carbon_calc.py`.
- Columns: `region_name`, `count_qualified`, `yearly_sunlight_kwh_total`, `carbon_offset_metric_tons`, `Grid`, `CO2e_nb_kg_per_MWh`, `carbon_offset_metric_tons_new`, `prop_diff`.
