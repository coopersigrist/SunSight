# Election Datasets

## `election_by_state.csv`
- Granularity: Candidate-state-year rows.
- Source: MIT Election Data and Science Lab, U.S. President 1976-2020: https://doi.org/10.7910/DVN/42MVDX
- Columns: `year`, `state`, `state_po`, `state_fips`, `state_cen`, `state_ic`, `office`, `candidate`, `party_detailed`, `writein`, `candidatevotes`, `totalvotes`, `version`, `notes`, `party_simplified`.

## `election_by_state_cleaner.csv`
- Granularity: State rollup (project default year is 2020).
- Source lineage: Derived in `Data/data_load_util.py` (`load_election_data`).
- Columns:
  - `state`: State name.
  - `Democrat`: Democratic candidate votes.
  - `Republican`: Republican candidate votes.
  - `Total`: Total votes.
  - `Democrat_prop`: Democratic vote share.
  - `Republican_prop`: Republican vote share.
