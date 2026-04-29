# Sunroof Datasets

Source (all files): Google Project Sunroof
- Explorer: https://sunroof.withgoogle.com/data-explorer/
- CSV feed base: https://storage.googleapis.com/project-sunroof/csv/latest/
- Ingestion: `Data/Data_scraping/scrape_util.py` (`project_sunroof_scrape`).

## `solar_by_zip.csv`
- Granularity: ZIP code.
- Source URL: https://storage.googleapis.com/project-sunroof/csv/latest/project-sunroof-postal_code.csv
- Columns: `region_name`, `state_name`, `lat_max`, `lat_min`, `lng_max`, `lng_min`, `lat_avg`, `lng_avg`, `yearly_sunlight_kwh_kw_threshold_avg`, `count_qualified`, `percent_covered`, `percent_qualified`, `number_of_panels_n`, `number_of_panels_s`, `number_of_panels_e`, `number_of_panels_w`, `number_of_panels_f`, `number_of_panels_median`, `number_of_panels_total`, `kw_median`, `kw_total`, `yearly_sunlight_kwh_n`, `yearly_sunlight_kwh_s`, `yearly_sunlight_kwh_e`, `yearly_sunlight_kwh_w`, `yearly_sunlight_kwh_f`, `yearly_sunlight_kwh_median`, `yearly_sunlight_kwh_total`, `install_size_kw_buckets_json`, `carbon_offset_metric_tons`, `existing_installs_count`.

## `solar_by_city.csv`
- Granularity: City.
- Source URL: https://storage.googleapis.com/project-sunroof/csv/latest/project-sunroof-city.csv
- Columns: same as `solar_by_zip.csv`.

## `solar_by_state.csv`
- Granularity: State/territory.
- Source URL: https://storage.googleapis.com/project-sunroof/csv/latest/project-sunroof-state.csv
- Columns: same as `solar_by_zip.csv`.
