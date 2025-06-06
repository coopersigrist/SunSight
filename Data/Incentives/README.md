The following is data that is included to calculate the incentive adoption likelihood per zipcode. 

The data included is as follows:

1. census_by_zip_complex.csv contains American Housing Survey data for every zipcode, with the number of households in different income bins, as well as other demographic data
2.  EIA_Annual_Household_Site_Fuel_Consumption.csv is data scraped from the following tables produced by the EIA:
https://www.eia.gov/consumption/residential/data/2020/c&e/pdf/ce2.5.pdf
https://www.eia.gov/consumption/residential/data/2020/c&e/pdf/ce2.4.pdf
https://www.eia.gov/consumption/residential/data/2020/c&e/pdf/ce2.3.pdf
https://www.eia.gov/consumption/residential/data/2020/c&e/pdf/ce2.2.pdf

The table contains average and RSE energy consumption per household per income level for the four different regions of the US (Northeast, West, South, Midwest)

4. elec_rate_zipcodes_2022.csv contains the electricity prices for residential areas in 2022 for each zipcode

5. incentives_by_state.csv contains existing incentives for each state 

6. CSI training data contains data from the California Solar Initiative. This data contains fine-graiend data about San Diego County. This is used for acceptance model 2 (which is based off of Zhang et.al's logistic regression model of incentive acceptance) (To Be Implemented)
