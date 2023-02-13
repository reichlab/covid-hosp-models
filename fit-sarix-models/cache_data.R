library(tidyverse)
library(covidData)

required_locations <-
  readr::read_csv(file = "./data/locations.csv") %>%
  dplyr::select("location", "abbreviation")

# The reference_date is the date of the Saturday relative to which week-ahead targets are defined.
# The forecast_date is the Monday of forecast creation.
# The forecast creation date is set to a Monday,
# even if we are delayed and create it Tuesday morning.
reference_date <- as.character(lubridate::floor_date(Sys.Date(), unit = "week") + 1)

# Load data (by dropping the last observation)
hosp_data <- covidData::load_data(
  spatial_resolution = c("national", "state"),
  temporal_resolution = "daily",
  measure = "hospitalizations",
  drop_last_date = TRUE
  ) %>%
  dplyr::left_join(covidData::fips_codes, by = "location") %>%
  dplyr::transmute(
    date,
    location,
    location_name = ifelse(location_name == "United States", "US", location_name),
    value = inc) %>%
  dplyr::arrange(location, date)

case_data <- covidData::load_data(as_of = reference_date,
                                  spatial_resolution = c("state", "national"),
                                  temporal_resolution = "daily",
                                  measure = "cases")

data <- hosp_data %>%
    dplyr::filter(date >= "2020-10-01") %>%
    dplyr::rename(hosps = value) %>%
    dplyr::left_join(case_data %>% dplyr::transmute(location, date, cases = inc),
                     by = c('location', 'date'))

location_info <- readr::read_csv('data/locations.csv')
data <- data %>%
    dplyr::left_join(location_info %>% dplyr::transmute(location, pop100k = population / 100000)) %>%
    dplyr::mutate(hosp_rate = hosps / pop100k, case_rate = cases / pop100k)

readr::write_csv(data, paste0('data/jhu_data_cached_', reference_date, '.csv'))
