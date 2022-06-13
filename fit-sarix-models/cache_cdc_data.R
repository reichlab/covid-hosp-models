library(tidyverse)

# cdc_data <- readr::read_csv("data/United_States_COVID-19_Cases_and_Deaths_by_State_over_Time.csv")
cdc_data <- readr::read_csv("https://data.cdc.gov/api/views/9mfq-cb36/rows.csv?accessType=DOWNLOAD")
cdc_data$date <- lubridate::mdy(cdc_data$submission_date)

cdc_data <- cdc_data %>%
  dplyr::group_by(state) %>%
  dplyr::arrange(date) %>%
  dplyr::mutate(
    inc_cases  = as.integer(diff(c(0, tot_cases)))) %>%
  dplyr::arrange(state, date) %>%
  dplyr::select(state, date, inc_cases)

cdc_data <- dplyr::bind_rows(
  cdc_data,
  cdc_data %>%
    dplyr::group_by(date) %>%
    dplyr::summarise(inc_cases = sum(inc_cases)) %>%
    dplyr::mutate(state = "US")
)

location_info <- readr::read_csv('data/locations.csv')
cdc_data <- cdc_data %>%
  dplyr::left_join(
    location_info %>%
      dplyr::transmute(abbreviation, location, pop100k = population / 100000),
    by = c("state" = "abbreviation")) %>%
  dplyr::mutate(case_rate = inc_cases / pop100k)

cdc_data <- cdc_data %>%
  dplyr::ungroup() %>%
  dplyr::select(location, date, case_rate) %>%
  dplyr::filter(date >= "2020-10-01", !is.na(location))

reference_date <- as.character(lubridate::floor_date(Sys.Date(), unit = "week") + 1)
readr::write_csv(cdc_data, paste0('data/cdc_data_cached_', reference_date, '.csv'))
