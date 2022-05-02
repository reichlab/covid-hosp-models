library(tidyverse)
library(covidData)

#' Obtain influenza signal at daily or weekly scale
#'
#' @param pathogen the pathogen we want data for, either flu or covid
#' @param as_of Date or string in format "YYYY-MM-DD" specifying the date which
#'   the data was available on or before. If `NULL`, the default returns the
#'   most recent available data.
#' @param locations optional list of FIPS or location abbreviations. Defaults to
#'   the US, the 50 states, DC, PR, and VI.
#' @param temporal_resolution "daily" or "weekly"
#' @param source either "covidcast" or "HealthData". HealthData only supports
#' `as_of` being `NULL` or the current date.
#' @param na.rm boolean indicating whether NA values should be dropped when
#'   aggregating state-level values and calculating weekly totals. Defaults to
#'   `FALSE`
#'
#' @return data frame of incidence with columns date, location,
#'   location_name, value
load_hosp_data <- function(pathogen = c("flu", "covid"),
                           as_of = NULL,
                           locations = "*",
                           temporal_resolution = "daily",
                           source = "HealthData",
                           na.rm = FALSE) {
  library(dplyr)
  library(readr)
  library(tidyverse)

  # load location data
  location_data <- readr::read_csv(file = "data/locations.csv",
                                   show_col_types = FALSE) %>%
    dplyr::mutate(geo_value = tolower(abbreviation)) %>%
    dplyr::select(-c("population", "abbreviation"))

  # validate function arguments
  if (!(source %in% c("covidcast", "HealthData"))) {
    stop("`source` must be either covidcast or HealthData")
  } else if (source == "HealthData" && !is.null(as_of)) {
    if (as_of != Sys.Date()) {
      stop("`as_of` must be either `NULL` or the current date if source is HealthData")
    }

  }

  valid_locations <- unique(c(
    "*",
    location_data$geo_value,
    tolower(location_data$location)
  ))
  locations <-
    match.arg(tolower(locations), valid_locations, several.ok = TRUE)
  temporal_resolution <- match.arg(temporal_resolution,
                                   c("daily", "weekly"),
                                   several.ok = FALSE)

  pathogen <- match.arg(pathogen)

  if (!is.logical(na.rm)) {
    stop("`na.rm` must be a logical value")
  }

  # get geo_value based on fips if fips are provided
  if (any(grepl("\\d", locations))) {
    locations <-
      location_data$geo_value[location_data$location %in% locations]
  } else {
    locations <- tolower(locations)
  }
  # if US is included, fetch all states
  if ("us" %in% locations) {
    locations_to_fetch <- "*"
  } else {
    locations_to_fetch <- locations
  }


  # pull daily state data
  if (source == "covidcast") {

    ## this chunk retrieves data from covidcast

    signal <- ifelse(pathogen=="flu", "confirmed_admissions_influenza_1d", "confirmed_admissions_covid_1d")

    state_dat <- covidcast::covidcast_signal(
      as_of = as_of,
      geo_values = locations_to_fetch,
      data_source = "hhs",
      signal = signal,
      geo_type = "state"
    ) %>%
      dplyr::mutate(
        epiyear = lubridate::epiyear(time_value),
        epiweek = lubridate::epiweek(time_value)
      ) %>%
      dplyr::select(geo_value, epiyear, epiweek, time_value, value) %>%
      dplyr::rename(date = time_value)

  } else {

    ## this chunk retrieves data from HeatlhData

    temp <- httr::GET(
      "https://healthdata.gov/resource/qqte-vkut.json",
      config = httr::config(ssl_verifypeer = FALSE)
    ) %>%
      as.character() %>%
      jsonlite::fromJSON() %>%
      dplyr::arrange(update_date)
    csv_path <- tail(temp$archive_link$url, 1)
    data <- readr::read_csv(csv_path)

    ## value returned depends on which pathogen was specified
    if(pathogen == "flu") {
      state_dat <- data %>%
      dplyr::transmute(
        geo_value = tolower(state),
        date = date - 1,
        epiyear = lubridate::epiyear(date),
        epiweek = lubridate::epiweek(date),
        value = previous_day_admission_influenza_confirmed
      ) %>%
      dplyr::arrange(geo_value, date)
    } else {
      state_dat <- data %>%
        dplyr::transmute(
          geo_value = tolower(state),
          date = date - 1,
          epiyear = lubridate::epiyear(date),
          epiweek = lubridate::epiweek(date),
          value = previous_day_admission_adult_covid_confirmed + previous_day_admission_pediatric_covid_confirmed
        ) %>%
        dplyr::arrange(geo_value, date)
    }
  }


  # creating US and bind to state-level data if US is specified or locations
  if (locations_to_fetch == "*") {
    us_dat <- state_dat %>%
      dplyr::group_by(epiyear, epiweek, date) %>%
      dplyr::summarize(value = sum(value, na.rm = na.rm), .groups = "drop") %>%
      dplyr::mutate(geo_value = "us") %>%
      dplyr::ungroup() %>%
      dplyr::select(geo_value, epiyear, epiweek, date, value)
    # bind to daily data
    if (locations != "*") {
      dat <- rbind(us_dat, state_dat) %>%
        dplyr::filter(geo_value %in% locations)
    } else {
      dat <- rbind(us_dat, state_dat)
    }
  } else {
    dat <- state_dat
  }

  # weekly aggregation
  if (temporal_resolution != "daily") {
    dat <- dat %>%
      dplyr::group_by(epiyear, epiweek, geo_value) %>%
      dplyr::summarize(
        date = max(date),
        num_days = n(),
        value = sum(value, na.rm = na.rm),
        .groups = "drop"
      ) %>%
      dplyr::filter(num_days == 7L) %>%
      dplyr::ungroup() %>%
      dplyr::select(-"num_days")
  }
  final_data <- dat %>%
    dplyr::left_join(location_data, by = "geo_value") %>%
    dplyr::select(date, location, location_name, value) %>%
    # drop data for locations retrieved from covidcast,
    # but not included in forecasting exercise -- mainly American Samoa
    dplyr::filter(!is.na(location))

  return(final_data)
}


load_flu_hosp_data <- function(as_of = NULL,
                               locations = "*",
                               temporal_resolution = "daily",
                               source = "HealthData",
                               na.rm = FALSE) {
  
  # load location data
  location_data <- readr::read_csv(file = "data/locations.csv",
                                   show_col_types = FALSE) %>%
    dplyr::mutate(geo_value = tolower(abbreviation)) %>%
    dplyr::select(-c("population", "abbreviation"))
  
  # validate function arguments
  if (!(source %in% c("covidcast", "HealthData"))) {
    stop("`source` must be either covidcast or HealthData")
  } else if (source == "HealthData" && !is.null(as_of)) {
    if (as_of != Sys.Date()) {
      stop("`as_of` must be either `NULL` or the current date if source is HealthData")
    }
    
  }
  
  valid_locations <- unique(c(
    "*",
    location_data$geo_value,
    tolower(location_data$location)
  ))
  locations <-
    match.arg(tolower(locations), valid_locations, several.ok = TRUE)
  temporal_resolution <- match.arg(temporal_resolution,
                                   c("daily", "weekly"),
                                   several.ok = FALSE)
  if (!is.logical(na.rm)) {
    stop("`na.rm` must be a logical value")
  }
  
  # get geo_value based on fips if fips are provided
  if (any(grepl("\\d", locations))) {
    locations <-
      location_data$geo_value[location_data$location %in% locations]
  } else {
    locations <- tolower(locations)
  }
  # if US is included, fetch all states
  if ("us" %in% locations) {
    locations_to_fetch <- "*"
  } else {
    locations_to_fetch <- locations
  }
  
  # pull daily state data
  if (source == "covidcast") {
    state_dat <- covidcast::covidcast_signal(
      as_of = as_of,
      geo_values = locations_to_fetch,
      data_source = "hhs",
      signal = "confirmed_admissions_influenza_1d",
      geo_type = "state"
    ) %>%
      dplyr::mutate(
        epiyear = lubridate::epiyear(time_value),
        epiweek = lubridate::epiweek(time_value)
      ) %>%
      dplyr::select(geo_value, epiyear, epiweek, time_value, value) %>%
      dplyr::rename(date = time_value)
  } else {
    temp <- httr::GET(
      "https://healthdata.gov/resource/qqte-vkut.json",
      config = httr::config(ssl_verifypeer = FALSE)
    ) %>%
      as.character() %>%
      jsonlite::fromJSON() %>%
      dplyr::arrange(update_date)
    csv_path <- tail(temp$archive_link$url, 1)
    data <- readr::read_csv(csv_path)
    state_dat <- data %>%
      dplyr::transmute(
        geo_value = tolower(state),
        date = date - 1,
        epiyear = lubridate::epiyear(date),
        epiweek = lubridate::epiweek(date),
        value = previous_day_admission_influenza_confirmed
      ) %>%
      dplyr::arrange(geo_value, date)
  }
  
  
  # creating US and bind to state-level data if US is specified or locations
  if (locations_to_fetch == "*") {
    us_dat <- state_dat %>%
      dplyr::group_by(epiyear, epiweek, date) %>%
      dplyr::summarize(value = sum(value, na.rm = na.rm), .groups = "drop") %>%
      dplyr::mutate(geo_value = "us") %>%
      dplyr::ungroup() %>%
      dplyr::select(geo_value, epiyear, epiweek, date, value)
    # bind to daily data
    if (locations != "*") {
      dat <- rbind(us_dat, state_dat) %>%
        dplyr::filter(geo_value %in% locations)
    } else {
      dat <- rbind(us_dat, state_dat)
    }
  } else {
    dat <- state_dat
  }
  
  # weekly aggregation
  if (temporal_resolution != "daily") {
    dat <- dat %>%
      dplyr::group_by(epiyear, epiweek, geo_value) %>%
      dplyr::summarize(
        date = max(date),
        num_days = n(),
        value = sum(value, na.rm = na.rm),
        .groups = "drop"
      ) %>%
      dplyr::filter(num_days == 7L) %>%
      dplyr::ungroup() %>%
      dplyr::select(-"num_days")
  }
  
  final_data <- dat %>%
    dplyr::left_join(location_data, by = "geo_value") %>%
    dplyr::select(date, location, location_name, value) %>%
    # drop data for locations retrieved from covidcast,
    # but not included in forecasting exercise -- mainly American Samoa
    dplyr::filter(!is.na(location))
  
  return(final_data)
}


required_locations <-
  readr::read_csv(file = "./data/locations.csv") %>%
  dplyr::select("location", "abbreviation")

# The reference_date is the date of the Saturday relative to which week-ahead targets are defined.
# The forecast_date is the Monday of forecast creation.
# The forecast creation date is set to a Monday,
# even if we are delayed and create it Tuesday morning.
reference_date <- as.character(lubridate::floor_date(Sys.Date(), unit = "week") + 1)

# Load data
hosp_data <- load_hosp_data(pathogen = "covid", as_of = reference_date)

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