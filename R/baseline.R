# adapted from weekly-submission/fit_baseline_model.R
library(covidcast)
library(epitools)
library(dplyr)
library(tidyr)
library(ggplot2)
library(simplets)
library(covidHubUtils)
library(hubEnsembles)
library(ggforce)
# library(here)
# setwd(here())
source("./R/fit_baseline_one_location.R")

# Set locations and quantiles
required_quantiles <-
  c(0.01, 0.025, seq(0.05, 0.95, by = 0.05), 0.975, 0.99)
required_locations <-
  readr::read_csv(file = "./data/locations.csv") %>%
  dplyr::select("location", "abbreviation")
# The reference_date is the date of the Saturday relative to which week-ahead targets are defined.
# The forecast_date is the Monday of forecast creation.
# The forecast creation date is set to a Monday,
# even if we are delayed and create it Tuesday morning.
reference_date <- Sys.Date()
  #as.character(lubridate::floor_date(Sys.Date(), unit = "week") - 1)
forecast_date <- reference_date# as.character(as.Date(reference_date) + 2)
# Load data
data <- covidData::load_data(
  spatial_resolution = c("national", "state"),
  temporal_resolution = "daily",
  measure = "hospitalizations",
  drop_last_date = FALSE
) %>%
  dplyr::left_join(covidData::fips_codes, by = "location") %>%
  dplyr::transmute(
    date,
    location,
    location_name = ifelse(location_name == "United States", "US", location_name),
    value = inc) %>%
  dplyr::arrange(location, date) %>%
  # the previous lines reproduce the output of the current `load_hosp_data` function
  # the following lines currently follow the call to `load_hosp_data` in baseline.R
  dplyr::filter(date >= as.Date("2020-09-01")) %>%
  dplyr::filter(location != "60") %>%
  dplyr::left_join(required_locations, by = "location") %>%
  dplyr::mutate(geo_value = tolower(abbreviation)) %>%
  dplyr::select(geo_value, time_value = date, value)

location_number <- length(required_locations$abbreviation)
# set variation of baseline to fit
transformation_variation <- c("none", "sqrt")
symmetrize_variation <- c(TRUE, FALSE)
window_size_variation <- c(14, 21, 28)


# fit baseline models
reference_date <- lubridate::ymd(reference_date)
quantile_forecasts <-
  purrr::map_dfr(required_locations$abbreviation,
                 function(location) {
                   print(paste("location", location))
                   data <- data %>%
                     dplyr::filter(geo_value == tolower(location))
                   location_results <-
                     fit_baseline_one_location(
                       reference_date = reference_date,
                       location_data = data,
                       transformation = transformation_variation,
                       symmetrize = symmetrize_variation,
                       window_size = window_size_variation,
                       taus = required_quantiles,
                       temporal_resolution = "daily",
                       target_name = "day ahead inc hosp"
                     )

                 }) %>%
  dplyr::left_join(required_locations, by = "abbreviation") %>%
  dplyr::select(forecast_date,
                target,
                target_end_date,
                location,
                type,
                quantile,
                value,
                model)

model_number <- length(unique(quantile_forecasts$model))
model_names <- c(unique(quantile_forecasts$model), "trends_ensemble")

## create the directories if they don't exist
for(i in 1:length(model_names)) {
  if(!dir.exists(paste0('./weekly-submission/forecasts','/UMass-',model_names[i],'/')))
    dir.create(paste0('./weekly-submission/forecasts','/UMass-',model_names[i],'/'))
  if(!dir.exists(paste0('./weekly-submission/baseline-plots','/UMass-',model_names[i],'/')))
    dir.create(paste0('./weekly-submission/baseline-plots','/UMass-',model_names[i],'/'))
}

model_folders <-
  paste0('/UMass-',
         model_names,
         '/',
         forecast_date,
         '-UMass-',
         model_names)
results_paths <-
  paste0('weekly-submission/forecasts', model_folders, '.csv')
plot_paths <-
  paste0('weekly-submission/baseline-plots', model_folders, '.pdf')

# save all the baseline models in hub format
for (i in 1:model_number) {
  model_name <- model_names[i]
  # set path
  results_path <- results_paths[i]
  model_forecasts <-  quantile_forecasts %>%
    dplyr::filter(model == model_name) %>%
    dplyr::select(-"model")
  write.csv(model_forecasts, file = results_path, row.names = FALSE)
}

# load them back in to a single data.frame having columns required by
# build_quantile_ensemble and plot_forecasts
all_baselines <- covidHubUtils::load_forecasts_repo(
  file_path = paste0('weekly-submission/forecasts/'),
  models = paste0('UMass-', model_names[1:model_number]),
  forecast_dates = forecast_date,
  locations = NULL,
  types = NULL,
  targets = NULL,
  hub = "US",
  verbose = TRUE
)

# build ensemble
trends_ensemble <- hubEnsembles::build_quantile_ensemble(
  all_baselines,
  forecast_date = forecast_date,
  model_name = "trends_ensemble"
)

# save ensemble in hub format
write.csv(trends_ensemble %>% dplyr::transmute(
  forecast_date = forecast_date,
  target = paste(horizon, temporal_resolution, "ahead", target_variable),
  target_end_date = target_end_date,
  location = location,
  type = type,
  quantile = quantile,
  value = value),
  file = results_paths[model_number + 1],
  row.names = FALSE)

# load ensemble back in and bind with other baselines for plotting
all_baselines <- dplyr::bind_rows(
  all_baselines,
  covidHubUtils::load_forecasts_repo(
  file_path = paste0('weekly-submission/forecasts/'),
  models = paste0('UMass-', model_names[model_number + 1]),
  forecast_dates = forecast_date,
  locations = NULL,
  types = NULL,
  targets = NULL,
  hub = "US",
  verbose = TRUE
)
)

truth_for_plotting <- load_truth(
  truth_source = "HealthData",
  target_variable = "inc hosp",
  data_location = "covidData"
)

for (i in 1:(model_number + 1)) {
  # plot
  plot_path <- plot_paths[i]
  p <-
    covidHubUtils::plot_forecasts(
      forecast_data = all_baselines %>%
        dplyr::filter(model == paste0('UMass-',model_names[i])),
      facet = "~location",
      hub = "US",
      truth_data = truth_for_plotting %>%
        dplyr::filter(target_end_date >= reference_date - (7 * 32) &
                        target_end_date <= reference_date + 28),
      truth_source = "HealthData",
      fill_transparency = .5,
      top_layer = "forecast",
      subtitle = "none",
      title = "none",
      show_caption = FALSE,
      plot = FALSE
    ) +
    scale_x_date(
      breaks = "1 month",
      date_labels = "%b-%y",
      limits = as.Date(c(
        reference_date - (7 * 32), reference_date + 28
      ), format = "%b-%y")
    ) +
    theme_update(
      legend.position = "bottom",
      legend.direction = "vertical",
      legend.text = element_text(size = 8),
      legend.title = element_text(size = 8),
      axis.text.x = element_text(angle = 90),
      axis.title.x = element_blank()
    ) +
    ggforce::facet_wrap_paginate(
      ~ location,
      scales = "free",
      ncol = 2,
      nrow = 3,
      page = 1
    )
  n <- n_pages(p)
  pdf(
    plot_path,
    paper = 'A4',
    width = 205 / 25,
    height = 270 / 25
  )
  for (i in 1:n) {
    suppressWarnings(print(
      p + ggforce::facet_wrap_paginate(
        ~ location,
        scales = "free",
        ncol = 2,
        nrow = 3,
        page = i
      )
    ))
  }
  dev.off()
}


