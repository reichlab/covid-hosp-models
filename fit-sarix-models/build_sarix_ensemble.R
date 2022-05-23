library(dplyr)
library(tidyr)
library(hubEnsembles)

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
forecast_date <- as.character(lubridate::floor_date(Sys.Date(), unit = "week") + 1)

# load them back in to a single data.frame having columns required by
# build_quantile_ensemble and plot_forecasts
hosps_path <- "weekly-submission/sarix-forecasts/hosps"
models <- list.dirs(
  hosps_path,
  full.names = FALSE,
  recursive = FALSE)

forecast_exists <- purrr::map_lgl(
    models,
    function(model) {
        file.exists(file.path(hosps_path, model, paste0(forecast_date, "-", model, ".csv")))
    })
models <- models[forecast_exists]

all_components <- covidHubUtils::load_forecasts_repo(
  file_path = paste0('weekly-submission/sarix-forecasts/hosps/'),
  models = models,
  forecast_dates = forecast_date,
  locations = NULL,
  types = NULL,
  targets = NULL,
  hub = "US",
  verbose = TRUE
)
all_components <- all_components %>%
    dplyr::filter(grepl("SARIX", model))

# build ensemble via envelope
# sarix_envelope_ensemble <- all_components %>%
#     dplyr::select(forecast_date, location, horizon, temporal_resolution,
#                   target_variable, target_end_date, type, quantile, value) %>%
#     dplyr::group_by(forecast_date, location, horizon, temporal_resolution,
#                     target_variable, target_end_date, type, quantile) %>%
#     dplyr::summarize(
#         max_value = max(value),
#         median_value = median(value),
#         min_value = min(value)
#     ) %>%
#     dplyr::ungroup() %>%
#     dplyr::mutate(
#         value = ifelse(quantile > 0.5001,
#                        max_value,
#                        ifelse(quantile < 0.4999,
#                               min_value,
#                               median_value))
#     ) %>%
#     dplyr::select(-min_value, -median_value, -max_value)

# # save ensemble in hub format
# target_dir <- 'weekly-submission/sarix-forecasts/hosps/UMass-sarix_env/'
# if (!dir.exists(target_dir)) {
#     dir.create(target_dir, recursive = TRUE)
# }
# write.csv(sarix_envelope_ensemble %>% dplyr::transmute(
#   forecast_date = forecast_date,
#   target = paste(horizon, temporal_resolution, "ahead", target_variable),
#   target_end_date = target_end_date,
#   location = location,
#   type = type,
#   quantile = quantile,
#   value = value),
#   file = paste0(target_dir, forecast_date, '-UMass-sarix_env.csv'),
#   row.names = FALSE)


# build ensemble via median
# all_components <- all_components %>%
#     dplyr::filter(!grepl("p_7", model))
sarix_ensemble <- hubEnsembles::build_quantile_ensemble(
  all_components,
  forecast_date = forecast_date,
  model_name = "sarix"
)

# save ensemble in hub format
target_dir <- 'weekly-submission/sarix-forecasts/hosps/UMass-sarix/'
if (!dir.exists(target_dir)) {
    dir.create(target_dir, recursive = TRUE)
}
write.csv(sarix_ensemble %>% dplyr::transmute(
  forecast_date = forecast_date,
  target = paste(horizon, temporal_resolution, "ahead", target_variable),
  target_end_date = target_end_date,
  location = location,
  type = type,
  quantile = quantile,
  value = value),
  file = paste0(target_dir, forecast_date, '-UMass-sarix.csv'),
  row.names = FALSE)




# build ensemble via median
all_components <- all_components %>%
    dplyr::filter(!grepl("cases", model))
sarix_ensemble <- hubEnsembles::build_quantile_ensemble(
  all_components,
  forecast_date = forecast_date,
  model_name = "sarix"
)

# save ensemble in hub format
target_dir <- 'weekly-submission/sarix-forecasts/hosps/UMass-sarix_no_cases/'
if (!dir.exists(target_dir)) {
    dir.create(target_dir, recursive = TRUE)
}
write.csv(sarix_ensemble %>% dplyr::transmute(
  forecast_date = forecast_date,
  target = paste(horizon, temporal_resolution, "ahead", target_variable),
  target_end_date = target_end_date,
  location = location,
  type = type,
  quantile = quantile,
  value = value),
  file = paste0(target_dir, forecast_date, '-UMass-sarix_no_cases.csv'),
  row.names = FALSE)






# build ensemble via median after dropping forecasts based on longer history
all_components <- all_components %>%
    dplyr::filter(!grepl("p_42", model)) %>%
    dplyr::filter(!grepl("p_56", model))
sarix_ensemble <- hubEnsembles::build_quantile_ensemble(
  all_components,
  forecast_date = forecast_date,
  model_name = "sarix"
)

# save ensemble in hub format
target_dir <- 'weekly-submission/sarix-forecasts/hosps/UMass-sarix_no_cases_short/'
if (!dir.exists(target_dir)) {
    dir.create(target_dir, recursive = TRUE)
}
write.csv(sarix_ensemble %>% dplyr::transmute(
  forecast_date = forecast_date,
  target = paste(horizon, temporal_resolution, "ahead", target_variable),
  target_end_date = target_end_date,
  location = location,
  type = type,
  quantile = quantile,
  value = value),
  file = paste0(target_dir, forecast_date, '-UMass-sarix_no_cases_short.csv'),
  row.names = FALSE)
